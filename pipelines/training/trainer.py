from __future__ import annotations

import gc

import torch

from tools.monitoring.inspection import ModelSummary
from tools.training.checkpoint   import Checkpoint
from tools.training.ema          import ExponentialMovingAverage
from tools.training.gradients    import GradientClipper
from tools.training.scheduling   import Scheduler, Warmup
from tools.training.stopping     import EarlyStopping

from pipelines.training.loss    import Loss
from pipelines.training.metrics import Metrics


class Trainer:
    def __init__(self, model, stats, training_config, run_metadata):
        self.config       = training_config
        self.run_metadata = run_metadata
        self.logger       = run_metadata.logger
        self.tracker      = run_metadata.tracker
        self.tuning_mode  = training_config.loop.tuning_mode

        self.device = training_config.loop.device if torch.cuda.is_available() else "cpu"
        self.model  = model.to(self.device)
        self.stats  = stats

        self.target_mean = torch.tensor(stats.target_mean, dtype=torch.float32, device=self.device)
        self.target_std  = torch.tensor(stats.target_std,  dtype=torch.float32, device=self.device)

        self.epochs               = training_config.loop.epochs
        self.validation_frequency = training_config.loop.validation_frequency
        self.accumulation_steps   = training_config.loop.gradient_accumulation_steps
        self.use_amp              = training_config.loop.use_amp and torch.cuda.is_available()
        self.scaler               = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.optimizer = torch.optim.AdamW(self._build_param_groups(), betas=training_config.optimizer.betas, eps=training_config.optimizer.eps)
        base_lrs       = [group["lr"] for group in self.optimizer.param_groups]

        self.warmup           = Warmup(training_config, self.logger, self.tracker)
        self.scheduler        = Scheduler(base_lrs, self.warmup, training_config, self.logger, self.tracker)
        self.early_stopping   = EarlyStopping(training_config, self.logger, self.tracker)
        self.ema              = ExponentialMovingAverage(self.model, training_config, self.logger, self.tracker)
        self.gradient_clipper = GradientClipper(training_config, self.logger, self.tracker)
        self.criterion        = Loss(training_config.loss, self.logger, self.tracker)
        self.metrics          = Metrics(training_config.loop.verbose, self.logger, self.tracker)
        self.checkpoint       = Checkpoint(self.logger, self.tracker, str(run_metadata.checkpoint_path))

        self.scheduler.set_total_epochs(self.epochs)

        self.best_val_loss = float("inf")
        self.best_epoch    = -1
        self.best_metrics  = {}
        self.global_step   = 0
        self.train_losses  = []
        self.val_losses    = []

        if not self.tuning_mode:
            summary = ModelSummary(self.logger, self.model)
            summary.run()
            summary.save_markdown(run_metadata.metadata_directory / "model_summary.md")

    def _build_param_groups(self):
        regression_head_parameters = list(self.model.regression_head.parameters())
        pool_parameters            = list(self.model.pool.parameters())

        pool_identifiers   = {id(parameter) for parameter in pool_parameters}
        encoder_parameters = [parameter for parameter in self.model.encoder.parameters() if id(parameter) not in pool_identifiers]
        encoder_parameters += list(self.model.norm.parameters())

        optimizer_config = self.config.optimizer
        groups = [
            {"params": regression_head_parameters, "lr": optimizer_config.learning_rate_regression_head, "weight_decay": optimizer_config.weight_decay_regression_head, "name": "regression_head"},
            {"params": pool_parameters,            "lr": optimizer_config.learning_rate_pool,            "weight_decay": optimizer_config.weight_decay_pool,            "name": "pool"},
            {"params": encoder_parameters,         "lr": optimizer_config.learning_rate_encoder,         "weight_decay": optimizer_config.weight_decay_encoder,         "name": "encoder"},
        ]

        self.logger.section("[Optimizer Parameter Groups]")
        for group in groups:
            parameter_count = sum(parameter.numel() for parameter in group["params"])
            self.logger.subsection(f"{group['name']}: lr={group['lr']} weight_decay={group['weight_decay']} parameters={parameter_count:,}")

        return groups

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_learning_rates(self):
        effective_learning_rates = self.scheduler.effective_lrs()
        for group, learning_rate in zip(self.optimizer.param_groups, effective_learning_rates):
            group["lr"] = learning_rate

    def denormalize_targets(self, target):
        return target * self.target_std + self.target_mean

    def capture_state(self, epoch) -> dict:
        return {
            "params"        : self.model.state_dict(),
            "optimizer"     : self.optimizer.state_dict(),
            "ema"           : self.ema.state_dict(),
            "scaler"        : self.scaler.state_dict() if self.scaler else None,
            "epoch"         : epoch,
            "global_step"   : self.global_step,
            "train_losses"  : self.train_losses,
            "val_losses"    : self.val_losses,
            "best_metrics"  : self.best_metrics,
            "stats"         : self.stats.as_dict(),
        }

    def forward(self, data):
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predictions = self.model(data)
            loss_dict   = self.criterion(predictions, data.y, self.global_step)
        return predictions, loss_dict

    def backward(self, loss, do_step: bool):
        if self.scaler:
            self.scaler.scale(loss).backward()
            if do_step:
                self.scaler.unscale_(self.optimizer)
                gradient_norm = self.gradient_clipper.maybe_clip(self.model, self.global_step)
                self.gradient_clipper.record(gradient_norm, self.global_step)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._finish_step()
        else:
            loss.backward()
            if do_step:
                gradient_norm = self.gradient_clipper.maybe_clip(self.model, self.global_step)
                self.gradient_clipper.record(gradient_norm, self.global_step)
                self.optimizer.step()
                self._finish_step()

    def _finish_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.warmup.step()
        self.global_step += 1
        self.ema.update(self.model, step=self.global_step)
        self._apply_learning_rates()

    def train_epoch(self, loader, epoch):
        self.model.train()

        with self.logger.track() as progress:
            task_id          = progress.add_task(f"Epoch {epoch + 1}/{self.epochs} train", total=len(loader))
            number_of_batches = len(loader)

            for batch_index, data in enumerate(loader):
                data                = data.to(self.device, non_blocking=True)
                _, loss_dict        = self.forward(data)
                loss                = loss_dict["total_loss"] / self.accumulation_steps

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at batch {batch_index}, step {self.global_step}.")

                do_step = (batch_index + 1) % self.accumulation_steps == 0 or (batch_index + 1) >= number_of_batches
                self.backward(loss, do_step)
                progress.advance(task_id)

    def evaluate(self, loader, epoch, stage="validation"):
        self.model.eval()
        self.ema.apply_to(self.model)

        total_loss          = 0.0
        number_of_batches   = 0
        prediction_segments = []
        target_segments     = []

        try:
            with torch.no_grad():
                for data in loader:
                    data                = data.to(self.device, non_blocking=True)
                    predictions, loss_dict = self.forward(data)
                    total_loss         += loss_dict["total_loss"].item()
                    number_of_batches  += 1
                    prediction_segments.append(predictions.detach().float().cpu())
                    target_segments.append(data.y.detach().float().cpu())
        finally:
            self.ema.restore(self.model)

        average_loss            = total_loss / max(1, number_of_batches)
        predictions             = torch.cat(prediction_segments)
        targets                 = torch.cat(target_segments)
        denormalized_predictions = self.denormalize_targets(predictions.to(self.device)).cpu()
        denormalized_targets     = self.denormalize_targets(targets.to(self.device)).cpu()
        metrics                  = self.metrics.calculate(epoch, denormalized_predictions, denormalized_targets, stage=stage)

        return {"avg_loss": average_loss, "metrics": metrics, "predictions": denormalized_predictions, "targets": denormalized_targets}

    def train(self, train_loader, val_loader, test_loader):
        self.logger.section("[Training Loop]")
        self.logger.subsection(f"train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}")

        self._clear_memory()
        self.optimizer.zero_grad()
        self._apply_learning_rates()

        train_results = None

        for epoch in range(self.epochs):
            self.scheduler.step(epoch)
            self._apply_learning_rates()

            self.train_epoch(train_loader, epoch)

            validation_results = self.evaluate(val_loader, epoch, stage="validation")

            if epoch % self.validation_frequency == 0 or epoch + 1 == self.epochs or train_results is None:
                train_results = self.evaluate(train_loader, epoch, stage="train")

            self.train_losses.append(train_results["avg_loss"])
            self.val_losses.append(validation_results["avg_loss"])

            self.tracker.log_metrics("loss_comparison", {"train": train_results["avg_loss"], "val": validation_results["avg_loss"]}, step=epoch)
            self.logger.subsection(f"Epoch {epoch + 1}: train_loss={train_results['avg_loss']:.4f} val_loss={validation_results['avg_loss']:.4f}")

            if self.checkpoint.step(validation_results["avg_loss"], epoch, self):
                self.best_val_loss = float(validation_results["avg_loss"])
                self.best_epoch    = epoch
                self.best_metrics  = dict(validation_results["metrics"])

            self._clear_memory()

            if self.early_stopping(validation_results["avg_loss"], epoch):
                break

        self.checkpoint.restore_best(self.model, self.device)

        train_final      = self.evaluate(train_loader, self.epochs, stage="final_train")
        validation_final = self.evaluate(val_loader,   self.epochs, stage="final_validation")
        test_final       = self.evaluate(test_loader,  self.epochs, stage="final_test")

        return train_final, validation_final, test_final
