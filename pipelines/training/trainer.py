from __future__ import annotations

import gc

import torch

from tools.monitoring.inspection import ModelSummary
from tools.training.checkpoint   import Checkpoint
from tools.training.gradients    import GradientClipper
from tools.training.scheduling   import Scheduler, Warmup
from tools.training.stopping     import EarlyStopping

from pipelines.training.loss import Loss


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

        self.epochs               = training_config.loop.epochs
        self.accumulation_steps   = training_config.loop.gradient_accumulation_steps
        self.use_amp              = training_config.loop.use_amp and torch.cuda.is_available()
        self.scaler               = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.optimizer = torch.optim.AdamW(self._build_param_groups(), betas=training_config.optimizer.betas, eps=training_config.optimizer.eps)
        base_lrs       = [group["lr"] for group in self.optimizer.param_groups]

        self.warmup           = Warmup(training_config, self.logger, self.tracker)
        self.scheduler        = Scheduler(base_lrs, self.warmup, training_config, self.logger, self.tracker)
        self.early_stopping   = EarlyStopping(training_config, self.logger, self.tracker)
        self.gradient_clipper = GradientClipper(training_config, self.logger, self.tracker)
        self.criterion        = Loss(training_config.loss, self.stats, self.logger, self.tracker)
        self.checkpoint       = Checkpoint(self.logger, self.tracker, str(run_metadata.checkpoint_path))

        self.scheduler.set_total_epochs(self.epochs)

        self.global_step   = 0
        self.train_losses  = []
        self.val_losses    = []

        if not self.tuning_mode:
            summary = ModelSummary(self.logger, self.model)
            summary.run()
            summary.save_markdown(run_metadata.metadata_directory / "model_summary.md")

    def _build_param_groups(self):
        regression_head_parameters = list(self.model.regression_head.parameters())
        regression_head_parameters += list(self.model.norm.parameters())
        pool_parameters            = list(self.model.pool.parameters())

        pool_identifiers   = {id(parameter) for parameter in pool_parameters}
        encoder_parameters = [parameter for parameter in self.model.encoder.parameters() if id(parameter) not in pool_identifiers]

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

    def capture_state(self, epoch) -> dict:
        return {
            "params"        : self.model.state_dict(),
            "optimizer"     : self.optimizer.state_dict(),
            "scaler"        : self.scaler.state_dict() if self.scaler else None,
            "epoch"         : epoch,
            "global_step"   : self.global_step,
            "train_losses"  : self.train_losses,
            "val_losses"    : self.val_losses,
            "stats"         : self.stats.as_dict(),
        }

    def forward(self, data):
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predictions = self.model(data)
            loss_dict   = self.criterion(predictions, data, self.global_step)
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
        self._apply_learning_rates()

    def train_epoch(self, loader, epoch):
        self.model.train()

        total_loss        = 0.0
        number_of_batches = len(loader)

        with self.logger.track() as progress:
            task_id = progress.add_task(f"Epoch {epoch + 1}/{self.epochs} train", total=number_of_batches)

            for batch_index, data in enumerate(loader):
                data         = data.to(self.device, non_blocking=True)
                _, loss_dict = self.forward(data)
                loss         = loss_dict["total_loss"] / self.accumulation_steps

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at batch {batch_index}, step {self.global_step}.")

                do_step     = (batch_index + 1) % self.accumulation_steps == 0 or (batch_index + 1) >= number_of_batches
                self.backward(loss, do_step)
                total_loss += loss_dict["total_loss"].item()
                progress.advance(task_id)

        return total_loss / max(1, number_of_batches)

    def evaluate(self, loader, epoch, stage="validation"):
        self.model.eval()

        total_loss        = 0.0
        number_of_batches = 0

        with torch.no_grad():
            for data in loader:
                data            = data.to(self.device, non_blocking=True)
                _, loss_dict    = self.forward(data)
                total_loss     += loss_dict["total_loss"].item()
                number_of_batches += 1

        average_loss = total_loss / max(1, number_of_batches)
        self.tracker.log_scalar(f"{stage}/loss", average_loss, epoch)
        return {"avg_loss": average_loss}

    def run_epoch(self, train_loader, val_loader, epoch):
        self.scheduler.step(epoch)
        self._apply_learning_rates()

        train_loss      = self.train_epoch(train_loader, epoch)
        validation_loss = self.evaluate(val_loader, epoch, stage="validation")["avg_loss"]
        return train_loss, validation_loss

    def train(self, train_loader, val_loader, test_loader):
        self.logger.section("[Training Loop]")
        self.logger.subsection(f"train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}")

        self._clear_memory()
        self.optimizer.zero_grad()
        self._apply_learning_rates()

        for epoch in range(self.epochs):
            train_loss, validation_loss = self.run_epoch(train_loader, val_loader, epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(validation_loss)

            self.tracker.log_metrics("loss_comparison", {"train": train_loss, "val": validation_loss}, step=epoch)
            self.logger.subsection(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={validation_loss:.4f}")

            self.checkpoint.step(validation_loss, epoch, self)

            self._clear_memory()

            if self.early_stopping(validation_loss, epoch):
                break

        if self.config.early_stopping.restore_best:
            self.checkpoint.restore_best(self.model, self.device)

        validation_loss = self.evaluate(val_loader, self.epochs, stage="final_validation")["avg_loss"]
        test_loss       = self.evaluate(test_loader, self.epochs, stage="final_test")["avg_loss"]

        return {
            "best_val_loss"         : self.checkpoint.best_val_loss,
            "best_epoch"            : self.checkpoint.best_epoch,
            "final_validation_loss" : validation_loss,
            "final_test_loss"       : test_loss,
            "checkpoint_path"       : str(self.run_metadata.checkpoint_path),
            "run_directory"         : str(self.run_metadata.run_directory),
            "train_losses"          : self.train_losses,
            "val_losses"            : self.val_losses,
        }
