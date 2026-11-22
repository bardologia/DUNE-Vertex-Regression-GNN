from __future__ import annotations

import gc
import math
import time
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter

from configuration.data.general        import CoordinateAxes
from configuration.shared              import IoNames
from configuration.training.general    import LossSettings, TrainerSettings
from tools.monitoring.inspection       import ModelSummary
from tools.monitoring.resource_monitor import ResourceMonitor
from tools.runtime.run_tag             import CompletionMarker
from tools.training.checkpoint         import Checkpoint, TrainerState, WeightEma
from tools.training.gradients          import GradientClipper
from tools.training.scheduling         import LearningRateScaler, Scheduler, Warmup
from tools.training.stopping           import EarlyStopping
from tools.training.vram_reservation   import VramReservation

from pipelines.shared.coordinate_errors import CoordinateErrors


class Loss:
    def __init__(self, loss_config, stats, logger):
        self.config = loss_config
        self.stats  = stats
        self.logger = logger

        self.logger.section("[Loss Function]")
        self.logger.kv_table({
            "Data term"            : f"{self.config.data_term} (weight {self.config.data_weight})",
            "Euclidean weight"     : self.config.euclidean_weight,
            "Light falloff weight" : self.config.light_falloff_weight,
        })

    def _elementwise(self, predictions, targets):
        if self.config.data_term == "mse":
            return (predictions - targets) ** 2
        if self.config.data_term == "huber":
            return F.huber_loss(predictions, targets, delta=self.config.huber_delta, reduction="none")

    def _data_term(self, predictions, targets, scored):
        return (self._elementwise(predictions, targets) * scored).sum() / scored.sum()

    def _euclidean_term(self, predictions, targets, scored):
        return torch.sqrt((((predictions - targets) ** 2) * scored).sum(dim=1) + LossSettings.epsilon).mean()

    def _light_falloff_term(self, predictions, data):
        device         = predictions.device
        physical_nodes = self.stats.node.inverse_torch(data.x, device)
        positions      = physical_nodes[:, 0:3]
        light          = physical_nodes[:, 3]

        physical_vertex = self.stats.target.inverse_torch(predictions, device)
        vertex_per_node = physical_vertex[data.batch]

        squared_distance    = ((positions - vertex_per_node) ** 2).sum(dim=1) + self.config.light_falloff_epsilon
        predicted_intensity = 1.0 / squared_distance

        light_mean     = scatter(light, data.batch, reduce="mean")[data.batch]
        intensity_mean = scatter(predicted_intensity, data.batch, reduce="mean")[data.batch]

        covariance     = scatter((light - light_mean) * (predicted_intensity - intensity_mean), data.batch, reduce="mean")
        light_variance = scatter((light - light_mean) ** 2, data.batch, reduce="mean")
        intensity_var  = scatter((predicted_intensity - intensity_mean) ** 2, data.batch, reduce="mean")

        correlation = covariance / torch.sqrt(light_variance * intensity_var + LossSettings.epsilon)
        return (1.0 - correlation).mean()

    def __call__(self, predictions, data):
        targets = data.y

        data_term  = self._data_term(predictions, targets, data.scored)
        total_loss = self.config.data_weight * data_term

        if self.config.euclidean_weight > 0.0:
            physical_predictions = self.stats.target.inverse_torch(predictions, predictions.device)
            physical_targets     = self.stats.target.inverse_torch(targets, predictions.device)

            euclidean  = self._euclidean_term(physical_predictions, physical_targets, data.scored)
            total_loss = total_loss + self.config.euclidean_weight * euclidean

        if self.config.light_falloff_weight > 0.0 and hasattr(data, "batch") and data.batch is not None:
            light_falloff = self._light_falloff_term(predictions, data)
            total_loss    = total_loss + self.config.light_falloff_weight * light_falloff

        return {"total_loss": total_loss}


class OptimizerOverrideApplier:
    def __init__(self, training_config, model_config, explicit_paths, logger):
        self.optimizer_config = training_config.optimizer
        self.overrides        = dict(model_config.optimizer_overrides)
        self.explicit_paths   = set(explicit_paths or ())
        self.logger           = logger

        self.valid_fields = {definition.name for definition in fields(self.optimizer_config)}

    def _validate(self) -> None:
        unknown = [key for key in self.overrides if key not in self.valid_fields]
        if unknown:
            raise ValueError(f"Model optimizer_overrides reference unknown OptimizerConfig field(s) {unknown}; valid fields: {sorted(self.valid_fields)}.")

    def _apply(self) -> tuple[dict, dict]:
        applied = {}
        skipped = {}

        for key, value in self.overrides.items():
            if f"training.optimizer.{key}" in self.explicit_paths:
                skipped[key] = getattr(self.optimizer_config, key)
                continue

            setattr(self.optimizer_config, key, value)
            applied[key] = value

        return applied, skipped

    def _report(self, applied, skipped) -> None:
        self.logger.section("[Per-Model Optimizer Overrides]")

        if applied:
            self.logger.kv_table(applied)

        for key, kept_value in skipped.items():
            self.logger.subsection(f"{key}: kept command-line value {kept_value}, model override not applied")

    def run(self) -> None:
        if not self.overrides:
            return

        self._validate()
        applied, skipped = self._apply()
        self._report(applied, skipped)


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

        self.epochs                  = training_config.loop.epochs
        self.validation_frequency    = training_config.loop.validation_frequency
        self.accumulation_steps      = training_config.loop.gradient_accumulation_steps
        self.abort_on_nonfinite_loss = training_config.loop.abort_on_nonfinite_loss
        self.use_amp                 = training_config.loop.use_amp and str(self.device).startswith("cuda")
        self.scaler                  = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.learning_rates = LearningRateScaler(training_config.optimizer, training_config.loop)

        self.optimizer = torch.optim.AdamW(self._build_param_groups(), betas=training_config.optimizer.betas, eps=training_config.optimizer.eps)
        base_lrs       = [group["lr"] for group in self.optimizer.param_groups]

        self.warmup           = Warmup(training_config, self.logger, self.tracker)
        self.scheduler        = Scheduler(base_lrs, self.warmup, training_config, self.logger, self.tracker)
        self.early_stopping   = EarlyStopping(training_config, self.logger, self.tracker)
        self.gradient_clipper = GradientClipper(training_config, self.logger, self.tracker, param_groups=self.optimizer.param_groups)
        self.criterion        = Loss(training_config.loss, self.stats, self.logger)
        self.checkpoint       = Checkpoint(self.logger, self.tracker, str(run_metadata.checkpoint_path), min_delta=training_config.early_stopping.min_delta)
        self.ema              = WeightEma(self.model, training_config.loop.ema_decay, training_config.loop.use_ema)
        self.resource_monitor = ResourceMonitor(config=training_config.resources, logger=self.logger, tracker=self.tracker, step_getter=lambda: self.global_step)
        self.vram_reservation = VramReservation(training_config.memory.reserve_vram, training_config.memory.vram_keep_free_gb, self.device, self.logger)

        self.resume     = training_config.loop.resume
        self.state_path = TrainerState.path(run_metadata.run_directory)

        self.global_step       = 0
        self.pending_backwards = 0
        self.train_losses      = []
        self.val_losses        = []

        if not self.tuning_mode:
            self.logger.section("[Training Configuration]")
            self.logger.kv_table({
                "Device"            : self.device,
                "AMP"               : self.use_amp,
                "Epochs"            : self.epochs,
                "Validate every"    : self.validation_frequency,
                "Grad accumulation" : self.accumulation_steps,
                "EMA"               : f"{training_config.loop.use_ema} (decay {training_config.loop.ema_decay})",
                "Resume"            : self.resume,
                "LR (encoder)"      : self.learning_rates.encoder,
                "LR (pool)"         : self.learning_rates.pool,
                "LR (head)"         : self.learning_rates.regression_head,
                "Optimizer betas"   : str(training_config.optimizer.betas),
                **self.learning_rates.summary(),
            })
            summary = ModelSummary(self.logger, self.model)
            summary.run()
            summary.save_markdown(run_metadata.metadata_directory / IoNames.model_summary_file)

    def _build_param_groups(self):
        regression_head_parameters = list(self.model.regression_head.parameters())
        regression_head_parameters += list(self.model.norm.parameters())
        if self.model.graph_scalars is not None:
            regression_head_parameters += list(self.model.graph_scalars.parameters())

        pool_parameters    = list(self.model.pool.parameters())
        encoder_parameters = list(self.model.encoder.parameters())

        optimizer_config = self.config.optimizer
        groups           = [
            {"params": regression_head_parameters, "lr": self.learning_rates.regression_head, "weight_decay": optimizer_config.weight_decay_regression_head, "name": "regression_head"},
            {"params": pool_parameters,            "lr": self.learning_rates.pool,            "weight_decay": optimizer_config.weight_decay_pool,            "name": "pool"},
            {"params": encoder_parameters,         "lr": self.learning_rates.encoder,         "weight_decay": optimizer_config.weight_decay_encoder,         "name": "encoder"},
        ]

        grouped_parameter_count = sum(parameter.numel() for group in groups for parameter in group["params"])
        total_parameter_count   = sum(parameter.numel() for parameter in self.model.parameters())
        if grouped_parameter_count != total_parameter_count:
            raise RuntimeError(f"Optimizer parameter groups cover {grouped_parameter_count} parameters but the model holds {total_parameter_count}; the encoder/pool/head partition is incomplete.")

        self.logger.section("[Optimizer Parameter Groups]")
        for group in groups:
            parameter_count = sum(parameter.numel() for parameter in group["params"])
            self.logger.subsection(f"{group['name']}: lr={group['lr']} weight_decay={group['weight_decay']} parameters={parameter_count:,}")

        return groups

    def _maybe_resume(self, loader_generator) -> int:
        if not self.resume:
            return 0

        start_epoch = TrainerState.restore(self, self.state_path, loader_generator)

        self.logger.section("[Resumed Trainer State]")
        self.logger.kv_table({
            "State path"  : str(self.state_path),
            "Next epoch"  : f"{start_epoch + 1}/{self.epochs}",
            "Global step" : self.global_step,
            "Best val"    : f"{self.checkpoint.best_val_loss:.4f} @ epoch {self.checkpoint.best_epoch + 1}",
        })

        return start_epoch

    def train_epoch(self, loader, epoch):
        self.model.train()

        weighted_loss     = 0.0
        graph_count       = 0
        number_of_batches = len(loader)
        clear_every       = self.config.memory.clear_cache_every_n_steps

        data_wait   = 0.0
        epoch_start = time.perf_counter()

        with self.logger.track() as progress:
            task_id     = progress.add_task(f"Epoch {epoch + 1}/{self.epochs} train", total=number_of_batches)
            fetch_start = time.perf_counter()

            for batch_index, data in enumerate(loader):
                data_wait   += time.perf_counter() - fetch_start
                data         = data.to(self.device, non_blocking=True)
                _, loss_dict = self.forward(data)

                group_start = (batch_index // self.accumulation_steps) * self.accumulation_steps
                group_size  = min(self.accumulation_steps, number_of_batches - group_start)
                batch_loss  = loss_dict["total_loss"]

                do_step          = (batch_index + 1) % self.accumulation_steps == 0 or (batch_index + 1) >= number_of_batches
                batch_loss_value = batch_loss.item()

                if math.isfinite(batch_loss_value):
                    self.backward(batch_loss / group_size, do_step)
                    weighted_loss += batch_loss_value * data.num_graphs
                    graph_count   += data.num_graphs
                elif self.abort_on_nonfinite_loss:
                    raise RuntimeError(f"Non-finite loss at batch {batch_index}, step {self.global_step}.")
                else:
                    self.logger.warning(f"Non-finite loss at batch {batch_index}, step {self.global_step}; batch skipped because abort_on_nonfinite_loss is disabled.")
                    if do_step:
                        self._flush_pending_step()

                if clear_every > 0 and self.global_step % clear_every == 0:
                    self._clear_memory()

                progress.advance(task_id)
                fetch_start = time.perf_counter()

        self._log_throughput(time.perf_counter() - epoch_start, data_wait, graph_count, epoch)
        self.tracker.log_memory(self.global_step)

        return weighted_loss / max(1, graph_count)

    def backward(self, loss, do_step: bool):
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.pending_backwards += 1
        if do_step:
            self._apply_step()

    def _flush_pending_step(self):
        if self.pending_backwards > 0:
            self._apply_step()

    def _apply_step(self):
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        gradient_norm = self.gradient_clipper.maybe_clip(self.model, self.global_step)
        self.gradient_clipper.record(gradient_norm, self.global_step)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.pending_backwards = 0
        self._finish_step()

    def _finish_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model)
        self.warmup.step()
        self.global_step += 1
        self._apply_learning_rates()

    def _log_throughput(self, epoch_time: float, data_wait: float, graph_count: int, epoch: int) -> None:
        elapsed = max(epoch_time, TrainerSettings.minimum_elapsed)

        self.tracker.log_scalar("throughput/graphs_per_s",   graph_count / elapsed, epoch)
        self.tracker.log_scalar("throughput/epoch_time_s",   epoch_time,            epoch)
        self.tracker.log_scalar("throughput/data_wait_frac", data_wait / elapsed,   epoch)

    def evaluate(self, loader):
        self.model.eval()

        weighted_loss     = 0.0
        graph_count       = 0
        coordinate_errors = CoordinateErrors()

        try:
            with torch.no_grad():
                for data in loader:
                    data                   = data.to(self.device, non_blocking=True)
                    predictions, loss_dict = self.forward(data)

                    coordinate_errors.update(self._physical_residual(predictions, data.y), data.scored)

                    weighted_loss += loss_dict["total_loss"].item() * data.num_graphs
                    graph_count   += data.num_graphs
        finally:
            if self.config.memory.clear_cache_after_eval:
                self._clear_memory()

        return {"avg_loss": weighted_loss / max(1, graph_count), "coordinate_errors": coordinate_errors.results()}

    def forward(self, data):
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predictions = self.model(data)
            loss_dict   = self.criterion(predictions, data)
        return predictions, loss_dict

    def _physical_residual(self, predictions, targets):
        physical_predictions = self.stats.target.inverse_torch(predictions, self.device)
        physical_targets     = self.stats.target.inverse_torch(targets, self.device)
        return physical_predictions - physical_targets

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.vram_reservation.refill()

    def run_epoch(self, train_loader, val_loader, epoch):
        self.scheduler.step(epoch)
        self._apply_learning_rates()
        train_loader.dataset.set_epoch(epoch)

        train_loss = self.train_epoch(train_loader, epoch)

        if not self._validates_at(epoch):
            return train_loss, float("nan"), {}

        with self.ema.applied(self.model):
            evaluation = self.evaluate(val_loader)

        self.scheduler.step_metric(evaluation["avg_loss"])
        return train_loss, evaluation["avg_loss"], evaluation["coordinate_errors"]

    def _validates_at(self, epoch: int) -> bool:
        return ((epoch + 1) % self.validation_frequency == 0) or ((epoch + 1) == self.epochs)

    def _apply_learning_rates(self):
        effective_learning_rates = self.scheduler.effective_lrs()
        for group, learning_rate in zip(self.optimizer.param_groups, effective_learning_rates):
            group["lr"] = learning_rate
            self.tracker.log_scalar(f"lr/{group['name']}", learning_rate, self.global_step)

    def _log_coordinate_errors(self, coordinate_errors, epoch):
        if not coordinate_errors:
            return

        for metric in TrainerSettings.error_metrics:
            for name in CoordinateAxes.names:
                self.tracker.log_scalar(f"val_{metric}/{name}", coordinate_errors[f"{metric}_{name}"], epoch)

            self.logger.subsection(f"Epoch {epoch + 1} val {metric.upper():<4} x/y/z (m) = {self._coordinate_row(coordinate_errors, metric)}")

    def capture_state(self, epoch) -> dict:
        with self.ema.applied(self.model):
            params = {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}

        return {
            "params"      : params,
            "epoch"       : epoch,
            "global_step" : self.global_step,
        }

    def _monitor_fields(self, coordinate_errors):
        if not coordinate_errors:
            return {}

        return {f"val_{metric} x/y/z (m)": self._coordinate_row(coordinate_errors, metric) for metric in TrainerSettings.error_metrics}

    def _coordinate_row(self, coordinate_errors, metric):
        return " / ".join(f"{coordinate_errors[f'{metric}_{name}']:.3f}" for name in CoordinateAxes.names)

    def train(self, train_loader, val_loader):
        self.logger.section("[Training Loop]")
        self.logger.subsection(f"train={len(train_loader)} | val={len(val_loader)}")

        CompletionMarker.clear(self.run_metadata.run_directory)
        self._clear_memory()
        self.optimizer.zero_grad()

        loader_generator = getattr(train_loader, "generator", None)
        start_epoch      = self._maybe_resume(loader_generator)
        last_epoch       = max(start_epoch - 1, 0)
        stopped          = False

        self._apply_learning_rates()
        self.vram_reservation.fill()

        if not self.tuning_mode:
            self.resource_monitor.start()

        try:
            with self.logger.live_monitor("Training Progress", enabled=not self.tuning_mode) as live_monitor:
                for epoch in range(start_epoch, self.epochs):
                    last_epoch                                     = epoch
                    train_loss, validation_loss, coordinate_errors = self.run_epoch(train_loader, val_loader, epoch)

                    self.train_losses.append(train_loss)
                    self.val_losses.append(validation_loss)

                    self.tracker.log_scalar("loss/train", train_loss, epoch)
                    self.logger.subsection(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={validation_loss:.4f}")

                    if self._validates_at(epoch):
                        self.tracker.log_scalar("loss/val", validation_loss, epoch)
                        self._log_coordinate_errors(coordinate_errors, epoch)
                        self.checkpoint.step(validation_loss, epoch, self)
                        stopped = self.early_stopping(validation_loss, epoch)

                    if self.config.memory.clear_cache_after_epoch:
                        self._clear_memory()

                    TrainerState.save(self, epoch, self.state_path, loader_generator)

                    live_monitor.update(
                        epoch         = f"{epoch + 1}/{self.epochs}",
                        train_loss    = train_loss,
                        val_loss      = validation_loss,
                        best_val_loss = self.checkpoint.best_val_loss,
                        best_epoch    = self.checkpoint.best_epoch + 1,
                        lr            = self.scheduler.effective_lrs()[0],
                        **self._monitor_fields(coordinate_errors),
                    )

                    if stopped:
                        break
        finally:
            if not self.tuning_mode:
                self.resource_monitor.stop()

        if self.config.early_stopping.restore_best:
            self.checkpoint.restore_best(self.model, self.device)

        self.logger.section("[Training Complete]")
        self.logger.kv_table({
            "Best val loss" : self.checkpoint.best_val_loss,
            "Best epoch"    : self.checkpoint.best_epoch + 1,
            "Epochs run"    : len(self.train_losses),
            "Checkpoint"    : str(self.run_metadata.checkpoint_path),
        })

        CompletionMarker.stamp(self.run_metadata.run_directory, {
            "stage"            : "training",
            "epochs_completed" : last_epoch + 1,
            "epochs_total"     : self.epochs,
            "early_stopped"    : bool(stopped),
            "best_val_loss"    : float(self.checkpoint.best_val_loss),
            "best_epoch"       : self.checkpoint.best_epoch + 1,
        })

        return {
            "best_val_loss"   : self.checkpoint.best_val_loss,
            "best_epoch"      : self.checkpoint.best_epoch,
            "checkpoint_path" : str(self.run_metadata.checkpoint_path),
            "run_directory"   : str(self.run_metadata.run_directory),
            "train_losses"    : self.train_losses,
            "val_losses"      : self.val_losses,
        }
