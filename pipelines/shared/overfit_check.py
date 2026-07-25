from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import fields
from pathlib     import Path

from torch_geometric.data import Batch

from tools.runtime.completion      import CompletionMarker
from tools.runtime.reproducibility import RngSnapshot


class RepeatedBatchLoader:
    def __init__(self, batch, steps: int) -> None:
        self.batch   = batch
        self.steps   = int(steps)
        self.dataset = FixedEpochDataset()

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            yield self.batch


class FixedEpochDataset:
    def set_epoch(self, epoch) -> None:
        return None


class OverfitCheck:

    REPORT_FILENAME = "overfit_report.json"
    DROPOUT_SUFFIX  = ("dropout", "drop_path_rate")

    def __init__(self, config, run_directory, metadata_directory, logger) -> None:
        self.config             = config
        self.run_directory      = Path(run_directory)
        self.metadata_directory = Path(metadata_directory)
        self.logger             = logger

        self.work_directory = self.run_directory / "overfit_check"
        self.report_path    = self.metadata_directory / self.REPORT_FILENAME
        self.overrides      = {}
        self.rng            = RngSnapshot() if config.enabled else None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def epoch_steps(self) -> int:
        return min(self.config.steps_per_epoch, self.config.max_steps)

    @property
    def planned_epochs(self) -> int:
        return math.ceil(self.config.max_steps / self.epoch_steps)

    def record(self, key: str, value) -> None:
        self.overrides[key] = value

    def sanitized_training_config(self, training_config):
        config = copy.deepcopy(training_config)

        config.loop.epochs                  = self.planned_epochs
        config.loop.validation_frequency    = self.planned_epochs
        config.loop.use_ema                 = False
        config.loop.resume                  = False
        config.loop.tuning_mode             = True

        config.optimizer.weight_decay_regression_head = 0.0
        config.optimizer.weight_decay_pool            = 0.0
        config.optimizer.weight_decay_encoder         = 0.0

        config.warmup.warmup_enabled       = False
        config.scheduler.type              = "constant"
        config.early_stopping.patience     = self.planned_epochs + 1
        config.early_stopping.restore_best = False
        config.resources.enabled           = False
        config.memory.reserve_vram         = False

        self.record("optimizer.weight_decay",      0.0)
        self.record("loop.use_ema",                False)
        self.record("warmup.warmup_enabled",       False)
        self.record("scheduler.type",              "constant")
        self.record("early_stopping.restore_best", False)

        return config

    def sanitized_model_overrides(self, model_config, model_overrides: dict) -> dict:
        overrides = dict(model_overrides)

        for field_definition in fields(model_config):
            if not field_definition.name.endswith(self.DROPOUT_SUFFIX):
                continue

            overrides[field_definition.name] = 0.0
            self.record(f"model.{field_definition.name}", 0.0)

        return overrides

    def _gate_batch(self, train_dataset):
        parts = [train_dataset, getattr(train_dataset, "base_dataset", None)]
        saved = [(part, part.augmentation) for part in parts if getattr(part, "augmentation", None) is not None]

        for part, _augmentation in saved:
            part.augmentation = None

        try:
            n_examples = min(self.config.n_examples, len(train_dataset))
            indices    = [(index * len(train_dataset)) // n_examples for index in range(n_examples)]
            items      = [train_dataset[index] for index in indices]
        finally:
            for part, augmentation in saved:
                part.augmentation = augmentation

        self.record("augmentation", "disabled")

        return Batch.from_data_list(items), indices

    def _verdict(self, train_losses: list[float]) -> dict:
        initial = float(train_losses[0])
        best    = float(min(train_losses))
        final   = float(train_losses[-1])
        ratio   = best / initial if initial > 0 else 0.0
        passed  = best <= self.config.stop_threshold or ratio <= self.config.pass_loss_ratio

        return {
            "passed"       : passed,
            "initial_loss" : initial,
            "best_loss"    : best,
            "final_loss"   : final,
            "loss_ratio"   : ratio,
        }

    def _write_report(self, verdict: dict, train_losses: list[float], indices: list[int], duration_s: float) -> None:
        report = {
            **verdict,
            "n_examples"          : len(indices),
            "example_indices"     : indices,
            "max_steps"           : self.config.max_steps,
            "steps_per_epoch"     : self.epoch_steps,
            "epochs_run"          : len(train_losses),
            "steps_run"           : len(train_losses) * self.epoch_steps,
            "pass_loss_ratio"     : self.config.pass_loss_ratio,
            "stop_threshold"      : self.config.stop_threshold,
            "epoch_losses"        : [float(loss) for loss in train_losses],
            "sanitized_overrides" : self.overrides,
            "duration_s"          : duration_s,
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _cleanup(self) -> None:
        for name in ("best_model.pt", "last.pt", CompletionMarker.FILENAME):
            (self.work_directory / name).unlink(missing_ok=True)

        if self.work_directory.is_dir() and not any(self.work_directory.iterdir()):
            self.work_directory.rmdir()

    def _emit(self, verdict: dict) -> None:
        self.logger.section("[Overfit Check Verdict]")
        self.logger.kv_table({
            "Passed"       : verdict["passed"],
            "Initial Loss" : f"{verdict['initial_loss']:.6f}",
            "Best Loss"    : f"{verdict['best_loss']:.6f}",
            "Loss Ratio"   : f"{verdict['loss_ratio']:.4f}",
            "Pass Ratio"   : self.config.pass_loss_ratio,
            "Report"       : str(self.report_path),
        })

        if not verdict["passed"]:
            raise RuntimeError(f"Overfit check failed: best loss {verdict['best_loss']:.6f} vs initial {verdict['initial_loss']:.6f} (ratio {verdict['loss_ratio']:.4f} > {self.config.pass_loss_ratio}); the model cannot fit {self.config.n_examples} examples with regularization disabled, so normal training was aborted.")

        self.logger.subsection("Overfit check passed; continuing with normal training.")

    def run(self, trainer, train_dataset) -> dict:
        self.work_directory.mkdir(parents=True, exist_ok=True)

        start          = time.perf_counter()
        batch, indices = self._gate_batch(train_dataset)
        loader         = RepeatedBatchLoader(batch, self.epoch_steps)

        self.logger.section("[Overfit Check]")
        self.logger.kv_table({
            "Examples"        : self.config.n_examples,
            "Max Steps"       : self.config.max_steps,
            "Steps per Epoch" : self.epoch_steps,
            "Pass Ratio"      : self.config.pass_loss_ratio,
            "Stop Threshold"  : self.config.stop_threshold,
            "Work Directory"  : str(self.work_directory),
            "Sanitized"       : ", ".join(sorted(self.overrides)),
        })

        try:
            summary = trainer.train(loader, loader)
        finally:
            self.rng.restore()

        verdict = self._verdict(summary["train_losses"])

        self._write_report(verdict, summary["train_losses"], indices, time.perf_counter() - start)
        self._cleanup()
        self._emit(verdict)

        return verdict
