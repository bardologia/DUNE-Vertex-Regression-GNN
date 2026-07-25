from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib     import Path
from typing      import Callable, Optional

import torch

from tools.monitoring.logger                 import Logger
from tools.training.pretraining.batch_finder import BatchSizeFinder
from tools.training.pretraining.loader_tuner import LoaderTuner


@dataclass
class PretrainContext:
    dataset        : object
    model          : object
    to_model_input : Callable
    forward_loss   : Callable
    trial_step     : Callable[[int], float]
    device         : torch.device
    use_amp        : bool                         = False
    context_gb     : float                        = 0.0
    on_oom         : Optional[Callable[[], None]] = None


class PretrainOrchestrator:

    RESULT_FILENAME = "pretrain_report.json"

    def __init__(self, pretrain_config, training_config, build_context: Callable[[], PretrainContext], logger: Logger, label: Optional[str] = None, result_dir: Optional[Path] = None) -> None:
        self.pretrain   = pretrain_config
        self.training   = training_config
        self.build      = build_context
        self.logger     = logger
        self.label      = label
        self.result_dir = Path(result_dir) if result_dir is not None else None
        self.report     = {"model": label, "batch_size": None, "loader": None}

    def _enabled(self) -> bool:
        return bool(self.pretrain.find_batch_size or self.pretrain.tune_loader)

    def _find_batch_size(self, context: PretrainContext) -> None:
        ceiling = min(self.pretrain.max_batch, len(context.dataset))
        if ceiling < self.pretrain.max_batch:
            self.logger.subsection(f"Ceiling lowered from {self.pretrain.max_batch} to {ceiling}: the dataset holds {len(context.dataset)} samples")

        finder = BatchSizeFinder(
            trial_step = context.trial_step,
            budget_gb  = self.pretrain.vram_budget_gb,
            ceiling    = ceiling,
            device     = context.device,
            logger     = self.logger,
            model_name = self.label,
            context_gb = context.context_gb,
            on_oom     = context.on_oom,
        )

        result = finder.run()

        if result["status"] != "PASS":
            raise SystemExit(f"max batch-size finder failed: {result['error']}")

        self.training.batch_size = int(result["batch_size"])
        self.report["batch_size"] = result

        self.logger.subsection(f"Resolved batch size: {self.training.batch_size} (peak {result['peak_gb']:.2f} GB of the {result['budget_gb']:.2f} GB budget)")

    def _release_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _tune_loader(self, context: PretrainContext) -> None:
        tuner = LoaderTuner(
            dataset          = context.dataset,
            model            = context.model,
            to_model_input   = context.to_model_input,
            forward_loss     = context.forward_loss,
            device           = context.device,
            logger           = self.logger,
            use_amp          = context.use_amp,
            seed             = self.pretrain.seed,
            warmup_batches   = self.pretrain.warmup_batches,
            timed_batches    = self.pretrain.timed_batches,
            worker_counts    = self.pretrain.worker_counts,
            prefetch_factors = self.pretrain.prefetch_factors,
            data_wait_target = self.pretrain.data_wait_target,
        )

        choice = tuner.run(self.training.batch_size)

        if choice is None:
            return

        self.training.num_workers     = int(choice["num_workers"])
        self.training.prefetch_factor = int(choice["prefetch_factor"])
        self.training.pin_memory      = bool(choice["pin_memory"])
        self.report["loader"]         = choice

        self.logger.subsection(f"Resolved loader: workers={self.training.num_workers} prefetch={self.training.prefetch_factor} pin_memory={self.training.pin_memory}")

    def _write_report(self) -> None:
        if self.result_dir is None:
            return

        self.result_dir.mkdir(parents=True, exist_ok=True)
        (self.result_dir / self.RESULT_FILENAME).write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    def run(self) -> None:
        if not self._enabled():
            return

        if self.pretrain.find_batch_size:
            self.logger.section("[Pretrain] Max batch-size finder")
            self.logger.subsection("Probe run: the dataset and model below belong to the batch-size probe, not the training run.")
            self._find_batch_size(self.build())
            self._release_cache()

        if self.pretrain.tune_loader:
            self.logger.section("[Pretrain] DataLoader tuner")
            self.logger.subsection("Probe run: the dataset and model below belong to the loader-timing probe, not the training run.")
            self._tune_loader(self.build())
            self._release_cache()

        self._write_report()
