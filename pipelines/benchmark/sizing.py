from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field

import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from configuration.architectures             import ModelConfigRegistry
from configuration.benchmark.general         import BatchProbeSettings
from configuration.shared                    import MeasurementUnits
from models                                  import get_model
from tools.runtime.reproducibility           import Reproducibility
from tools.training.pretraining.batch_finder import BatchSizeFinder

from pipelines.training.trainer import Loss


@dataclass
class SizeMatchResult:
    model         : str
    width         : int
    overrides     : dict
    parameters    : int
    target        : int
    deviation_pct : float
    iterations    : int
    flags         : list = field(default_factory=list)


class ModelSizer:
    def __init__(self, config, logger, model_overrides):
        self.config          = config.size_match
        self.logger          = logger
        self.attribute       = self.config.scaled_attribute
        self.model_overrides = {model_name: dict(overrides) for model_name, overrides in model_overrides.items()}

    def reference_count(self):
        reference = self.config.reference_model
        model, _  = get_model(reference, **self.model_overrides[reference])
        return self.parameter_count(model)

    def _alignment(self, model_name):
        config = ModelConfigRegistry.by_name()[model_name]()
        heads  = getattr(config, "heads", 1)
        return max(1, int(heads))

    def _bracket_upper(self, model_name, target, alignment):
        width = self._snap(self.config.minimum_width, alignment)
        while width < self.config.maximum_width and self._count_at(model_name, width) < target:
            widened = self._snap(width * 2, alignment)
            if widened <= width:
                break
            width = widened
        return width

    def _snap(self, width, alignment):
        floor   = math.ceil(self.config.minimum_width / alignment) * alignment
        ceiling = (self.config.maximum_width // alignment) * alignment
        aligned = max(1, int(round(width / alignment))) * alignment
        return min(ceiling, max(floor, aligned))

    def _count_at(self, model_name, width):
        model, _ = get_model(model_name, **{**self.model_overrides[model_name], self.attribute: int(width)})
        count    = self.parameter_count(model)
        del model
        return count

    @staticmethod
    def parameter_count(model):
        return int(sum(parameter.numel() for parameter in model.parameters()))

    def match(self, model_name, target):
        alignment  = self._alignment(model_name)
        low_width  = self._snap(self.config.minimum_width, alignment)
        high_width = self._bracket_upper(model_name, target, alignment)

        best_width      = low_width
        best_count      = self._count_at(model_name, low_width)
        best_deviation  = abs(best_count - target)
        iterations      = 1

        while low_width < high_width and iterations < self.config.max_iterations:
            mid_width = self._snap((low_width + high_width) // 2, alignment)
            if mid_width in (low_width, high_width):
                break

            count      = self._count_at(model_name, mid_width)
            iterations += 1

            deviation = abs(count - target)
            if deviation < best_deviation:
                best_width, best_count, best_deviation = mid_width, count, deviation

            if count < target:
                low_width = mid_width
            else:
                high_width = mid_width

        flags         = []
        deviation_pct = MeasurementUnits.percent * (best_count - target) / target if target > 0 else 0.0

        if best_width <= self.config.minimum_width and best_count > target:
            flags.append("clamped at minimum width")
        if best_width >= self.config.maximum_width and best_count < target:
            flags.append("clamped at maximum width")
        if abs(deviation_pct) > MeasurementUnits.percent * self.config.tolerance:
            flags.append("outside tolerance")

        return SizeMatchResult(
            model         = model_name,
            width         = int(best_width),
            overrides     = {**self.model_overrides[model_name], self.attribute: int(best_width)},
            parameters    = int(best_count),
            target        = int(target),
            deviation_pct = float(deviation_pct),
            iterations    = int(iterations),
            flags         = flags,
        )


class MaxBatchProbe:
    def __init__(self, config, model_name, dataset, stats, logger, overrides=None):
        self.config           = config
        self.max_batch_config = config.max_batch
        self.model_name       = model_name
        self.dataset          = dataset
        self.stats            = stats
        self.logger           = logger
        self.overrides        = overrides or {}

        self.device   = config.training.loop.device if torch.cuda.is_available() else "cpu"
        self.use_amp  = config.training.loop.use_amp and str(self.device).startswith("cuda")

    def _skipped(self) -> dict:
        return {
            "model"      : self.model_name,
            "status"     : "SKIP",
            "batch_size" : None,
            "peak_gb"    : None,
            "budget_gb"  : self.max_batch_config.vram_budget_gb,
            "ceiling"    : self.max_batch_config.maximum_batch,
            "context_gb" : 0.0,
            "trials"     : [],
            "error"      : "no CUDA device available; max batch probe skipped",
        }

    def _build_components(self):
        model, _  = get_model(self.model_name, **self.overrides)
        model     = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BatchProbeSettings.learning_rate)
        criterion = Loss(self.config.training.loss, self.stats, self.logger)
        return model, optimizer, criterion

    def _measure(self, model, optimizer, criterion, batch_size) -> float:
        loader = GraphDataLoader(self.dataset, batch_size=batch_size, shuffle=True, generator=Reproducibility.generator(self.config.seed))

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        model.train()

        steps = 0
        for data in loader:
            data = data.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                predictions = model(data)
                loss_dict   = criterion(predictions, data)

            loss_dict["total_loss"].backward()
            optimizer.step()

            steps += 1
            if steps >= self.max_batch_config.measure_steps:
                break

        return torch.cuda.max_memory_reserved(self.device) / MeasurementUnits.bytes_per_gib

    def _failed(self, error: str) -> dict:
        return {**self._skipped(), "status": "FAIL", "error": error}

    def run(self) -> dict:
        if self.device == "cpu":
            return self._skipped()

        try:
            model, optimizer, criterion = self._build_components()

            ceiling = min(self.max_batch_config.maximum_batch, len(self.dataset))
            self.logger.subsection(f"{self.model_name}: probing batches up to {ceiling} (budget {self.max_batch_config.vram_budget_gb:.1f} GB)")

            finder = BatchSizeFinder(
                trial_step = lambda batch_size: self._measure(model, optimizer, criterion, batch_size),
                budget_gb  = self.max_batch_config.vram_budget_gb,
                ceiling    = ceiling,
                device     = torch.device(self.device),
                logger     = self.logger,
                model_name = self.model_name,
                on_oom     = torch.cuda.empty_cache,
            )

            return finder.run()

        except Exception:
            return self._failed(traceback.format_exc())
