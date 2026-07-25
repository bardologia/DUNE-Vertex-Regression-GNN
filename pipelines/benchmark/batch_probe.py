from __future__ import annotations

import traceback

import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from models                                  import get_model
from tools.training.pretraining.batch_finder import BatchSizeFinder

from pipelines.training.loss import Loss


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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = Loss(self.config.training.loss, self.stats, self.logger)
        return model, optimizer, criterion

    def _measure(self, model, optimizer, criterion, batch_size) -> float:
        loader = GraphDataLoader(self.dataset, batch_size=batch_size, shuffle=False)

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

        return torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 3)

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
