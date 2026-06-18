from __future__ import annotations

import traceback

import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from models                   import get_model
from tools.monitoring.tracker import NullTracker
from pipelines.training.loss  import Loss


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
        self.use_amp  = config.training.loop.use_amp and torch.cuda.is_available()

    def _candidate_batches(self):
        ceiling   = min(self.max_batch_config.maximum_batch, len(self.dataset))
        candidate = 1
        batches   = []
        while candidate <= ceiling:
            batches.append(candidate)
            candidate *= 2
        return batches

    def _build_components(self):
        model, _  = get_model(self.model_name, **self.overrides)
        model     = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = Loss(self.config.training.loss, self.stats, self.logger, NullTracker())
        return model, optimizer, criterion

    def _measure(self, model, optimizer, criterion, batch_size):
        loader = GraphDataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        torch.cuda.reset_peak_memory_stats(self.device)
        model.train()

        steps = 0
        for data in loader:
            data = data.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                predictions = model(data)
                loss_dict   = criterion(predictions, data, 0)

            loss_dict["total_loss"].backward()
            optimizer.step()

            steps += 1
            if steps >= self.max_batch_config.measure_steps:
                break

        peak_bytes = torch.cuda.max_memory_reserved(self.device)
        return peak_bytes / 1e9

    def run(self):
        result = {
            "model"      : self.model_name,
            "status"     : None,
            "batch_size" : None,
            "peak_gb"    : None,
            "budget_gb"  : self.max_batch_config.vram_budget_gb,
            "ceiling"    : self.max_batch_config.maximum_batch,
            "trials"     : [],
            "error"      : None,
        }

        if self.device == "cpu":
            result["status"] = "SKIP"
            result["error"]  = "no CUDA device available; max batch probe skipped"
            return result

        try:
            model, optimizer, criterion = self._build_components()

            candidates = self._candidate_batches()
            self.logger.subsection(f"{self.model_name}: probing batches {candidates} (budget {self.max_batch_config.vram_budget_gb:.1f} GB)")

            best_batch = None
            best_peak  = None

            for batch_size in candidates:
                try:
                    peak_gb = self._measure(model, optimizer, criterion, batch_size)
                except torch.cuda.OutOfMemoryError:
                    result["trials"].append({"batch_size": batch_size, "peak_gb": None, "status": "OOM"})
                    self.logger.subsection(f"  batch {batch_size}: OOM")
                    torch.cuda.empty_cache()
                    break

                within_budget = peak_gb <= self.max_batch_config.vram_budget_gb
                status        = "FIT" if within_budget else "OVER"
                result["trials"].append({"batch_size": batch_size, "peak_gb": float(peak_gb), "status": status})
                self.logger.subsection(f"  batch {batch_size}: {peak_gb:.2f} GB ({status})")

                if not within_budget:
                    break

                best_batch, best_peak = batch_size, peak_gb

            if best_batch is None:
                result["status"] = "FAIL"
                result["error"]  = "no batch size fits within the VRAM budget"
            else:
                result["status"]     = "PASS"
                result["batch_size"] = int(best_batch)
                result["peak_gb"]    = float(best_peak)

        except Exception:
            result["status"] = "FAIL"
            result["error"]  = traceback.format_exc()

        return result
