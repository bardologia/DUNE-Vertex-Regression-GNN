from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from configuration.entry             import InferenceEntryConfig, TrainEntryConfig
from models                          import get_model
from tools.monitoring.logger         import Logger
from tools.runtime.config_cli        import ConfigCli
from pipelines.dataset.pipeline       import DatasetPipeline
from pipelines.dataset.normalization import NormalizationStats

from pipelines.inference.animations import AnalysisAnimations
from pipelines.inference.metrics    import InferenceMetrics
from pipelines.inference.plots      import AnalysisPlots
from pipelines.inference.predictor  import Predictor
from pipelines.inference.report     import AnalysisReport


class InferencePipeline:
    def __init__(self, run_directory, logger=None, device=None, splits=None, batch_size=None):
        defaults = InferenceEntryConfig()

        self.run_directory   = Path(run_directory)
        self.logger          = logger if logger is not None else Logger(log_dir=str(self.run_directory / "logs"), name="inference")
        self.device          = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.splits          = splits if splits is not None else defaults.splits
        self.batch_size      = batch_size if batch_size is not None else defaults.batch_size
        self.analysis_directory = self.run_directory / "analysis"

    def _load_run_config(self):
        entry          = TrainEntryConfig()
        resolved_path  = self.run_directory / "metadata" / "resolved_config.json"
        self.entry     = ConfigCli.load_resolved(entry, resolved_path)

    def _prepare(self):
        self.stats                 = NormalizationStats.load(self.run_directory / "metadata")
        self.dataset_splits, _     = DatasetPipeline(self.entry.dataset, self.logger, stats=self.stats, evaluation_mode=True).run()

        model, _       = get_model(self.entry.model_name, **self.entry.model_overrides)
        checkpoint_path = self.run_directory / "checkpoints" / self.entry.training.io.checkpoint_name
        self.predictor  = Predictor(model, checkpoint_path, self.stats, self.device, self.logger).prepare()

    def _analyze_split(self, split_name):
        self.logger.section(f"[Inference: {split_name}]")
        loader               = GraphDataLoader(self.dataset_splits[split_name], batch_size=self.batch_size, shuffle=False)
        predictions, targets = self.predictor.predict(loader)

        metrics    = InferenceMetrics(self.logger).compute(predictions, targets, stage=split_name)
        split_directory = self.analysis_directory / split_name
        plots      = AnalysisPlots(split_directory, self.logger).run(predictions, targets)
        animations = AnalysisAnimations(split_directory, self.logger).run(predictions, targets)

        return {"metrics": metrics, "plots": plots, "animations": animations}

    def _write_metrics(self, per_split_results):
        metrics_payload = {split_name: payload["metrics"] for split_name, payload in per_split_results.items()}
        metrics_path    = self.run_directory / "metadata" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        self.logger.subsection(f"Metrics written to {metrics_path}")

    def run(self):
        self.logger.section("[Inference Pipeline]")
        self._load_run_config()
        self._prepare()

        per_split_results = {}
        for split_name in self.splits:
            if split_name in self.dataset_splits:
                per_split_results[split_name] = self._analyze_split(split_name)

        AnalysisReport(self.analysis_directory, self.logger).build(per_split_results)
        self._write_metrics(per_split_results)
        return per_split_results
