from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from configuration.data.general      import CoordinateAxes
from configuration.entry             import InferenceEntryConfig, TrainEntryConfig
from configuration.shared            import IoNames
from models                          import get_model
from tools.monitoring.logger         import Logger
from tools.runtime.config_cli        import ConfigCli
from pipelines.dataset.pipeline      import DatasetPipeline
from pipelines.dataset.normalization import NormalizationStats
from pipelines.shared.store_geometry import StoreGeometryRecord

from pipelines.inference.report  import AnalysisAnimations, AnalysisPlots, AnalysisReport
from pipelines.inference.scoring import InferenceMetrics, Predictor


class InferencePipeline:
    def __init__(self, run_directory, logger=None, device=None, splits=None, batch_size=None):
        defaults = InferenceEntryConfig()

        self.run_directory   = Path(run_directory)
        self.geometry_record = StoreGeometryRecord.for_run(self.run_directory)
        self.logger          = logger if logger is not None else Logger(log_dir=str(self.run_directory / IoNames.logs_dir), name=IoNames.inference_log)
        self.device          = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.splits          = DatasetPipeline.validate_split_names(splits if splits is not None else defaults.splits)
        self.batch_size      = batch_size if batch_size is not None else defaults.batch_size

        self.analysis_directory = self.run_directory / IoNames.analysis_dir

    def _load_run_config(self):
        resolved_path = self.run_directory / IoNames.metadata_dir / IoNames.resolved_config_file
        self.entry    = ConfigCli.load_resolved(TrainEntryConfig(), resolved_path)

        self.logger.subsection(f"Config loaded: model={self.entry.model_name} | run={self.run_directory.name}")

    def _prepare(self):
        self.logger.kv_table({
            "Run directory" : str(self.run_directory),
            "Device"        : self.device,
            "Batch size"    : self.batch_size,
            "Splits"        : ", ".join(self.splits),
        })

        self.stats          = NormalizationStats.load(self.run_directory / IoNames.metadata_dir)
        split_base_ids      = json.loads((self.run_directory / IoNames.metadata_dir / IoNames.split_base_ids_file).read_text(encoding="utf-8"))
        dataset_pipeline    = DatasetPipeline(self.entry.dataset, self.logger, stats=self.stats, evaluation_mode=True)

        dataset_pipeline.prepare_samples()
        self.store_geometry = dataset_pipeline.store_geometry()
        self.geometry_record.require(self.store_geometry)

        self.dataset_splits, _ = dataset_pipeline.run_with_base_ids(split_base_ids, split_names=self.splits)

        model, _        = get_model(self.entry.model_name, **self.entry.model_overrides)
        checkpoint_path = self.run_directory / IoNames.checkpoints_dir / self.entry.training.io.checkpoint_name
        self.predictor  = Predictor(model, checkpoint_path, self.stats, self.device, self.logger).prepare()

    def _analyze_split(self, split_name):
        self.logger.section(f"[Inference: {split_name}]")

        loader               = GraphDataLoader(self.dataset_splits[split_name], batch_size=self.batch_size, shuffle=False)
        predictions, targets, scored = self.predictor.predict(loader)

        metrics         = InferenceMetrics(self.logger).compute(predictions, targets, scored, stage=split_name)
        split_directory = self.analysis_directory / split_name
        plots           = AnalysisPlots(split_directory, self.logger).run(predictions, targets, scored)
        animations      = AnalysisAnimations(split_directory, self.logger).run(predictions, targets, scored)

        return {"metrics": metrics, "plots": plots, "animations": animations}

    def _build_report(self, per_split_results):
        AnalysisReport(self.analysis_directory, self.logger).build(per_split_results)

    def _write_metrics(self, per_split_results):
        metrics_payload = {
            "unit"   : CoordinateAxes.unit,
            "splits" : {split_name: payload["metrics"] for split_name, payload in per_split_results.items()},
        }
        metrics_path = self.run_directory / IoNames.metadata_dir / IoNames.metrics_file
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_payload, indent=IoNames.json_indent), encoding="utf-8")
        self.logger.subsection(f"Metrics written to {metrics_path}")

    def run(self):
        self.logger.section("[Inference Pipeline]")
        self._load_run_config()
        self._prepare()

        per_split_results = {}
        for split_name in self.splits:
            per_split_results[split_name] = self._analyze_split(split_name)

        self._build_report(per_split_results)
        self._write_metrics(per_split_results)
        return per_split_results
