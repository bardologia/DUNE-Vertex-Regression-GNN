from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

import numpy as np
import pandas as pd
import torch

from models                          import get_model
from tools.monitoring.logger         import Logger
from tools.reporting.markdown        import MarkdownDoc, MarkdownTable, ScalarFormatter
from tools.runtime.reproducibility   import Reproducibility

from pipelines.dataset.pipeline   import DatasetPipeline
from pipelines.dataset.splitting  import CrossValidationSplitter
from pipelines.inference.metrics  import InferenceMetrics
from pipelines.inference.predictor import Predictor
from pipelines.shared.run_metadata import TrainingRunMetadata
from pipelines.training.trainer    import Trainer


class CrossValidationReport:
    SUMMARY_METRICS = ("euclidean_mean", "euclidean_median", "mae", "rmse", "median_error", "r2")

    def __init__(self, output_directory, logger, cross_validation_config):
        self.output_directory = Path(output_directory)
        self.logger           = logger
        self.config           = cross_validation_config

    def _aggregate(self, fold_metrics):
        shared_keys = set(fold_metrics[0])
        for metrics in fold_metrics[1:]:
            shared_keys &= set(metrics)

        aggregate = {}
        for key in sorted(shared_keys):
            values = [metrics[key] for metrics in fold_metrics]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
                continue

            array         = np.array(values, dtype=np.float64)
            aggregate[key] = {
                "mean" : float(array.mean()),
                "std"  : float(array.std(ddof=0)),
                "min"  : float(array.min()),
                "max"  : float(array.max()),
            }
        return aggregate

    def _write_json(self, fold_records, aggregate):
        payload = {
            "n_folds"             : self.config.n_folds,
            "stratified"          : self.config.stratified,
            "validation_fraction" : self.config.validation_fraction,
            "random_state"        : self.config.random_state,
            "folds"               : fold_records,
            "aggregate"           : aggregate,
        }
        metrics_path = self.output_directory / "cross_validation_metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return metrics_path

    def _build_markdown(self, fold_records, aggregate):
        document = MarkdownDoc(title="Cross-Validation Report")
        document.bold_kv("Folds", self.config.n_folds)
        document.bold_kv("Stratified", self.config.stratified)
        document.bold_kv("Validation fraction", self.config.validation_fraction)
        document.blank()

        per_fold_table = MarkdownTable(["Fold", *self.SUMMARY_METRICS], align=["right"] + ["right"] * len(self.SUMMARY_METRICS))
        for record in fold_records:
            metrics = record["metrics"]
            cells   = [ScalarFormatter.format_scalar(metrics.get(key)) for key in self.SUMMARY_METRICS]
            per_fold_table.add_row(record["fold"], *cells)
        document.heading("Held-out test metrics per fold", level=2)
        document.table(per_fold_table)

        aggregate_table = MarkdownTable(["Metric", "Mean", "Std", "Min", "Max"], align=["left", "right", "right", "right", "right"])
        for key in self.SUMMARY_METRICS:
            if key not in aggregate:
                continue
            statistics = aggregate[key]
            aggregate_table.add_row(
                key,
                ScalarFormatter.format_scalar(statistics["mean"]),
                ScalarFormatter.format_scalar(statistics["std"]),
                ScalarFormatter.format_scalar(statistics["min"]),
                ScalarFormatter.format_scalar(statistics["max"]),
            )
        document.heading("Aggregate across folds", level=2)
        document.table(aggregate_table)

        report_path = self.output_directory / "cross_validation_report.md"
        return document.save(report_path)

    def _log_summary(self, aggregate):
        rows = []
        for key in self.SUMMARY_METRICS:
            if key not in aggregate:
                continue
            statistics = aggregate[key]
            rows.append({"Metric": key, "Mean": statistics["mean"], "Std": statistics["std"], "Min": statistics["min"], "Max": statistics["max"]})
        self.logger.metrics_table(rows, ["Metric", "Mean", "Std", "Min", "Max"], title="Cross-Validation Aggregate (held-out test)")

    def build(self, fold_records):
        self.logger.section("[Cross-Validation Aggregation]")
        fold_metrics = [record["metrics"] for record in fold_records]
        aggregate    = self._aggregate(fold_metrics)

        metrics_path = self._write_json(fold_records, aggregate)
        report_path  = self._build_markdown(fold_records, aggregate)
        self._log_summary(aggregate)

        self.logger.subsection(f"Aggregate metrics written to {metrics_path}")
        self.logger.subsection(f"Report written to {report_path}")
        return {"aggregate": aggregate, "metrics_path": str(metrics_path), "report_path": str(report_path)}


class CrossValidationPipeline:
    def __init__(self, entry_config, logger=None):
        self.entry           = entry_config
        self.training_config = entry_config.training
        self.cross_validation = entry_config.cross_validation
        self.external_logger = logger

    def _prepare_run(self):
        Reproducibility.seed_everything(self.entry.seed)

        stamp              = datetime.now().strftime("%Y%m%d-%H%M%S")
        resolved_name      = self.entry.run_name if self.entry.run_name else stamp
        self.run_directory = Path(self.training_config.io.log_base_dir) / f"cv_{self.entry.model_name}_{resolved_name}"
        self.run_directory.mkdir(parents=True, exist_ok=True)

        self.logger = self.external_logger if self.external_logger is not None else Logger(log_dir=str(self.run_directory / "logs"), name=f"cv_{self.entry.model_name}")
        self.logger.section("[Cross-Validation Pipeline]")
        self.logger.subsection(f"Model={self.entry.model_name} | folds={self.cross_validation.n_folds} | output={self.run_directory}")

    def _prepare_folds(self):
        self.dataset_pipeline = DatasetPipeline(self.entry.dataset, self.logger)
        self.samples          = self.dataset_pipeline.prepare_samples()

        present_base_ids  = np.unique(self.samples[:, 7].astype(np.int64))
        canonical_targets = self.dataset_pipeline.event_targets[present_base_ids]

        target_dataframe = pd.DataFrame(canonical_targets, columns=list(self.entry.dataset.data.coordinate_columns))
        event_folds      = CrossValidationSplitter(self.logger, self.entry.dataset.data.coordinate_columns, self.cross_validation).split(target_dataframe)
        self.folds       = self._expand_event_folds(present_base_ids, event_folds)

        self.logger.subsection(f"Samples: {len(self.samples)} | Base events: {len(present_base_ids)} | Folds: {len(self.folds)}")

    def _expand_event_folds(self, present_base_ids, event_folds):
        sample_base_ids = self.samples[:, 7].astype(np.int64)

        expanded_folds = []
        for train_positions, validation_positions, test_positions in event_folds:
            train_indices      = np.where(np.isin(sample_base_ids, present_base_ids[train_positions]))[0]
            validation_indices = np.where(np.isin(sample_base_ids, present_base_ids[validation_positions]))[0]
            test_indices       = np.where(np.isin(sample_base_ids, present_base_ids[test_positions]))[0]
            expanded_folds.append((train_indices, validation_indices, test_indices))

        return expanded_folds

    def _train_fold(self, fold_index, train_indices, validation_indices, test_indices):
        Reproducibility.seed_everything(self.entry.seed + fold_index)

        datasets, stats = self.dataset_pipeline.run_with_indices(train_indices, validation_indices, test_indices)
        loop            = self.training_config.loop
        loaders         = DatasetPipeline.build_loaders(datasets, loop.batch_size, num_workers=loop.num_workers, pin_memory=loop.pin_memory, persistent_workers=loop.persistent_workers)
        train_loader, validation_loader, test_loader = loaders

        self.entry.model_overrides = DatasetPipeline.inject_feature_dimensions(self.entry.model_overrides, self.entry.dataset)

        run_metadata = TrainingRunMetadata(self.training_config, self.entry.model_name, self.run_directory, run_name=f"fold_{fold_index}")
        run_metadata.save_resolved_config(self.entry)
        run_metadata.save_normalization_stats(stats)
        run_metadata.save_split(self.dataset_pipeline.base_ids_for_indices(train_indices, validation_indices, test_indices))

        model, _ = get_model(self.entry.model_name, **self.entry.model_overrides)
        trainer  = Trainer(model, stats, self.training_config, run_metadata)
        summary  = trainer.train(train_loader, validation_loader, test_loader)

        metrics = self._evaluate_fold(trainer.model, run_metadata, stats, test_loader)
        run_metadata.close()

        return {"fold": fold_index, "run_directory": str(run_metadata.run_directory), "metrics": metrics, "summary": summary}

    def _evaluate_fold(self, model, run_metadata, stats, test_loader):
        device          = self.training_config.loop.device if torch.cuda.is_available() else "cpu"
        checkpoint_path = run_metadata.checkpoint_path
        predictor       = Predictor(model, checkpoint_path, stats, device, run_metadata.logger).prepare()

        predictions, targets = predictor.predict(test_loader)
        return InferenceMetrics(run_metadata.logger, run_metadata.tracker).compute(predictions, targets, stage="cross_validation_test")

    def _run_folds(self):
        self.fold_records = []
        for fold_index, (train_indices, validation_indices, test_indices) in enumerate(self.folds):
            self.logger.section(f"[Fold {fold_index + 1}/{self.cross_validation.n_folds}]")
            self.fold_records.append(self._train_fold(fold_index, train_indices, validation_indices, test_indices))

    def _aggregate(self):
        report = CrossValidationReport(self.run_directory, self.logger, self.cross_validation)
        return report.build(self.fold_records)

    def run(self):
        self._prepare_run()
        self._prepare_folds()
        self._run_folds()
        aggregate_summary = self._aggregate()

        return {
            "run_directory" : str(self.run_directory),
            "folds"         : self.fold_records,
            "aggregate"     : aggregate_summary["aggregate"],
            "metrics_path"  : aggregate_summary["metrics_path"],
            "report_path"   : aggregate_summary["report_path"],
        }
