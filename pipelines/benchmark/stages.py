from __future__ import annotations

import json
import time
import traceback
from copy        import deepcopy
from dataclasses import asdict
from pathlib     import Path

import torch
from torch.utils.data       import Subset
from torch_geometric.loader import DataLoader as GraphDataLoader

from models                          import get_model
from pipelines.dataset.pipeline       import DatasetPipeline
from pipelines.inference.metrics      import InferenceMetrics
from pipelines.inference.predictor    import Predictor
from pipelines.shared.run_metadata    import LightweightRunContext, TrainingRunMetadata
from pipelines.training.trainer       import Trainer

from pipelines.benchmark.batch_probe import MaxBatchProbe
from pipelines.benchmark.results     import ComparisonReport, TrialCollector
from pipelines.benchmark.sizing      import ModelSizer


class BenchmarkStage:
    RESULT_FILE = None

    def __init__(self, config, run_directory, pipeline_directory, logger, models):
        self.config             = config
        self.run_directory      = Path(run_directory)
        self.pipeline_directory = Path(pipeline_directory)
        self.logger             = logger
        self.models             = list(models)

    def _result_path(self):
        return self.pipeline_directory / self.RESULT_FILE

    def _load_existing(self):
        path = self._result_path()
        if self.config.resume and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _save(self, records):
        self._result_path().write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _overrides(self, model_name):
        return self.size_records.get(model_name, {}).get("overrides", {})

    def _prepare(self):
        return None

    def _compute(self, model_name):
        raise NotImplementedError

    def _report(self, records):
        return None

    def run(self):
        self._prepare()

        records = self._load_existing()
        for model_name in self.models:
            if model_name in records:
                self.logger.info(f"{model_name}: cached result reused")
                continue

            records[model_name] = self._compute(model_name)
            self._save(records)

        self._report(records)
        return records


class SizeMatchStage(BenchmarkStage):
    RESULT_FILE = "size_match.json"

    def _prepare(self):
        self.sizer  = ModelSizer(self.config, self.logger)
        self.target = self.sizer.reference_count()
        self.logger.section("[Benchmark | Capacity Matching]")
        self.logger.subsection(f"Reference '{self.config.size_match.reference_model}': {self.target:,} parameters")

    def _compute(self, model_name):
        if model_name == self.config.size_match.reference_model:
            parameters = self.target
            result     = {"model": model_name, "width": None, "overrides": dict(self.sizer.base_overrides), "parameters": parameters, "target": self.target, "deviation_pct": 0.0, "iterations": 0, "flags": []}
            self.logger.subsection(f"{model_name}: reference ({parameters:,} parameters)")
            return result

        result = asdict(self.sizer.match(model_name, self.target))
        self.logger.subsection(f"{model_name}: {result['parameters']:,} parameters (width {result['width']}, dev {result['deviation_pct']:+.2f} %)")
        for flag in result["flags"]:
            self.logger.warning(f"{model_name}: {flag}")
        return result


class OverfitGateStage(BenchmarkStage):
    RESULT_FILE = "overfit_results.json"

    def __init__(self, config, run_directory, pipeline_directory, logger, models, dataset, stats, size_records):
        super().__init__(config, run_directory, pipeline_directory, logger, models)
        self.dataset      = dataset
        self.stats        = stats
        self.size_records = size_records
        self.gate         = config.overfit

    def _prepare(self):
        self.logger.section("[Benchmark | Overfit Gate]")
        subset_size  = min(self.gate.sample_count, len(self.dataset))
        self.subset  = Subset(self.dataset, list(range(subset_size)))
        self.loader  = GraphDataLoader(self.subset, batch_size=self.gate.batch_size, shuffle=True)

    def _train_tiny(self, model_name):
        training_config                     = deepcopy(self.config.training)
        training_config.loop.epochs         = self.gate.epochs
        training_config.loop.tuning_mode    = True
        training_config.loop.use_amp        = False

        model, _ = get_model(model_name, **self._overrides(model_name))
        context  = LightweightRunContext(self.logger, self.run_directory / "overfit" / model_name / "best_model.pt")
        trainer  = Trainer(model, self.stats, training_config, context)

        final_loss = None
        for epoch in range(self.gate.epochs):
            trainer.scheduler.step(epoch)
            trainer._apply_learning_rates()
            final_loss = trainer.train_epoch(self.loader, epoch)

        return final_loss

    def _compute(self, model_name):
        record = {"model": model_name, "status": None, "final_loss": None, "converged": None, "threshold": self.gate.stop_threshold, "error": None}

        try:
            final_loss          = self._train_tiny(model_name)
            record["final_loss"] = float(final_loss) if final_loss is not None else None
            record["converged"]  = bool(record["final_loss"] is not None and record["final_loss"] <= self.gate.stop_threshold)

            if record["final_loss"] is None:
                record["status"] = "FAIL"
                record["error"]  = "overfit gate produced no final loss"
            elif self.gate.require_convergence and not record["converged"]:
                record["status"] = "FAIL"
                record["error"]  = f"final loss {record['final_loss']:.3e} above threshold {self.gate.stop_threshold:.0e}"
            else:
                record["status"] = "PASS"
        except Exception:
            record["status"] = "FAIL"
            record["error"]  = traceback.format_exc().strip().splitlines()[-1]

        status_message = f"{model_name}: {record['status']}"
        if record["final_loss"] is not None:
            status_message += f" (final loss {record['final_loss']:.3e})"
        self.logger.subsection(status_message)
        return record

    def passed(self, records):
        return all(records[model_name]["status"] == "PASS" for model_name in self.models)


class MaxBatchStage(BenchmarkStage):
    RESULT_FILE = "max_batch.json"

    def __init__(self, config, run_directory, pipeline_directory, logger, models, dataset, stats, size_records):
        super().__init__(config, run_directory, pipeline_directory, logger, models)
        self.dataset      = dataset
        self.stats        = stats
        self.size_records = size_records

    def _prepare(self):
        self.logger.section("[Benchmark | Maximum Batch Size]")
        subset_size = min(self.config.max_batch.sample_count, len(self.dataset))
        self.subset = Subset(self.dataset, list(range(subset_size)))

    def _compute(self, model_name):
        probe  = MaxBatchProbe(self.config, model_name, self.subset, self.stats, self.logger, overrides=self._overrides(model_name))
        result = probe.run()

        if result["status"] == "PASS":
            self.logger.subsection(f"{model_name}: batch {result['batch_size']} (peak {result['peak_gb']:.2f} GB)")
        else:
            self.logger.subsection(f"{model_name}: {result['status']}")
        return result


class TrainingStage(BenchmarkStage):
    RESULT_FILE = "training_results.json"

    def __init__(self, config, run_directory, pipeline_directory, logger, models, datasets, stats, size_records, max_batch_records):
        super().__init__(config, run_directory, pipeline_directory, logger, models)
        self.datasets          = datasets
        self.stats             = stats
        self.size_records      = size_records
        self.max_batch_records = max_batch_records

    def _prepare(self):
        self.logger.section("[Benchmark | Training]")
        self.training_directory = self.run_directory / "training"

    def _batch_size(self, model_name):
        record = self.max_batch_records.get(model_name, {})
        if self.config.benchmark_training.use_measured_batch and record.get("status") == "PASS" and record.get("batch_size"):
            return int(record["batch_size"])
        return self.config.training.loop.batch_size

    def _compute(self, model_name):
        record = {"model": model_name, "status": None, "best_val_loss": None, "best_epoch": None, "duration_s": None, "run_directory": None, "error": None}

        try:
            torch.manual_seed(self.config.seed)

            training_config             = deepcopy(self.config.training)
            training_config.loop.epochs = self.config.benchmark_training.epochs
            batch_size                  = self._batch_size(model_name)

            loaders                          = DatasetPipeline.build_loaders(self.datasets, batch_size, num_workers=training_config.loop.num_workers, pin_memory=training_config.loop.pin_memory, persistent_workers=training_config.loop.persistent_workers)
            train_loader, validation_loader, _ = loaders

            run_metadata = TrainingRunMetadata(training_config, model_name, self.training_directory, run_name="run")
            run_metadata.save_normalization_stats(self.stats)

            model, _ = get_model(model_name, **self._overrides(model_name))
            trainer  = Trainer(model, self.stats, training_config, run_metadata)

            started = time.perf_counter()
            summary = trainer.train(train_loader, validation_loader)
            elapsed = time.perf_counter() - started

            run_metadata.close()

            record.update({
                "status"        : "DONE",
                "best_val_loss" : summary["best_val_loss"],
                "best_epoch"    : summary["best_epoch"],
                "duration_s"    : float(elapsed),
                "run_directory" : summary["run_directory"],
            })
            self.logger.subsection(f"{model_name}: best val loss {summary['best_val_loss']:.4f} in {elapsed:.1f} s")
        except Exception:
            record["status"] = "FAILED"
            record["error"]  = traceback.format_exc().strip().splitlines()[-1]
            self.logger.error(f"{model_name}: training FAILED ({record['error']})")

        return record


class EvaluationStage(BenchmarkStage):
    RESULT_FILE = "evaluation_results.json"

    def __init__(self, config, run_directory, pipeline_directory, logger, models, datasets, stats, size_records, training_records):
        super().__init__(config, run_directory, pipeline_directory, logger, models)
        self.datasets         = datasets
        self.stats            = stats
        self.size_records     = size_records
        self.training_records = training_records

    def _prepare(self):
        self.logger.section("[Benchmark | Evaluation]")
        self.device     = self.config.training.loop.device if torch.cuda.is_available() else "cpu"
        self.batch_size = self.config.training.loop.batch_size

    def _compute(self, model_name):
        training_record = self.training_records.get(model_name, {})
        if training_record.get("status") != "DONE" or not training_record.get("run_directory"):
            self.logger.subsection(f"{model_name}: skipped (no trained checkpoint)")
            return {}

        run_directory   = Path(training_record["run_directory"])
        checkpoint_path = run_directory / "checkpoints" / self.config.training.io.checkpoint_name

        model, _    = get_model(model_name, **self._overrides(model_name))
        predictor   = Predictor(model, checkpoint_path, self.stats, self.device, self.logger).prepare()
        test_loader = GraphDataLoader(self.datasets["test"], batch_size=self.batch_size, shuffle=False)

        predictions, targets = predictor.predict(test_loader)
        metrics              = InferenceMetrics(self.logger).compute(predictions, targets, stage=f"benchmark_{model_name}")
        return metrics


class ComparisonStage(BenchmarkStage):
    RESULT_FILE = "benchmark_summary.json"

    def run(self):
        self.logger.section("[Benchmark | Comparison]")

        collector = TrialCollector(self.pipeline_directory, self.models, self.logger)
        records   = collector.collect()

        output_directory = self.run_directory / "comparison"
        report           = ComparisonReport(records, output_directory, self.config, self.logger)
        written          = report.write_all()

        self.logger.subsection(f"Report written to {written['report_path']}")
        return written
