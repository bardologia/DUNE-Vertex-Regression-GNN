from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

import numpy as np
import torch

from configuration.architectures import MODEL_CONFIG_REGISTRY
from tools.monitoring.logger      import Logger
from tools.runtime.config_cli     import ConfigCli

from pipelines.dataset.pipeline import DatasetPipeline

from pipelines.benchmark.stages import ComparisonStage, EvaluationStage, MaxBatchStage, OverfitGateStage, SizeMatchStage, TrainingStage


class BenchmarkPipeline:
    def __init__(self, config, logger=None):
        self.config          = config
        self.external_logger = logger

    def _prepare_run(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        stamp              = datetime.now().strftime("%Y%m%d-%H%M%S")
        resolved_name      = self.config.run_name if self.config.run_name else stamp
        self.run_directory = Path(self.config.training.io.log_base_dir) / f"benchmark_{resolved_name}"
        self.pipeline_directory = self.run_directory / "pipeline"
        self.pipeline_directory.mkdir(parents=True, exist_ok=True)

        self.logger = self.external_logger if self.external_logger is not None else Logger(log_dir=str(self.run_directory / "logs"), name="benchmark")
        self.state  = {"run_directory": str(self.run_directory), "stages": {}}

        ConfigCli.save_resolved(self.config, self.pipeline_directory / "resolved_config.json")

        self.models = [model_name for model_name in MODEL_CONFIG_REGISTRY if model_name not in set(self.config.skip_models)]

        self.logger.section("[Benchmark Pipeline]")
        self.logger.kv_table({
            "Models"          : len(self.models),
            "Reference"       : self.config.size_match.reference_model,
            "Training epochs" : self.config.benchmark_training.epochs,
            "Resume"          : self.config.resume,
            "Run directory"   : str(self.run_directory),
        }, title="Configuration")

    def _prepare_data(self):
        self.datasets, self.stats = DatasetPipeline(self.config.dataset, self.logger).run()

    def _run_size_match(self):
        if not self.config.size_match.enabled:
            return {}
        records = SizeMatchStage(self.config, self.run_directory, self.pipeline_directory, self.logger, self.models).run()
        self._mark_stage("size_match", "completed")
        return records

    def _run_overfit_gate(self):
        if not self.config.overfit.enabled:
            return True

        stage   = OverfitGateStage(self.config, self.run_directory, self.pipeline_directory, self.logger, self.models, self.datasets["train"], self.stats, self.size_records)
        records = stage.run()
        passed  = stage.passed(records)

        self._mark_stage("overfit", "completed" if passed else "failed")
        return passed

    def _run_max_batch(self):
        if not self.config.max_batch.enabled or not torch.cuda.is_available():
            return {}
        records = MaxBatchStage(self.config, self.run_directory, self.pipeline_directory, self.logger, self.models, self.datasets["train"], self.stats, self.size_records).run()
        self._mark_stage("max_batch", "completed")
        return records

    def _run_training(self):
        records = TrainingStage(self.config, self.run_directory, self.pipeline_directory, self.logger, self.models, self.datasets, self.stats, self.size_records, self.max_batch_records).run()
        self._mark_stage("training", "completed")
        return records

    def _run_evaluation(self):
        records = EvaluationStage(self.config, self.run_directory, self.pipeline_directory, self.logger, self.models, self.datasets, self.stats, self.size_records, self.training_records).run()
        self._mark_stage("evaluation", "completed")
        return records

    def _run_comparison(self):
        written = ComparisonStage(self.config, self.run_directory, self.pipeline_directory, self.logger, self.models).run()
        self._mark_stage("comparison", "completed")
        return written

    def _mark_stage(self, stage_name, status):
        self.state["stages"][stage_name] = {"status": status, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        (self.pipeline_directory / "state.json").write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def run(self):
        self._prepare_run()
        try:
            self._prepare_data()

            self.size_records = self._run_size_match()

            gate_passed = self._run_overfit_gate()
            if not gate_passed and self.config.overfit.abort_on_fail:
                self._mark_stage("pipeline", "aborted")
                self.logger.error("Overfit gate failed — benchmark aborted. Inspect the overfit results before retrying.")
                raise SystemExit(1)

            self.max_batch_records = self._run_max_batch()
            self.training_records  = self._run_training()
            self.evaluation_records = self._run_evaluation()
            comparison              = self._run_comparison()

            self._mark_stage("pipeline", "completed")

            self.logger.section("[Benchmark Summary]")
            self.logger.kv_table({
                "Models"   : len(self.models),
                "Trained"  : sum(1 for record in self.training_records.values() if record.get("status") == "DONE"),
                "Report"   : comparison["report_path"],
            }, title="Done")

            return {"run_directory": str(self.run_directory), "report_path": comparison["report_path"], "summary_path": comparison["summary_path"], "ranked": comparison["ranked"]}
        finally:
            if self.external_logger is None:
                self.logger.close()
