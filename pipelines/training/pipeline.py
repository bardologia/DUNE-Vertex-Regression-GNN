from __future__ import annotations

import copy

from models                         import get_model
from tools.runtime.reproducibility  import Reproducibility
from pipelines.dataset.pipeline     import DatasetPipeline
from pipelines.shared.overfit_check import OverfitCheck
from pipelines.shared.run_metadata  import LightweightRunContext, TrainingRunMetadata

from pipelines.training.optimizer_overrides import OptimizerOverrideApplier
from pipelines.training.trainer             import Trainer


class TrainingPipeline:
    def __init__(self, entry_config, logger=None, explicit_paths=None):
        self.entry           = entry_config
        self.training_config = entry_config.training
        self.external_logger = logger
        self.explicit_paths  = explicit_paths

    def _prepare_run(self):
        Reproducibility.seed_everything(self.entry.seed)

        self.run_metadata = TrainingRunMetadata(
            self.training_config,
            self.entry.model_name,
            self.training_config.io.log_base_dir,
            run_name = self.entry.run_name or None,
            logger   = self.external_logger,
        )
        self.logger = self.run_metadata.logger

        self.logger.section("[Training Pipeline]")
        self.logger.kv_table({
            "Model"       : self.entry.model_name,
            "Seed"        : self.entry.seed,
            "Device"      : self.training_config.loop.device,
            "Epochs"      : self.training_config.loop.epochs,
            "Infer after" : self.entry.infer_after,
        })

    def _prepare_data(self):
        self.dataset_pipeline     = DatasetPipeline(self.entry.dataset, self.logger)
        self.datasets, self.stats = self.dataset_pipeline.run()

        loop         = self.training_config.loop
        self.loaders = DatasetPipeline.build_loaders(self.datasets, loop.batch_size, num_workers=loop.num_workers, pin_memory=loop.pin_memory, persistent_workers=loop.persistent_workers, prefetch_factor=loop.prefetch_factor, seed=self.entry.seed)

        train_loader, val_loader, test_loader = self.loaders
        self.logger.section("[Data Loaders]")
        self.logger.kv_table({
            "Train batches"   : len(train_loader),
            "Val batches"     : len(val_loader),
            "Test batches"    : len(test_loader),
            "Batch size"      : loop.batch_size,
            "Workers"         : loop.num_workers,
            "Pin memory"      : loop.pin_memory,
            "Prefetch factor" : loop.prefetch_factor,
        })

    def _resolve_model_overrides(self, dataset_pipeline, train_dataset):
        overrides        = dataset_pipeline.inject_model_overrides(self.entry.model_overrides)
        degree_histogram = dataset_pipeline.pna_degree_histogram(self.entry.model_name, train_dataset)

        if degree_histogram is None:
            return overrides

        return {**overrides, "degree_histogram": degree_histogram}

    def _build_model(self):
        self.entry.model_overrides = self._resolve_model_overrides(self.dataset_pipeline, self.datasets["train"])

        self.model, self.model_config = get_model(self.entry.model_name, **self.entry.model_overrides)

        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.section("[Model Architecture]")
        self.logger.kv_table({
            "Architecture" : self.entry.model_name,
            "Input dim"    : self.entry.model_overrides.get("input_dim"),
            "Edge dim"     : self.entry.model_overrides.get("edge_dim"),
            "Parameters"   : f"{total_params:,}",
        })

    def _run_overfit_check(self):
        check = OverfitCheck(self.training_config.overfit_check, self.run_metadata.run_directory, self.run_metadata.metadata_directory, self.logger)
        if not check.enabled:
            return

        gate_config    = check.sanitized_training_config(self.training_config)
        gate_overrides = check.sanitized_model_overrides(self.model_config, self.entry.model_overrides)

        gate_model, gate_model_config = get_model(self.entry.model_name, **gate_overrides)

        OptimizerOverrideApplier(gate_config, gate_model_config, self.explicit_paths, self.logger).run()

        gate_context = LightweightRunContext(self.logger, check.work_directory / "best_model.pt")
        gate_trainer = Trainer(gate_model, self.stats, gate_config, gate_context)

        check.run(gate_trainer, self.datasets["train"])

    def _train(self):
        OptimizerOverrideApplier(self.training_config, self.model_config, self.explicit_paths, self.logger).run()

        self.run_metadata.save_resolved_config(self.entry)
        self.run_metadata.save_normalization_stats(self.stats)
        self.run_metadata.save_split(self.dataset_pipeline.split_base_ids)

        trainer                     = Trainer(self.model, self.stats, self.training_config, self.run_metadata)
        train_loader, val_loader, _ = self.loaders
        return trainer.train(train_loader, val_loader)

    def _infer(self, summary):
        from pipelines.inference import InferencePipeline

        InferencePipeline(
            summary["run_directory"],
            logger     = self.logger,
            device     = self.training_config.loop.device,
            batch_size = self.training_config.loop.batch_size,
        ).run()

    def build_pretrain_trainer(self, work_directory, logger):
        Reproducibility.seed_everything(self.entry.seed)

        dataset_pipeline = DatasetPipeline(self.entry.dataset, logger)
        datasets, stats  = dataset_pipeline.run()

        overrides           = self._resolve_model_overrides(dataset_pipeline, datasets["train"])
        model, model_config = get_model(self.entry.model_name, **overrides)

        probe_config                   = copy.deepcopy(self.training_config)
        probe_config.loop.tuning_mode  = True
        probe_config.resources.enabled = False

        OptimizerOverrideApplier(probe_config, model_config, self.explicit_paths, logger).run()

        trainer = Trainer(model, stats, probe_config, LightweightRunContext(logger, work_directory / "probe.pt"))

        return trainer, datasets["train"], model

    def run(self):
        self._prepare_run()
        try:
            self._prepare_data()
            self._build_model()
            self._run_overfit_check()
            summary = self._train()
            if self.entry.infer_after:
                self._infer(summary)
        finally:
            self.run_metadata.close()
        return summary
