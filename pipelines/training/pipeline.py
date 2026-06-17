from __future__ import annotations

import numpy as np
import torch

from models import get_model
from pipelines.dataset.pipeline import DatasetPipeline
from pipelines.shared.run_metadata import TrainingRunMetadata

from pipelines.training.trainer import Trainer


class TrainingPipeline:
    def __init__(self, entry_config, logger=None):
        self.entry            = entry_config
        self.training_config  = entry_config.training
        self.external_logger  = logger

    def _prepare_run(self):
        torch.manual_seed(self.entry.seed)
        np.random.seed(self.entry.seed)

        self.run_metadata = TrainingRunMetadata(
            self.training_config,
            self.entry.model_name,
            self.training_config.io.log_base_dir,
            run_name = self.entry.run_name or None,
            logger   = self.external_logger,
        )
        self.logger = self.run_metadata.logger

    def _prepare_data(self):
        dataset_pipeline                          = DatasetPipeline(self.entry.dataset_dir, self.logger, self.entry.split, subset_fraction=self.entry.subset_fraction)
        self.splits, self.stats, self.saved_config = dataset_pipeline.run()

        loop          = self.training_config.loop
        self.loaders  = DatasetPipeline.build_loaders(self.splits, loop.batch_size, num_workers=loop.num_workers, pin_memory=loop.pin_memory, persistent_workers=loop.persistent_workers)

    def _build_model(self):
        self.model, self.model_config = get_model(self.entry.model_name, **self.entry.model_overrides)

    def _train(self):
        self.run_metadata.save_resolved_config(self.entry)
        self.run_metadata.save_normalization_stats(self.stats)

        trainer                              = Trainer(self.model, self.stats, self.training_config, self.run_metadata)
        train_loader, val_loader, test_loader = self.loaders
        return trainer.train(train_loader, val_loader, test_loader)

    def _infer(self, summary):
        from pipelines.inference import InferencePipeline

        InferencePipeline(
            summary["run_directory"],
            logger     = self.logger,
            device     = self.training_config.loop.device,
            batch_size = self.training_config.loop.batch_size,
        ).run()

    def run(self):
        self._prepare_run()
        try:
            self._prepare_data()
            self._build_model()
            summary = self._train()
            if self.entry.infer_after:
                self._infer(summary)
        finally:
            self.run_metadata.close()
        return summary
