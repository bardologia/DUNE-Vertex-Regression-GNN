from __future__ import annotations

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader as GraphDataLoader

from pipelines.dataset.augmentation         import Augmentation
from pipelines.dataset.coordinate_correction import DatasetCorrector
from pipelines.dataset.graph                import Graph
from pipelines.dataset.graph_dataset        import GraphDataset, StatsEstimator
from pipelines.dataset.parquet_store        import ParquetDatasetWriter, ParquetEventReader
from pipelines.dataset.splitting            import TargetBalancer


class DatasetPipeline:
    def __init__(self, dataset_config, logger, stats=None):
        self.config = dataset_config
        self.logger = logger
        self.stats  = stats

        self.geometry_positions = None
        self.light_matrix       = None
        self.datasets           = None

    def _ensure_store(self):
        store_directory = self.config.data.parquet_store_dir
        if (store_directory / "events.parquet").exists():
            return

        if not self.config.data.build_store:
            raise FileNotFoundError(f"Parquet store not found at {store_directory}. Build it first or set data.build_store=true.")

        self.logger.section("[Building Parquet Store]")
        store_parent = store_directory.parent
        DatasetCorrector(self.config.data.raw_input_dir, store_parent, logger=self.logger).run()
        ParquetDatasetWriter(store_parent / "corrected", store_parent, worker_count=self.config.data.store_worker_count, logger=self.logger).run()

    def _load_store(self):
        reader                  = ParquetEventReader(self.config.data.parquet_store_dir).load_store()
        self.geometry_positions = reader.geometry_frame[["x", "y", "z"]].values.astype(np.float32)
        self.light_matrix       = reader.light_matrix
        return reader

    def _build_samples(self, reader):
        if self.config.data.augment_octants:
            frame   = reader.octant_frame
            samples = np.column_stack([
                frame["event_id"].values, frame["sign_x"].values, frame["sign_y"].values, frame["sign_z"].values,
                frame["target_x"].values, frame["target_y"].values, frame["target_z"].values,
            ]).astype(np.float32)
        else:
            count   = len(reader.light_matrix)
            ones    = np.ones(count, dtype=np.float32)
            samples = np.column_stack([np.arange(count, dtype=np.float32), ones, ones, ones, reader.event_targets]).astype(np.float32)

        subset_fraction = self.config.data.subset_fraction
        if 0.0 < subset_fraction < 1.0:
            generator = np.random.default_rng(self.config.split.random_state)
            selection = np.sort(generator.choice(len(samples), size=max(1, int(len(samples) * subset_fraction)), replace=False))
            samples   = samples[selection]

        self.logger.subsection(f"Built {len(samples)} samples (augment_octants={self.config.data.augment_octants})")
        return samples

    def _split_samples(self, samples):
        target_dataframe = pd.DataFrame(samples[:, 4:7], columns=list(self.config.data.coordinate_columns))
        balancer         = TargetBalancer(self.logger, self.config.data.coordinate_columns, self.config.split)
        train_indices, validation_indices, test_indices, _ = balancer.balance(target_dataframe)
        return samples[train_indices], samples[validation_indices], samples[test_indices]

    def _make_dataset(self, samples, augmentation, stats):
        graph_builder = Graph(self.config)
        return GraphDataset(samples, self.geometry_positions, self.light_matrix, graph_builder, self.config.physics, augmentation=augmentation, stats=stats)

    def _fit_stats(self, train_samples):
        if self.stats is not None:
            return
        clean_train_dataset = self._make_dataset(train_samples, augmentation=None, stats=None)
        self.stats          = StatsEstimator(clean_train_dataset, self.config.data.stats_sample_size, self.logger).fit()

    def _build_datasets(self, train_samples, validation_samples, test_samples):
        augmentation  = Augmentation(self.config.augmentation)
        self.datasets = {
            "train" : self._make_dataset(train_samples,      augmentation, self.stats),
            "val"   : self._make_dataset(validation_samples, None,         self.stats),
            "test"  : self._make_dataset(test_samples,       None,         self.stats),
        }

    def run(self):
        self.logger.section("[Dataset Pipeline]")
        self._ensure_store()
        reader                                          = self._load_store()
        samples                                         = self._build_samples(reader)
        train_samples, validation_samples, test_samples = self._split_samples(samples)
        self._fit_stats(train_samples)
        self._build_datasets(train_samples, validation_samples, test_samples)
        return self.datasets, self.stats

    @staticmethod
    def build_loaders(datasets, batch_size, num_workers=0, pin_memory=False, persistent_workers=False):
        persistent   = persistent_workers and num_workers > 0
        train_loader = GraphDataLoader(datasets["train"], batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
        val_loader   = GraphDataLoader(datasets["val"],   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
        test_loader  = GraphDataLoader(datasets["test"],  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
        return train_loader, val_loader, test_loader
