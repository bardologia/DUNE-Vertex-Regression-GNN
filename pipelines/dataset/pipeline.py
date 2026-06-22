from __future__ import annotations

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader as GraphDataLoader

from pipelines.dataset.augmentation         import Augmentation
from pipelines.dataset.graph                import FeatureSchema, Graph
from pipelines.dataset.graph_dataset        import CachedGraphDataset, DegreeHistogramEstimator, GraphDataset, StatsEstimator
from pipelines.dataset.parquet_store        import ParquetDatasetWriter, ParquetEventReader
from pipelines.dataset.splitting            import TargetBalancer


class DatasetPipeline:
    def __init__(self, dataset_config, logger, stats=None, evaluation_mode=False):
        self.config          = dataset_config
        self.logger          = logger
        self.stats           = stats
        self.evaluation_mode = evaluation_mode

        self.geometry_positions = None
        self.light_matrix       = None
        self.event_targets      = None
        self.octant_frame       = None
        self.samples            = None
        self.datasets           = None
        self.split_base_ids     = None

    def _ensure_store(self):
        store_directory = self.config.data.parquet_store_dir
        if (store_directory / "events.parquet").exists():
            return

        if not self.config.data.build_store:
            raise FileNotFoundError(f"Parquet store not found at {store_directory}. Build it first or set data.build_store=true.")

        self.logger.section("[Building Parquet Store]")
        store_parent = store_directory.parent
        ParquetDatasetWriter(self.config.data.raw_input_dir, store_parent, worker_count=self.config.data.store_worker_count, logger=self.logger, hot_channels=self.config.data.hot_channels).run()

    def _load_store(self):
        reader                  = ParquetEventReader(self.config.data.parquet_store_dir).load_store(load_octant_light=False)
        self.geometry_positions = reader.geometry_frame[["x", "y", "z"]].values.astype(np.float32)
        self.event_targets      = reader.event_targets
        self.light_matrix       = reader.light_matrix
        self.octant_frame       = reader.octant_frame
        self.logger.subsection(f"Store loaded: {len(self.event_targets)} base events | {self.geometry_positions.shape[0]} channels | octant rows {len(self.octant_frame)}")
        return reader

    def _build_samples(self):
        count       = len(self.light_matrix)
        base_ids    = np.arange(count, dtype=np.float32)
        ones        = np.ones(count, dtype=np.float32)
        samples     = np.column_stack([base_ids, ones, ones, ones, self.event_targets, base_ids]).astype(np.float32)

        self.logger.subsection(f"Built {len(samples)} base-event samples")
        return samples

    def prepare_samples(self):
        if self.samples is not None:
            return self.samples

        self._ensure_store()
        self._load_store()
        self.samples = self._build_samples()
        return self.samples

    def _octant_expand(self, base_samples):
        present_ids = set(base_samples[:, 7].astype(np.int64).tolist())
        member_mask = self.octant_frame["base_event_id"].isin(present_ids).values
        member_rows = self.octant_frame.loc[member_mask]

        light_index = np.nonzero(member_mask)[0].astype(np.float32)
        signs       = member_rows[["sign_x", "sign_y", "sign_z"]].values.astype(np.float32)
        targets     = member_rows[["target_x", "target_y", "target_z"]].values.astype(np.float32)
        base_ids    = member_rows["base_event_id"].values.astype(np.float32)
        return np.column_stack([light_index, signs, targets, base_ids]).astype(np.float32)

    def _subsample(self, samples):
        subset_fraction = self.config.data.subset_fraction
        if not (0.0 < subset_fraction < 1.0):
            return samples

        generator = np.random.default_rng(self.config.split.random_state)
        selection = np.sort(generator.choice(len(samples), size=max(1, int(len(samples) * subset_fraction)), replace=False))
        return samples[selection]

    def _prepare_train_samples(self, train_base_samples):
        if self.evaluation_mode:
            return train_base_samples

        train_samples = self._octant_expand(train_base_samples) if self.config.data.augment_octants else train_base_samples
        train_samples = self._subsample(train_samples)
        self.logger.subsection(f"Train samples: {len(train_samples)} (octants={self.config.data.augment_octants}, subset_fraction={self.config.data.subset_fraction})")
        return train_samples

    def _split_samples(self, samples):
        present_base_ids  = np.unique(samples[:, 7].astype(np.int64))
        canonical_targets = self.event_targets[present_base_ids]

        target_dataframe = pd.DataFrame(canonical_targets, columns=list(self.config.data.coordinate_columns))
        balancer         = TargetBalancer(self.logger, self.config.data.coordinate_columns, self.config.split)
        train_rows, validation_rows, test_rows, _ = balancer.balance(target_dataframe)

        self.split_base_ids = {
            "train"      : present_base_ids[train_rows].tolist(),
            "validation" : present_base_ids[validation_rows].tolist(),
            "test"       : present_base_ids[test_rows].tolist(),
        }
        return self._partition_by_base_ids(samples, self.split_base_ids)

    def _partition_by_base_ids(self, samples, split_base_ids):
        sample_base_ids = samples[:, 7].astype(np.int64)
        train_ids       = set(split_base_ids["train"])
        validation_ids  = set(split_base_ids["validation"])
        test_ids        = set(split_base_ids["test"])

        train_mask      = np.fromiter((base_id in train_ids      for base_id in sample_base_ids), dtype=bool, count=len(sample_base_ids))
        validation_mask = np.fromiter((base_id in validation_ids for base_id in sample_base_ids), dtype=bool, count=len(sample_base_ids))
        test_mask       = np.fromiter((base_id in test_ids       for base_id in sample_base_ids), dtype=bool, count=len(sample_base_ids))

        return samples[train_mask], samples[validation_mask], samples[test_mask]

    def _make_dataset(self, samples, augmentation, stats):
        graph_builder = Graph(self.config)
        return GraphDataset(samples, self.geometry_positions, self.light_matrix, graph_builder, self.config.physics, augmentation=augmentation, stats=stats)

    def _fit_stats(self, train_samples):
        if self.stats is not None:
            self.logger.subsection("Normalization stats reused from caller (skipping fit)")
            return
        clean_train_dataset = self._make_dataset(train_samples, augmentation=None, stats=None)
        self.stats          = StatsEstimator(clean_train_dataset, self.config.data.stats_sample_size, self.logger).fit()

    def _build_datasets(self, train_samples, validation_samples, test_samples):
        train_augmentation  = None if self.evaluation_mode else Augmentation(self.config.augmentation)
        train_is_stochastic = train_augmentation is not None and train_augmentation.active

        train_dataset = self._make_dataset(train_samples,      train_augmentation, self.stats)
        val_dataset   = self._make_dataset(validation_samples, None,               self.stats)
        test_dataset  = self._make_dataset(test_samples,       None,               self.stats)

        self.logger.kv_table({
            "Train samples" : len(train_samples),
            "Val samples"   : len(validation_samples),
            "Test samples"  : len(test_samples),
            "Train caching" : "live (augmented)" if train_is_stochastic else "cached",
        }, title="Dataset Splits")

        self.datasets = {
            "train" : train_dataset if train_is_stochastic else CachedGraphDataset(train_dataset, self.logger),
            "val"   : CachedGraphDataset(val_dataset,  self.logger),
            "test"  : CachedGraphDataset(test_dataset, self.logger),
        }

    @staticmethod
    def build_loaders(datasets, batch_size, num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2):
        persistent    = persistent_workers and num_workers > 0
        worker_options = {"prefetch_factor": prefetch_factor} if num_workers > 0 else {}

        train_loader = GraphDataLoader(datasets["train"], batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, **worker_options)
        val_loader   = GraphDataLoader(datasets["val"],   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, **worker_options)
        test_loader  = GraphDataLoader(datasets["test"],  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, **worker_options)
        return train_loader, val_loader, test_loader

    @staticmethod
    def inject_feature_dimensions(model_overrides, dataset_config):
        node_dimension, edge_dimension = FeatureSchema(dataset_config).dimensions()
        return {**model_overrides, "input_dim": node_dimension, "edge_dim": edge_dimension}

    def pna_degree_histogram(self, model_name, dataset):
        if model_name != "pna":
            return None
        return DegreeHistogramEstimator(dataset, self.config.data.stats_sample_size, self.logger).fit()

    def base_ids_for_indices(self, train_indices, validation_indices, test_indices):
        sample_base_ids = self.samples[:, 7].astype(np.int64)
        return {
            "train"      : np.unique(sample_base_ids[train_indices]).tolist(),
            "validation" : np.unique(sample_base_ids[validation_indices]).tolist(),
            "test"       : np.unique(sample_base_ids[test_indices]).tolist(),
        }

    def run_with_indices(self, train_indices, validation_indices, test_indices):
        self.logger.section("[Dataset Pipeline | Fold]")
        self.prepare_samples()

        train_base         = self.samples[train_indices]
        validation_samples = self.samples[validation_indices]
        test_samples       = self.samples[test_indices]

        self.stats    = None
        train_samples = self._prepare_train_samples(train_base)
        self._fit_stats(train_samples)
        self._build_datasets(train_samples, validation_samples, test_samples)
        return self.datasets, self.stats

    def run_with_base_ids(self, split_base_ids):
        self.logger.section("[Dataset Pipeline | Persisted Split]")
        self.prepare_samples()

        train_base, validation_samples, test_samples = self._partition_by_base_ids(self.samples, split_base_ids)
        train_samples = self._prepare_train_samples(train_base)
        self._fit_stats(train_samples)
        self._build_datasets(train_samples, validation_samples, test_samples)
        return self.datasets, self.stats

    def build_evaluation_split(self, split_base_ids, split_name):
        if self.stats is None:
            raise ValueError("build_evaluation_split requires normalization stats supplied to the pipeline; physics-only re-evaluation must reuse the training-split statistics.")

        self.prepare_samples()

        partitions   = dict(zip(("train", "val", "test"), self._partition_by_base_ids(self.samples, split_base_ids)))
        if split_name not in partitions:
            raise KeyError(f"Unknown split '{split_name}'. Expected one of {tuple(partitions)}.")

        split_samples = partitions[split_name]
        split_dataset = self._make_dataset(split_samples, augmentation=None, stats=self.stats)
        return CachedGraphDataset(split_dataset, self.logger)

    def run(self):
        self.logger.section("[Dataset Pipeline]")
        self.prepare_samples()

        train_base, validation_samples, test_samples = self._split_samples(self.samples)
        train_samples = self._prepare_train_samples(train_base)
        self._fit_stats(train_samples)
        self._build_datasets(train_samples, validation_samples, test_samples)
        return self.datasets, self.stats
