from __future__ import annotations

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader as GraphDataLoader

from tools.runtime.reproducibility          import Reproducibility

from pipelines.dataset.augmentation         import Augmentation
from pipelines.dataset.graph                import FeatureSchema, Graph
from pipelines.dataset.graph_dataset        import CachedGraphDataset, DegreeHistogramEstimator, GraphDataset, GraphNormalizer, StatsEstimator
from pipelines.dataset.parquet_store        import ParquetDatasetWriter, ParquetEventReader
from pipelines.dataset.splitting            import TargetBalancer


class DatasetPipeline:
    SPLIT_NAMES = ("train", "val", "test")

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
        self.raw_train_graphs   = None

    @classmethod
    def validate_split_names(cls, split_names):
        unknown = tuple(name for name in split_names if name not in cls.SPLIT_NAMES)
        if unknown:
            raise KeyError(f"Unknown split name(s) {unknown}. Expected any of {cls.SPLIT_NAMES}.")
        return tuple(split_names)

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
        reader                  = ParquetEventReader(self.config.data.parquet_store_dir).load_store()
        self.geometry_positions = reader.geometry_frame[list(self.config.data.position_columns)].values.astype(np.float32)
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

    def _subsample(self, samples):
        subset_fraction = self.config.data.subset_fraction
        if not (0.0 < subset_fraction < 1.0):
            return samples

        generator = np.random.default_rng(self.config.split.random_state)
        selection = np.sort(generator.choice(len(samples), size=max(1, int(len(samples) * subset_fraction)), replace=False))
        return samples[selection]

    def _octant_expand(self, base_samples):
        present_ids = set(base_samples[:, 7].astype(np.int64).tolist())
        member_mask = self.octant_frame["base_event_id"].isin(present_ids).values
        member_rows = self.octant_frame.loc[member_mask]

        light_index = np.nonzero(member_mask)[0].astype(np.float32)
        signs       = member_rows[["sign_x", "sign_y", "sign_z"]].values.astype(np.float32)
        targets     = member_rows[["target_x", "target_y", "target_z"]].values.astype(np.float32)
        base_ids    = member_rows["base_event_id"].values.astype(np.float32)
        return np.column_stack([light_index, signs, targets, base_ids]).astype(np.float32)

    def _record_train_base_ids(self, train_samples):
        if self.split_base_ids is None:
            return
        self.split_base_ids["train"] = np.unique(train_samples[:, 7].astype(np.int64)).tolist()

    def _prepare_train_samples(self, train_base_samples):
        if self.evaluation_mode:
            return train_base_samples

        selected_base_samples = self._subsample(train_base_samples)
        train_samples         = self._octant_expand(selected_base_samples) if self.config.data.augment_octants else selected_base_samples

        self._record_train_base_ids(train_samples)
        self.logger.subsection(f"Train samples: {len(train_samples)} rows from {len(selected_base_samples)} base events (octants={self.config.data.augment_octants}, subset_fraction={self.config.data.subset_fraction})")
        return train_samples

    def _split_samples(self, samples):
        present_base_ids  = np.unique(samples[:, 7].astype(np.int64))
        canonical_targets = self.event_targets[present_base_ids]

        target_dataframe = pd.DataFrame(canonical_targets, columns=list(self.config.data.coordinate_columns))
        balancer         = TargetBalancer(self.logger, self.config.data.coordinate_columns, self.config.split)
        train_rows, validation_rows, test_rows, _ = balancer.balance(target_dataframe)

        self.split_base_ids = {
            "train" : present_base_ids[train_rows].tolist(),
            "val"   : present_base_ids[validation_rows].tolist(),
            "test"  : present_base_ids[test_rows].tolist(),
        }
        return self._partition_by_base_ids(samples, self.split_base_ids)

    def _partition_by_base_ids(self, samples, split_base_ids):
        sample_base_ids = samples[:, 7].astype(np.int64)
        train_ids       = set(split_base_ids["train"])
        validation_ids  = set(split_base_ids["val"])
        test_ids        = set(split_base_ids["test"])

        train_mask      = np.fromiter((base_id in train_ids      for base_id in sample_base_ids), dtype=bool, count=len(sample_base_ids))
        validation_mask = np.fromiter((base_id in validation_ids for base_id in sample_base_ids), dtype=bool, count=len(sample_base_ids))
        test_mask       = np.fromiter((base_id in test_ids       for base_id in sample_base_ids), dtype=bool, count=len(sample_base_ids))

        return samples[train_mask], samples[validation_mask], samples[test_mask]

    def _make_dataset(self, samples, augmentation, stats):
        graph_builder = Graph(self.config)
        return GraphDataset(samples, self.geometry_positions, self.light_matrix, graph_builder, self.config.physics, augmentation=augmentation, stats=stats)

    def _augmentation(self):
        augmentation = Augmentation(self.config.augmentation)
        return augmentation if augmentation.active else None

    def _fit_stats(self, train_samples, split_names):
        if self.stats is not None:
            self.logger.subsection("Normalization stats reused from caller (skipping fit)")
            return

        clean_train_dataset = self._make_dataset(train_samples, augmentation=None, stats=None)
        reusable            = "train" in split_names and self._augmentation() is None

        if reusable:
            self.raw_train_graphs = CachedGraphDataset(clean_train_dataset, self.logger).graphs
            stats_source          = self.raw_train_graphs
        else:
            stats_source = clean_train_dataset

        self.stats = StatsEstimator(stats_source, self.config.data.stats_sample_size, self.logger, self.config).fit()

    def _train_dataset(self, train_samples, augmentation):
        if self.raw_train_graphs is not None:
            base_dataset          = self._make_dataset(train_samples, None, self.stats)
            normalized_graphs     = GraphNormalizer(self.stats).apply_all(self.raw_train_graphs)
            self.raw_train_graphs = None
            return CachedGraphDataset(base_dataset, self.logger, graphs=normalized_graphs)

        dataset = self._make_dataset(train_samples, augmentation, self.stats)
        if augmentation is not None and not self.evaluation_mode:
            return dataset
        return CachedGraphDataset(dataset, self.logger)

    def _build_datasets(self, train_samples, validation_samples, test_samples, split_names):
        augmentation = self._augmentation()
        live_train   = augmentation is not None and not self.evaluation_mode

        self.logger.kv_table({
            "Train samples" : len(train_samples),
            "Val samples"   : len(validation_samples),
            "Test samples"  : len(test_samples),
            "Built splits"  : ", ".join(split_names),
            "Augmentation"  : "every split" if augmentation is not None else "off",
            "Train caching" : "live (resampled per epoch)" if live_train else "cached",
        }, title="Dataset Splits")

        self.datasets = {}
        if "train" in split_names:
            self.datasets["train"] = self._train_dataset(train_samples, augmentation)
        if "val" in split_names:
            self.datasets["val"] = CachedGraphDataset(self._make_dataset(validation_samples, augmentation, self.stats), self.logger)
        if "test" in split_names:
            self.datasets["test"] = CachedGraphDataset(self._make_dataset(test_samples, augmentation, self.stats), self.logger)

    @staticmethod
    def build_loaders(datasets, batch_size, num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2, seed=0):
        persistent     = persistent_workers and num_workers > 0
        worker_options = {"prefetch_factor": prefetch_factor, "worker_init_fn": Reproducibility.worker_init(seed)} if num_workers > 0 else {}

        train_loader = GraphDataLoader(datasets["train"], batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, generator=Reproducibility.generator(seed), **worker_options)
        val_loader   = GraphDataLoader(datasets["val"],   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, **worker_options)
        test_loader  = GraphDataLoader(datasets["test"],  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent, **worker_options)
        return train_loader, val_loader, test_loader

    @staticmethod
    def inject_feature_dimensions(model_overrides, dataset_config):
        node_dimension, edge_dimension, graph_dimension = FeatureSchema(dataset_config).dimensions()
        return {**model_overrides, "input_dim": node_dimension, "edge_dim": edge_dimension, "graph_dim": graph_dimension}

    def pna_degree_histogram(self, model_name, dataset):
        if model_name != "pna":
            return None
        return DegreeHistogramEstimator(dataset, self.config.data.stats_sample_size, self.logger).fit()

    def base_ids_for_indices(self, train_indices, validation_indices, test_indices):
        sample_base_ids = self.samples[:, 7].astype(np.int64)
        return {
            "train" : np.unique(sample_base_ids[train_indices]).tolist(),
            "val"   : np.unique(sample_base_ids[validation_indices]).tolist(),
            "test"  : np.unique(sample_base_ids[test_indices]).tolist(),
        }

    def run_with_indices(self, train_indices, validation_indices, test_indices):
        self.logger.section("[Dataset Pipeline | Fold]")
        self.prepare_samples()

        train_base         = self.samples[train_indices]
        validation_samples = self.samples[validation_indices]
        test_samples       = self.samples[test_indices]

        self.stats           = None
        self.split_base_ids  = self.base_ids_for_indices(train_indices, validation_indices, test_indices)

        train_samples = self._prepare_train_samples(train_base)
        self._fit_stats(train_samples, self.SPLIT_NAMES)
        self._build_datasets(train_samples, validation_samples, test_samples, self.SPLIT_NAMES)
        return self.datasets, self.stats

    def run_with_base_ids(self, split_base_ids, split_names=None):
        self.logger.section("[Dataset Pipeline | Persisted Split]")
        self.prepare_samples()

        requested           = self.validate_split_names(split_names) if split_names is not None else self.SPLIT_NAMES
        self.split_base_ids = dict(split_base_ids)

        train_base, validation_samples, test_samples = self._partition_by_base_ids(self.samples, split_base_ids)
        train_samples = self._prepare_train_samples(train_base)
        self._fit_stats(train_samples, requested)
        self._build_datasets(train_samples, validation_samples, test_samples, requested)
        return self.datasets, self.stats

    def build_evaluation_split(self, split_base_ids, split_name):
        if self.stats is None:
            raise ValueError("build_evaluation_split requires normalization stats supplied to the pipeline; physics-only re-evaluation must reuse the training-split statistics.")

        self.prepare_samples()

        self.validate_split_names((split_name,))
        partitions    = dict(zip(self.SPLIT_NAMES, self._partition_by_base_ids(self.samples, split_base_ids)))
        split_samples = partitions[split_name]
        split_dataset = self._make_dataset(split_samples, self._augmentation(), self.stats)
        return CachedGraphDataset(split_dataset, self.logger)

    def run_split_samples(self):
        self.logger.section("[Dataset Pipeline | Split Samples]")
        self.prepare_samples()

        train_base, validation_samples, test_samples = self._split_samples(self.samples)
        train_samples = self._prepare_train_samples(train_base)
        return {"train": train_samples, "val": validation_samples, "test": test_samples}

    def run(self):
        self.logger.section("[Dataset Pipeline]")
        self.prepare_samples()

        train_base, validation_samples, test_samples = self._split_samples(self.samples)
        train_samples = self._prepare_train_samples(train_base)
        self._fit_stats(train_samples, self.SPLIT_NAMES)
        self._build_datasets(train_samples, validation_samples, test_samples, self.SPLIT_NAMES)
        return self.datasets, self.stats
