from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    EPOCH_SEED_STRIDE = 1_000_003

    def __init__(self, samples, geometry_positions, light_matrix, graph_builder, physics_config, augmentation=None, stats=None):
        self.samples            = samples
        self.geometry_positions = geometry_positions.astype(np.float32)
        self.light_matrix       = light_matrix
        self.graph_builder      = graph_builder
        self.augmentation       = augmentation
        self.stats              = stats

        self.scale_factor         = physics_config.scale_factor
        self.detection_efficiency = physics_config.detection_efficiency
        self.efficiency_seed      = physics_config.efficiency_seed
        self.augmentation_seed    = augmentation.config.seed if augmentation is not None else 0

        self.epoch = mp.Value("q", 0, lock=False)

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch.value = int(epoch)

    def _base_light(self, raw_counts, base_event_id):
        generator     = np.random.default_rng(self.efficiency_seed + int(base_event_id))
        scaled_counts = np.maximum(np.round(raw_counts * self.scale_factor), 0).astype(np.int64)
        return generator.binomial(scaled_counts, self.detection_efficiency).astype(np.float32)

    def _light_for_sample(self, light_row, base_event_id):
        raw_counts = self.light_matrix[base_event_id].astype(np.float64)
        light      = self._base_light(raw_counts, base_event_id)

        if self.augmentation is not None and self.augmentation.active:
            augmentation_seed = self.augmentation_seed + int(light_row) + self.epoch.value * self.EPOCH_SEED_STRIDE
            generator         = np.random.default_rng(augmentation_seed)
            light             = self.augmentation.apply_to_counts(light, generator)
            light             = self.augmentation.apply_to_light(light, generator)

        return light

    def _normalize(self, data):
        data.x         = torch.tensor(self.stats.node.forward_numpy(data.x.numpy()),         dtype=torch.float32)
        data.edge_attr = torch.tensor(self.stats.edge.forward_numpy(data.edge_attr.numpy()), dtype=torch.float32)
        data.y         = torch.tensor(self.stats.target.forward_numpy(data.y.numpy()),       dtype=torch.float32)
        return data

    def __getitem__(self, index):
        sample        = self.samples[index]
        light_row     = int(sample[0])
        signs         = sample[1:4].astype(np.float32)
        target        = sample[4:7].astype(np.float32)
        base_event_id = int(sample[7])

        positions = self.geometry_positions * signs[None, :]
        light     = self._light_for_sample(light_row, base_event_id)

        data   = self.graph_builder.build_from_arrays(positions, light)
        data.y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        if self.stats is not None:
            data = self._normalize(data)

        return data


class CachedGraphDataset(Dataset):
    def __init__(self, base_dataset, logger):
        self.base_dataset = base_dataset
        self.logger       = logger
        self.graphs       = self._materialize()

    @property
    def samples(self):
        return self.base_dataset.samples

    def _materialize(self):
        self.logger.section("[Graph Cache]")
        graphs = [None] * len(self.base_dataset)

        with self.logger.track() as progress:
            task_id = progress.add_task("Caching deterministic graphs", total=len(self.base_dataset))
            for index in range(len(self.base_dataset)):
                graphs[index] = self.base_dataset[index]
                progress.advance(task_id)

        self.logger.subsection(f"Cached {len(graphs)} graphs")
        return graphs

    def set_epoch(self, epoch):
        return None

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


class StatsEstimator:
    def __init__(self, dataset, sample_size, logger):
        self.dataset     = dataset
        self.sample_size = sample_size
        self.logger      = logger

    def fit(self):
        from pipelines.dataset.normalization import FeatureGroupNormalizer, NormalizationStats

        self.logger.section("[Normalization Fit]")
        count   = min(self.sample_size, len(self.dataset))
        indices = np.linspace(0, len(self.dataset) - 1, count).astype(np.int64)

        node_segments   = []
        edge_segments   = []
        target_segments = []

        with self.logger.track() as progress:
            task_id = progress.add_task("Estimating normalization statistics", total=len(indices))
            for index in indices:
                data = self.dataset[int(index)]
                node_segments.append(data.x.numpy())
                edge_segments.append(data.edge_attr.numpy())
                target_segments.append(data.y.numpy())
                progress.advance(task_id)

        node_group   = FeatureGroupNormalizer.fit(np.concatenate(node_segments, axis=0))
        edge_group   = FeatureGroupNormalizer.fit(np.concatenate(edge_segments, axis=0))
        target_group = FeatureGroupNormalizer.fit(np.concatenate(target_segments, axis=0))

        self.logger.subsection(f"Fitted normalization on {count} events")

        rows = []
        for group_name, group in [("node", node_group), ("edge", edge_group), ("target", target_group)]:
            for strategy, channel_count in sorted(group.strategy_counts().items()):
                rows.append({"Group": group_name, "Strategy": strategy, "Channels": channel_count})
        self.logger.metrics_table(rows, ["Group", "Strategy", "Channels"], title="Normalization Strategies")

        return NormalizationStats(node_group, edge_group, target_group)


class DegreeHistogramEstimator:
    def __init__(self, dataset, sample_size, logger):
        self.dataset     = dataset
        self.sample_size = sample_size
        self.logger      = logger

    def fit(self):
        from torch_geometric.utils import degree

        self.logger.section("[PNA Degree Histogram]")
        count   = min(self.sample_size, len(self.dataset))
        indices = np.linspace(0, len(self.dataset) - 1, count).astype(np.int64)

        degree_counts = torch.zeros(1, dtype=torch.long)
        for index in indices:
            data           = self.dataset[int(index)]
            node_in_degree = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            maximum_degree = int(node_in_degree.max().item()) if node_in_degree.numel() else 0

            if maximum_degree + 1 > degree_counts.numel():
                expanded                       = torch.zeros(maximum_degree + 1, dtype=torch.long)
                expanded[: degree_counts.numel()] = degree_counts
                degree_counts                  = expanded

            degree_counts += torch.bincount(node_in_degree, minlength=degree_counts.numel())

        self.logger.subsection(f"Degree histogram over {count} graphs: {degree_counts.tolist()}")
        return tuple(int(value) for value in degree_counts.tolist())
