from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):
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

    def __len__(self):
        return len(self.samples)

    def _base_light(self, raw_counts, event_id):
        generator    = np.random.default_rng(self.efficiency_seed + int(event_id))
        scaled_counts = np.maximum(np.round(raw_counts * self.scale_factor), 0).astype(np.int64)
        return generator.binomial(scaled_counts, self.detection_efficiency).astype(np.float32)

    def _light_for_sample(self, event_id):
        raw_counts = self.light_matrix[event_id].astype(np.float64)
        light      = self._base_light(raw_counts, event_id)

        if self.augmentation is not None and self.augmentation.active:
            generator = np.random.default_rng()
            light     = self.augmentation.apply_to_counts(light, generator)
            light     = self.augmentation.apply_to_light(light, generator)

        return light

    def _normalize(self, data):
        data.x         = torch.tensor(self.stats.node.forward_numpy(data.x.numpy()),         dtype=torch.float32)
        data.edge_attr = torch.tensor(self.stats.edge.forward_numpy(data.edge_attr.numpy()), dtype=torch.float32)
        data.y         = torch.tensor(self.stats.target.forward_numpy(data.y.numpy()),       dtype=torch.float32)
        return data

    def __getitem__(self, index):
        sample    = self.samples[index]
        event_id  = int(sample[0])
        signs     = sample[1:4].astype(np.float32)
        target    = sample[4:7].astype(np.float32)

        positions = self.geometry_positions * signs[None, :]
        light     = self._light_for_sample(event_id)

        data   = self.graph_builder.build_from_arrays(positions, light)
        data.y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        if self.stats is not None:
            data = self._normalize(data)

        return data


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
        return NormalizationStats(node_group, edge_group, target_group)
