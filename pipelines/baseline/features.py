from __future__ import annotations

import numpy as np

from pipelines.dataset.augmentation  import Augmentation
from pipelines.dataset.graph         import GraphAssembler
from pipelines.dataset.graph_dataset import LightRealization
from pipelines.dataset.pipeline      import DatasetPipeline


class SummaryFeatureExtractor(GraphAssembler):
    AXES = ("x", "y", "z")

    def __init__(self, feature_config):
        self.top_channel_count = int(feature_config.top_channel_count)
        self.quantiles         = tuple(float(quantile) for quantile in feature_config.quantiles)

    def names(self):
        feature_names  = ["total_light", "active_count", "max_light", "max_light_fraction"]
        feature_names += [f"centroid_{axis}"       for axis in self.AXES]
        feature_names += [f"sharp_centroid_{axis}" for axis in self.AXES]
        feature_names += [f"spread_{axis}"         for axis in self.AXES]
        feature_names += [f"skew_{axis}"           for axis in self.AXES]
        feature_names += [f"inertia_eigenvalue_{index}" for index in (1, 2, 3)]
        feature_names += [f"brightest_offset_{axis}" for axis in self.AXES]
        feature_names += ["brightest_offset_norm"]

        for quantile in self.quantiles:
            feature_names += [f"position_{axis}_q{int(round(quantile * 100)):02d}" for axis in self.AXES]

        for rank in range(1, self.top_channel_count + 1):
            feature_names += [f"top{rank}_{axis}" for axis in self.AXES]
            feature_names += [f"top{rank}_light_fraction"]

        return feature_names

    def _scale_block(self, light, total_light):
        return np.array([total_light, float(light.shape[0]), float(light.max()), float(light.max()) / total_light], dtype=np.float64)

    def _sharp_centroid(self, positions, light):
        sharp_weights = light.astype(np.float64) ** 2
        return (sharp_weights[:, None] * positions).sum(axis=0) / (sharp_weights.sum() + self.EPSILON)

    def _axis_moments(self, positions, light, total_light, centroid):
        centered = positions - centroid[None, :]
        weights  = light.astype(np.float64)[:, None]

        variance = (weights * centered ** 2).sum(axis=0) / total_light
        spread   = np.sqrt(np.clip(variance, 0.0, None))
        third    = (weights * centered ** 3).sum(axis=0) / total_light
        skew     = third / (spread ** 3 + self.EPSILON)

        return spread, skew

    def _brightest_offset(self, positions, light, centroid):
        offset = positions[int(np.argmax(light))] - centroid
        return np.concatenate([offset, [np.linalg.norm(offset)]])

    def _weighted_quantiles(self, positions, light):
        blocks = []
        for axis_index in range(positions.shape[1]):
            order             = np.argsort(positions[:, axis_index], kind="stable")
            cumulative_weight = np.cumsum(light[order].astype(np.float64))
            targets           = np.asarray(self.quantiles) * cumulative_weight[-1]
            indices           = np.minimum(np.searchsorted(cumulative_weight, targets), positions.shape[0] - 1)
            blocks.append(positions[order][indices, axis_index])
        return np.stack(blocks, axis=1).reshape(-1)

    def _top_channels(self, positions, light, total_light, centroid):
        brightest_first = np.argsort(light, kind="stable")[::-1][: self.top_channel_count]

        block = []
        for rank in range(self.top_channel_count):
            if rank < brightest_first.shape[0]:
                index = brightest_first[rank]
                block.append(np.concatenate([positions[index], [light[index] / total_light]]))
            else:
                block.append(np.concatenate([centroid, [0.0]]))
        return np.concatenate(block)

    def extract(self, positions, light):
        positions = np.asarray(positions, dtype=np.float64)
        light     = np.maximum(np.asarray(light, dtype=np.float64), 0.0)

        active = light > 0.0
        if not active.any():
            raise ValueError("No sensor carries light; summary features are undefined. Raise physics.detection_efficiency or physics.scale_factor.")

        positions   = positions[active]
        light       = light[active]
        total_light = light.sum() + self.EPSILON

        centroid     = self._light_centroid(positions, light, total_light)
        spread, skew = self._axis_moments(positions, light, total_light, centroid)

        blocks = [
            self._scale_block(light, total_light),
            centroid,
            self._sharp_centroid(positions, light),
            spread,
            skew,
            self._light_covariance_eigenvalues(positions, light, total_light),
            self._brightest_offset(positions, light, centroid),
            self._weighted_quantiles(positions, light),
            self._top_channels(positions, light, total_light, centroid),
        ]
        return np.concatenate(blocks).astype(np.float32)


class TabularSplitBuilder:
    def __init__(self, dataset_config, feature_config, logger):
        self.config    = dataset_config
        self.extractor = SummaryFeatureExtractor(feature_config)
        self.logger    = logger

        self.feature_names  = self.extractor.names()
        self.split_base_ids = None

    def _features_for_split(self, split_name, samples, geometry_positions, realizer, augmentation):
        features = np.empty((len(samples), len(self.feature_names)), dtype=np.float32)
        targets  = samples[:, 4:7].astype(np.float32).copy()

        with self.logger.track() as progress:
            task_id = progress.add_task(f"Extracting {split_name} features", total=len(samples))
            for index, sample in enumerate(samples):
                positions       = geometry_positions * sample[1:4][None, :]
                light           = realizer.realize(int(sample[0]), int(sample[7]), augmentation)
                features[index] = self.extractor.extract(positions, light)
                progress.advance(task_id)

        return features, targets

    def run(self):
        self.logger.section("[Tabular Feature Extraction]")

        dataset_pipeline    = DatasetPipeline(self.config, self.logger)
        split_samples       = dataset_pipeline.run_split_samples()
        self.split_base_ids = dataset_pipeline.split_base_ids

        realizer     = LightRealization(dataset_pipeline.light_matrix, self.config.physics)
        augmentation = Augmentation(self.config.augmentation)
        splits       = {name: self._features_for_split(name, samples, dataset_pipeline.geometry_positions, realizer, augmentation) for name, samples in split_samples.items()}

        self.logger.kv_table({
            "Features"      : len(self.feature_names),
            "Train rows"    : len(splits["train"][0]),
            "Val rows"      : len(splits["val"][0]),
            "Test rows"     : len(splits["test"][0]),
            "Octants"       : self.config.data.augment_octants,
            "Augmentation"  : "on" if self.config.augmentation.enabled else "off",
        }, title="Tabular Splits")

        return splits
