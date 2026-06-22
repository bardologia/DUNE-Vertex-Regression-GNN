from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class HotChannelCorrector:
    def __init__(self, geometry_positions, config, logger):
        self.positions       = np.asarray(geometry_positions, dtype=np.float64)
        self.logger          = logger

        self.enabled         = config.enabled
        self.active_fraction = config.active_fraction
        self.median_factor   = config.median_factor
        self.neighbor_count  = int(config.neighbor_count)
        self.min_events      = int(config.min_events)

    def _detect(self, light_matrix):
        active_fraction = (light_matrix > 0).mean(axis=0)
        global_median   = float(np.median(light_matrix[light_matrix > 0]))
        threshold       = self.median_factor * global_median

        channel_medians = np.zeros(light_matrix.shape[1], dtype=np.float64)
        for channel in range(light_matrix.shape[1]):
            lit = light_matrix[light_matrix[:, channel] > 0, channel]
            if lit.size:
                channel_medians[channel] = np.median(lit)

        return np.where((active_fraction >= self.active_fraction) & (channel_medians >= threshold))[0]

    def _neighbor_indices(self, hot_channels):
        healthy_mask               = np.ones(self.positions.shape[0], dtype=bool)
        healthy_mask[hot_channels] = False
        healthy_indices            = np.where(healthy_mask)[0]

        tree              = cKDTree(self.positions[healthy_indices])
        _, neighbor_ranks = tree.query(self.positions[hot_channels], k=self.neighbor_count)
        return healthy_indices[neighbor_ranks]

    def fit(self, light_matrix):
        hot_channels = self._detect(light_matrix)
        if hot_channels.size == 0:
            return hot_channels, np.empty((0, self.neighbor_count), dtype=np.int64)

        neighbor_indices = self._neighbor_indices(hot_channels)
        return hot_channels, neighbor_indices

    def _replace(self, light_matrix, hot_channels, neighbor_indices):
        for position, channel in enumerate(hot_channels):
            neighbor_light           = light_matrix[:, neighbor_indices[position]]
            light_matrix[:, channel] = np.rint(neighbor_light.mean(axis=1)).astype(light_matrix.dtype)
        return light_matrix

    def apply_to_event(self, light_row, hot_channels, neighbor_indices):
        if hot_channels.size == 0:
            return np.array(light_row, dtype=np.float32)

        single    = np.array(light_row, dtype=np.float64, copy=True).reshape(1, -1)
        corrected = self._replace(single, hot_channels, neighbor_indices)
        return corrected.reshape(-1).astype(np.float32)

    def run(self, light_matrix):
        if not self.enabled:
            return light_matrix

        if light_matrix.shape[0] < self.min_events:
            self.logger.subsection(f"Hot-channel correction skipped ({light_matrix.shape[0]} < {self.min_events} events)")
            return light_matrix

        hot_channels, neighbor_indices = self.fit(light_matrix)
        if hot_channels.size == 0:
            self.logger.subsection("Hot-channel correction: no hot channels detected")
            return light_matrix

        light_matrix = self._replace(light_matrix, hot_channels, neighbor_indices)

        self.logger.subsection(f"Hot-channel correction: {hot_channels.size} channels replaced by {self.neighbor_count}-neighbour mean -> {hot_channels.tolist()}")
        return light_matrix
