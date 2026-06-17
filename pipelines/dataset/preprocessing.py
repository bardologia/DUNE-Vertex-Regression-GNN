from __future__ import annotations

import numpy as np
import pandas as pd
import ray
from scipy.spatial import cKDTree

from pipelines.dataset.ray_executor import RayExecutor


class DataProcessor:
    def __init__(self, logger, config):
        self.logger       = logger
        self.config       = config
        self.executor     = RayExecutor(logger, config.ray)
        self.light_column = config.data.light_column

    def detect_spatial_outliers(self, dataframes, zscore_threshold, radius):
        self.logger.section("[Spatial Outlier Detection]")
        self.logger.subsection(f"zscore_threshold={zscore_threshold:.2f} | radius={radius:.2f}")

        if len(dataframes) == 0:
            return [], []

        self.executor.ensure_initialized()

        def detect_outliers_single(dataframe, zscore_threshold_value, radius_value, light_column):
            modified          = dataframe.copy()
            coordinates       = modified[["x", "y", "z"]].values
            tree              = cKDTree(coordinates)
            light_values      = modified[light_column].values.copy()
            number_of_sensors = coordinates.shape[0]
            all_neighbors     = tree.query_ball_point(coordinates, r=radius_value, p=2.0)

            owner_segments          = []
            neighbor_light_segments = []
            for sensor_index, neighbor_indices in enumerate(all_neighbors):
                filtered_neighbor_indices = np.fromiter((neighbor_index for neighbor_index in neighbor_indices if neighbor_index != sensor_index), dtype=np.int64)
                if filtered_neighbor_indices.size > 0:
                    owner_segments.append(np.full(filtered_neighbor_indices.size, sensor_index, dtype=np.int64))
                    neighbor_light_segments.append(light_values[filtered_neighbor_indices])

            if owner_segments:
                flat_owner_indices  = np.concatenate(owner_segments)
                flat_neighbor_light = np.concatenate(neighbor_light_segments).astype(np.float64)
            else:
                flat_owner_indices  = np.empty(0, dtype=np.int64)
                flat_neighbor_light = np.empty(0, dtype=np.float64)

            neighbor_count        = np.bincount(flat_owner_indices, minlength=number_of_sensors).astype(np.float64)
            neighbor_light_sum    = np.bincount(flat_owner_indices, weights=flat_neighbor_light, minlength=number_of_sensors)
            neighbor_light_sum_sq = np.bincount(flat_owner_indices, weights=flat_neighbor_light * flat_neighbor_light, minlength=number_of_sensors)

            has_neighbors     = neighbor_count > 0
            safe_count        = np.where(has_neighbors, neighbor_count, 1.0)
            neighbor_mean     = neighbor_light_sum / safe_count
            neighbor_variance = neighbor_light_sum_sq / safe_count - neighbor_mean * neighbor_mean
            neighbor_std      = np.sqrt(np.maximum(neighbor_variance, 0.0))

            current_light     = light_values.astype(np.float64)
            valid_std         = has_neighbors & (neighbor_std > 0)
            zscore            = np.zeros(number_of_sensors, dtype=np.float64)
            zscore[valid_std] = np.abs(current_light[valid_std] - neighbor_mean[valid_std]) / neighbor_std[valid_std]
            outlier_mask      = valid_std & (zscore > zscore_threshold_value)

            light_values[outlier_mask] = neighbor_mean[outlier_mask].astype(light_values.dtype)
            outlier_indices            = np.nonzero(outlier_mask)[0].tolist()

            modified[light_column] = light_values
            return modified, dataframe.iloc[outlier_indices] if outlier_indices else pd.DataFrame()

        def detect_outliers_batch(dataframe_batch, zscore_threshold_value, radius_value, light_column):
            cleaned_batch = []
            outlier_batch = []
            for dataframe in dataframe_batch:
                cleaned_dataframe, outlier_dataframe = detect_outliers_single(dataframe, zscore_threshold_value, radius_value, light_column)
                cleaned_batch.append(cleaned_dataframe)
                outlier_batch.append(outlier_dataframe)
            return cleaned_batch, outlier_batch

        remote_detect     = ray.remote(detect_outliers_batch)
        dataframe_batches = self.executor.make_batches(dataframes)
        batch_results     = self.executor.run_batches(
            remote_detect,
            [(batch, zscore_threshold, radius, self.light_column) for batch in dataframe_batches],
            "Detecting outliers (Ray batches)",
        )

        all_cleaned  = []
        all_outliers = []
        for cleaned_batch, outlier_batch in batch_results:
            all_cleaned.extend(cleaned_batch)
            all_outliers.extend(outlier_batch)

        self.logger.subsection(f"Cleaned: {len(all_cleaned)} | Outliers: {sum(len(outlier) for outlier in all_outliers)}")
        return all_cleaned, all_outliers

    def scale_light_intensity(self, dataframes, scale_factor):
        self.logger.section("[Scaling Light Intensity]")
        self.logger.subsection(f"scale_factor={scale_factor:.2f}")

        if len(dataframes) == 0:
            return []

        self.executor.ensure_initialized()

        def scale_batch(dataframe_batch, scale_factor_value, light_column):
            scaled_batch = []
            for dataframe in dataframe_batch:
                modified                = dataframe.copy()
                modified[light_column] *= scale_factor_value
                scaled_batch.append(modified)
            return scaled_batch

        remote_scale      = ray.remote(scale_batch)
        dataframe_batches = self.executor.make_batches(dataframes, stage="scaling")
        batch_results     = self.executor.run_batches(
            remote_scale,
            [(batch, scale_factor, self.light_column) for batch in dataframe_batches],
            "Scaling intensity (Ray batches)",
            stage="scaling",
        )

        scaled = []
        for scaled_batch in batch_results:
            scaled.extend(scaled_batch)

        self.logger.subsection(f"Scaled {len(scaled)} dataframes")
        return scaled

    def apply_detection_efficiency(self, dataframes, efficiency, seed=42):
        self.logger.section("[Applying Detection Efficiency]")
        self.logger.subsection(f"efficiency={efficiency:.2f} | seed={seed}")

        if len(dataframes) == 0:
            return []

        self.executor.ensure_initialized()

        def efficiency_batch(indexed_dataframe_batch, efficiency_value, seed_value, light_column):
            processed_batch = []
            for dataframe_index, dataframe in indexed_dataframe_batch:
                random_generator       = np.random.default_rng(seed_value + dataframe_index)
                modified               = dataframe.copy()
                trials                 = modified[light_column].round().astype(int).clip(0)
                modified[light_column] = random_generator.binomial(trials, efficiency_value)
                processed_batch.append(modified)
            return processed_batch

        remote_efficiency  = ray.remote(efficiency_batch)
        indexed_dataframes = list(enumerate(dataframes))
        indexed_batches    = self.executor.make_batches(indexed_dataframes)
        batch_results      = self.executor.run_batches(
            remote_efficiency,
            [(batch, efficiency, seed, self.light_column) for batch in indexed_batches],
            "Applying efficiency (Ray batches)",
        )

        processed = []
        for processed_batch in batch_results:
            processed.extend(processed_batch)

        self.logger.subsection(f"Applied detection efficiency to {len(processed)} dataframes")
        return processed

    def apply_log_transform(self, dataframes):
        self.logger.section("[Applying Log Transform]")
        self.logger.subsection("Transform: log1p on light column")

        if len(dataframes) == 0:
            return []

        self.executor.ensure_initialized()

        def log_transform_batch(dataframe_batch, light_column):
            updated_batch = []
            for dataframe in dataframe_batch:
                modified               = dataframe.copy()
                modified[light_column] = np.log1p(np.maximum(modified[light_column].fillna(0), 0))
                updated_batch.append(modified)
            return updated_batch

        remote_log        = ray.remote(log_transform_batch)
        dataframe_batches = self.executor.make_batches(dataframes)
        batch_results     = self.executor.run_batches(
            remote_log,
            [(batch, self.light_column) for batch in dataframe_batches],
            "Applying log1p (Ray batches)",
        )

        updated = []
        for updated_batch in batch_results:
            updated.extend(updated_batch)

        self.logger.subsection(f"Applied log transform to {len(updated)} dataframes")
        return updated
