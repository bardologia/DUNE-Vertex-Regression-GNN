from pathlib import Path
from collections import deque
import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import ray
from .logger import Logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")


class DataLoader:
    def __init__(self, logger: Logger = None, config = None):
        self.logger              = logger
        self.config              = config
        self.input_dir           = config.data.input_dir
        self.max_files           = config.data.max_files
        
        self.light_column        = config.data.light_column
        self.coordinate_columns  = config.data.coordinate_columns
        self.header_rows_to_skip = config.data.header_rows_to_skip

    def _ensure_ray_initialized(self):
        if ray.is_initialized():
            return

        num_cpus = None
        if self.config is not None and hasattr(self.config, "ray"):
            num_cpus = self.config.ray.num_cpus

        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            logging_level="ERROR",
            num_cpus=num_cpus,
            runtime_env={"working_dir": str(PROJECT_ROOT)},
        )

    @staticmethod
    def _shutdown_ray():
        if ray.is_initialized():
            ray.shutdown()

    def _get_ray_batch_size(self, total_items: int, stage: str | None = None) -> int:
        configured_cpus = self._get_ray_cpu_count()
        target_tasks_per_cpu = 8
        max_batch_size = None
        if self.config is not None and hasattr(self.config, "ray"):
            target_tasks_per_cpu = self.config.ray.batch_tasks_per_cpu
            max_batch_size = self.config.ray.max_batch_size
            if stage == "scaling":
                target_tasks_per_cpu = self.config.ray.scaling_batch_tasks_per_cpu
                max_batch_size = self.config.ray.scaling_max_batch_size

        target_tasks = max(1, int(configured_cpus) * int(target_tasks_per_cpu))
        batch_size = max(1, int(np.ceil(total_items / target_tasks)))
        if max_batch_size is not None:
            batch_size = min(batch_size, max(1, int(max_batch_size)))
        return batch_size

    def _get_ray_cpu_count(self) -> int:
        configured_cpus = None
        if self.config is not None and hasattr(self.config, "ray"):
            configured_cpus = self.config.ray.num_cpus

        if configured_cpus is None:
            if ray.is_initialized():
                configured_cpus = max(1, int(ray.available_resources().get("CPU", 1)))
            else:
                configured_cpus = max(1, os.cpu_count() or 1)

        return max(1, int(configured_cpus))

    def _get_max_inflight_tasks(self, total_batches: int, stage: str | None = None) -> int:
        inflight_per_cpu = 2
        if self.config is not None and hasattr(self.config, "ray"):
            inflight_per_cpu = self.config.ray.max_inflight_tasks_per_cpu
            if stage == "scaling":
                inflight_per_cpu = self.config.ray.scaling_max_inflight_tasks_per_cpu

        max_inflight = self._get_ray_cpu_count() * int(inflight_per_cpu)
        return max(1, min(total_batches, max_inflight))

    def _run_ray_batches(self, remote_function, batch_arguments, progress_description: str, stage: str | None = None):
        total_batches = len(batch_arguments)
        if total_batches == 0:
            return []

        pending_arguments = deque(enumerate(batch_arguments))
        in_flight = {}
        completed_results = {}
        max_inflight = self._get_max_inflight_tasks(total_batches, stage=stage)

        while pending_arguments and len(in_flight) < max_inflight:
            batch_index, batch_args = pending_arguments.popleft()
            in_flight[remote_function.remote(*batch_args)] = batch_index

        with self.logger.progress_task(progress_description, total_batches) as (progress, task_id):
            while in_flight:
                ready_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1)
                ready_ref = ready_refs[0]
                batch_index = in_flight.pop(ready_ref)
                completed_results[batch_index] = ray.get(ready_ref)
                progress.advance(task_id)

                if pending_arguments:
                    next_index, next_args = pending_arguments.popleft()
                    in_flight[remote_function.remote(*next_args)] = next_index

        return [completed_results[index] for index in range(total_batches)]

    @staticmethod
    def _extract_coordinates_from_filename(file_path: Path):
        filename = file_path.stem
        parts = filename.split('_')
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        return (x, y, z)

    def parse(self, file_path: Path):
        coordinates = self._extract_coordinates_from_filename(file_path)
   
        dataframe = pd.read_csv(
            file_path,
            skiprows=self.header_rows_to_skip,
            names=["bin", "x", "y", "z", self.light_column],
            comment="b",
            dtype={"x": float, "y": float, "z": float, self.light_column: float},
        )
        return dataframe, coordinates

    def load(self, input_directory=None):
        input_dir = Path(input_directory) if input_directory else self.input_dir
        all_files = list(Path(input_dir).glob("*.csv"))
        if self.max_files is None or self.max_files >= len(all_files):
            file_list = all_files
        else:
            rng = np.random.default_rng(self.config.split.random_state)
            indices = rng.choice(len(all_files), size=self.max_files, replace=False)
            file_list = [all_files[i] for i in indices]

        self.logger.section("[Data Loading]")
        self.logger.subsection(f"Loading {len(file_list)} files")

        if len(file_list) == 0:
            coordinate_dataframe = pd.DataFrame(columns=list(self.coordinate_columns))
            self.logger.subsection("No CSV files found. Returning empty dataset.\n")
            return [], coordinate_dataframe

        self._ensure_ray_initialized()
        self.logger.subsection("Ray enabled for CSV parsing")

        def parse_file_batch_remote(file_path_batch, header_rows_to_skip: int, light_column: str):
            batch_results = []
            for file_path_str in file_path_batch:
                file_path = Path(file_path_str)
                filename = file_path.stem
                parts = filename.split('_')
                coordinates = (float(parts[1]), float(parts[2]), float(parts[3]))

                dataframe = pd.read_csv(
                    file_path,
                    skiprows=header_rows_to_skip,
                    names=["bin", "x", "y", "z", light_column],
                    comment="b",
                    dtype={"x": float, "y": float, "z": float, light_column: float},
                )
                batch_results.append((dataframe, coordinates))
            return batch_results

        batch_size = self._get_ray_batch_size(len(file_list))
        self.logger.subsection(f"Ray batch size for loading: {batch_size}")
        parse_file_batch_remote_ray = ray.remote(parse_file_batch_remote)
        file_path_strs = [str(file_path) for file_path in file_list]
        file_batches = [file_path_strs[i:i + batch_size] for i in range(0, len(file_path_strs), batch_size)]
        self.logger.subsection(f"Ray max in-flight loading batches: {self._get_max_inflight_tasks(len(file_batches))}")

        batch_results = self._run_ray_batches(
            parse_file_batch_remote_ray,
            [(file_batch, self.header_rows_to_skip, self.light_column) for file_batch in file_batches],
            "Loading files (Ray batches)",
        )

        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        dataframes      = [df for df, _ in results]
        coordinate_rows = [coord for _, coord in results]
            
        coordinate_dataframe = pd.DataFrame(coordinate_rows, columns=list(self.coordinate_columns))
        self.logger.subsection(f"Loaded {len(dataframes)} dataframes with coordinates \n")
        return dataframes, coordinate_dataframe


class DataProcessor:
    def __init__(self, logger: Logger = None, config = None):
        self.logger = logger
        self.config = config
        self.light_column = config.data.light_column if config is not None else "light_amount"

    def _ensure_ray_initialized(self):
        if ray.is_initialized():
            return

        num_cpus = None
        if self.config is not None and hasattr(self.config, "ray"):
            num_cpus = self.config.ray.num_cpus

        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
            logging_level="ERROR",
            num_cpus=num_cpus,
            runtime_env={"working_dir": str(PROJECT_ROOT)},
        )

    @staticmethod
    def _shutdown_ray():
        if ray.is_initialized():
            ray.shutdown()

    def _get_ray_batch_size(self, total_items: int, stage: str | None = None) -> int:
        configured_cpus = self._get_ray_cpu_count()
        target_tasks_per_cpu = 8
        max_batch_size = None
        if self.config is not None and hasattr(self.config, "ray"):
            target_tasks_per_cpu = self.config.ray.batch_tasks_per_cpu
            max_batch_size = self.config.ray.max_batch_size
            if stage == "scaling":
                target_tasks_per_cpu = self.config.ray.scaling_batch_tasks_per_cpu
                max_batch_size = self.config.ray.scaling_max_batch_size

        target_tasks = max(1, int(configured_cpus) * int(target_tasks_per_cpu))
        batch_size = max(1, int(np.ceil(total_items / target_tasks)))
        if max_batch_size is not None:
            batch_size = min(batch_size, max(1, int(max_batch_size)))
        return batch_size

    def _get_ray_cpu_count(self) -> int:
        configured_cpus = None
        if self.config is not None and hasattr(self.config, "ray"):
            configured_cpus = self.config.ray.num_cpus

        if configured_cpus is None:
            if ray.is_initialized():
                configured_cpus = max(1, int(ray.available_resources().get("CPU", 1)))
            else:
                configured_cpus = max(1, os.cpu_count() or 1)

        return max(1, int(configured_cpus))

    def _get_max_inflight_tasks(self, total_batches: int, stage: str | None = None) -> int:
        inflight_per_cpu = 2
        if self.config is not None and hasattr(self.config, "ray"):
            inflight_per_cpu = self.config.ray.max_inflight_tasks_per_cpu
            if stage == "scaling":
                inflight_per_cpu = self.config.ray.scaling_max_inflight_tasks_per_cpu

        max_inflight = self._get_ray_cpu_count() * int(inflight_per_cpu)
        return max(1, min(total_batches, max_inflight))

    def _run_ray_batches(self, remote_function, batch_arguments, progress_description: str, stage: str | None = None):
        total_batches = len(batch_arguments)
        if total_batches == 0:
            return []

        pending_arguments = deque(enumerate(batch_arguments))
        in_flight = {}
        completed_results = {}
        max_inflight = self._get_max_inflight_tasks(total_batches, stage=stage)

        while pending_arguments and len(in_flight) < max_inflight:
            batch_index, batch_args = pending_arguments.popleft()
            in_flight[remote_function.remote(*batch_args)] = batch_index

        with self.logger.progress_task(progress_description, total_batches) as (progress, task_id):
            while in_flight:
                ready_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1)
                ready_ref = ready_refs[0]
                batch_index = in_flight.pop(ready_ref)
                completed_results[batch_index] = ray.get(ready_ref)
                progress.advance(task_id)

                if pending_arguments:
                    next_index, next_args = pending_arguments.popleft()
                    in_flight[remote_function.remote(*next_args)] = next_index

        return [completed_results[index] for index in range(total_batches)]
    
    def detect_spatial_outliers(self, dataframes, zscore_threshold, radius):
        self.logger.section("[Spatial Outlier Detection]")
        self.logger.subsection(f"zscore_threshold={zscore_threshold:.2f} | radius={radius:.2f}")

        def detect_outliers_single(dataframe, zscore_threshold, radius, light_column):
            modified        = dataframe.copy()
            coordinates     = modified[["x", "y", "z"]].values
            tree            = cKDTree(coordinates)
            light_values    = modified[light_column].values.copy()
            number_of_sensors = coordinates.shape[0]
            all_neighbors   = tree.query_ball_point(coordinates, r=radius, p=2.0)

            owner_segments         = []
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

            neighbor_count       = np.bincount(flat_owner_indices, minlength=number_of_sensors).astype(np.float64)
            neighbor_light_sum   = np.bincount(flat_owner_indices, weights=flat_neighbor_light, minlength=number_of_sensors)
            neighbor_light_sum_sq = np.bincount(flat_owner_indices, weights=flat_neighbor_light * flat_neighbor_light, minlength=number_of_sensors)

            has_neighbors    = neighbor_count > 0
            safe_count       = np.where(has_neighbors, neighbor_count, 1.0)
            neighbor_mean    = neighbor_light_sum / safe_count
            neighbor_variance = neighbor_light_sum_sq / safe_count - neighbor_mean * neighbor_mean
            neighbor_std     = np.sqrt(np.maximum(neighbor_variance, 0.0))

            current_light     = light_values.astype(np.float64)
            valid_std         = has_neighbors & (neighbor_std > 0)
            zscore            = np.zeros(number_of_sensors, dtype=np.float64)
            zscore[valid_std] = np.abs(current_light[valid_std] - neighbor_mean[valid_std]) / neighbor_std[valid_std]
            outlier_mask      = valid_std & (zscore > zscore_threshold)

            light_values[outlier_mask] = neighbor_mean[outlier_mask].astype(light_values.dtype)
            outlier_indices            = np.nonzero(outlier_mask)[0].tolist()

            modified[light_column] = light_values
            return modified, dataframe.iloc[outlier_indices] if outlier_indices else pd.DataFrame()

        if len(dataframes) == 0:
            self.logger.subsection("No dataframes to process for outlier detection.\n")
            return [], []

        self._ensure_ray_initialized()
        self.logger.subsection("Ray enabled for outlier detection")

        def detect_outliers_batch_remote(dataframe_batch, zscore_threshold_value, radius_value, light_column):
            cleaned_batch = []
            outlier_batch = []
            for dataframe in dataframe_batch:
                cleaned_dataframe, outlier_dataframe = detect_outliers_single(
                    dataframe,
                    zscore_threshold_value,
                    radius_value,
                    light_column,
                )
                cleaned_batch.append(cleaned_dataframe)
                outlier_batch.append(outlier_dataframe)
            return cleaned_batch, outlier_batch

        batch_size = self._get_ray_batch_size(len(dataframes))
        self.logger.subsection(f"Ray batch size for outlier detection: {batch_size}")
        detect_outliers_batch_remote_ray = ray.remote(detect_outliers_batch_remote)
        dataframe_batches = [dataframes[i:i + batch_size] for i in range(0, len(dataframes), batch_size)]
        self.logger.subsection(f"Ray max in-flight outlier batches: {self._get_max_inflight_tasks(len(dataframe_batches))}")

        batch_results = self._run_ray_batches(
            detect_outliers_batch_remote_ray,
            [(batch, zscore_threshold, radius, self.light_column) for batch in dataframe_batches],
            "Detecting outliers (Ray batches)",
        )

        all_cleaned = []
        all_outliers = []
        for cleaned_batch, outlier_batch in batch_results:
            all_cleaned.extend(cleaned_batch)
            all_outliers.extend(outlier_batch)

        self.logger.subsection(f"Completed outlier detection. Cleaned: {len(all_cleaned)} | Outliers: {sum(len(o) for o in all_outliers)} \n")
        return all_cleaned, all_outliers

    def scale_light_intensity(self, dataframes, scale_factor):
        self.logger.section("[Scaling Light Intensity]")
        self.logger.subsection(f"scale_factor={scale_factor:.2f}")

        self.logger.subsection(f"Scaling {len(dataframes)} dataframes")
        if len(dataframes) == 0:
            self.logger.subsection("No dataframes to scale.\n")
            return []

        self._ensure_ray_initialized()
        self.logger.subsection("Ray enabled for intensity scaling")

        def scale_light_intensity_batch_remote(dataframe_batch, scale_factor_value, light_column):
            scaled_batch = []
            for dataframe in dataframe_batch:
                modified = dataframe.copy()
                modified[light_column] *= scale_factor_value
                scaled_batch.append(modified)
            return scaled_batch

        batch_size = self._get_ray_batch_size(len(dataframes), stage="scaling")
        self.logger.subsection(f"Ray batch size for scaling: {batch_size}")
        scale_light_intensity_batch_remote_ray = ray.remote(scale_light_intensity_batch_remote)
        dataframe_batches = [dataframes[i:i + batch_size] for i in range(0, len(dataframes), batch_size)]
        self.logger.subsection(f"Ray max in-flight scaling batches: {self._get_max_inflight_tasks(len(dataframe_batches), stage='scaling')}")

        batch_results = self._run_ray_batches(
            scale_light_intensity_batch_remote_ray,
            [(batch, scale_factor, self.light_column) for batch in dataframe_batches],
            "Scaling intensity (Ray batches)",
            stage="scaling",
        )

        scaled = []
        for scaled_batch in batch_results:
            scaled.extend(scaled_batch)
        
        self.logger.subsection(f"Completed scaling of light intensity for {len(scaled)} dataframes \n")
        return scaled
    
    def apply_detection_efficiency(self, dataframes, efficiency, seed=42):
        self.logger.section("[Applying Detection Efficiency]")
        self.logger.subsection(f"efficiency={efficiency:.2f} | seed={seed}")
        
        if len(dataframes) == 0:
            self.logger.subsection("No dataframes to process for detection efficiency.\n")
            return []

        self._ensure_ray_initialized()
        self.logger.subsection("Ray enabled for detection efficiency")

        def apply_detection_efficiency_batch_remote(indexed_dataframe_batch, efficiency_value, seed_value, light_column):
            processed_batch = []
            for dataframe_index, dataframe in indexed_dataframe_batch:
                random_generator = np.random.default_rng(seed_value + dataframe_index)
                modified = dataframe.copy()
                trials = modified[light_column].round().astype(int).clip(0)
                modified[light_column] = random_generator.binomial(trials, efficiency_value)
                processed_batch.append(modified)
            return processed_batch

        batch_size = self._get_ray_batch_size(len(dataframes))
        self.logger.subsection(f"Ray batch size for detection efficiency: {batch_size}")
        apply_detection_efficiency_batch_remote_ray = ray.remote(apply_detection_efficiency_batch_remote)

        indexed_dataframes = list(enumerate(dataframes))
        indexed_batches = [indexed_dataframes[i:i + batch_size] for i in range(0, len(indexed_dataframes), batch_size)]
        self.logger.subsection(f"Ray max in-flight efficiency batches: {self._get_max_inflight_tasks(len(indexed_batches))}")

        batch_results = self._run_ray_batches(
            apply_detection_efficiency_batch_remote_ray,
            [(batch, efficiency, seed, self.light_column) for batch in indexed_batches],
            "Applying efficiency (Ray batches)",
        )

        processed = []
        for processed_batch in batch_results:
            processed.extend(processed_batch)
        
        self.logger.subsection(f"Completed applying detection efficiency for {len(processed)} dataframes \n")
        return processed

    def apply_log_transform(self, dataframes):
        self.logger.section("[Applying Log Transform]")
        self.logger.subsection("Transform: log1p to light column")
        
        if len(dataframes) == 0:
            self.logger.subsection("No dataframes to transform with log1p.\n")
            return []

        self._ensure_ray_initialized()
        self.logger.subsection("Ray enabled for log transform")

        def apply_log_transform_batch_remote(dataframe_batch, light_column):
            updated_batch = []
            for dataframe in dataframe_batch:
                modified = dataframe.copy()
                modified[light_column] = np.log1p(np.maximum(modified[light_column].fillna(0), 0))
                updated_batch.append(modified)
            return updated_batch

        batch_size = self._get_ray_batch_size(len(dataframes))
        self.logger.subsection(f"Ray batch size for log transform: {batch_size}")
        apply_log_transform_batch_remote_ray = ray.remote(apply_log_transform_batch_remote)
        dataframe_batches = [dataframes[i:i + batch_size] for i in range(0, len(dataframes), batch_size)]
        self.logger.subsection(f"Ray max in-flight log-transform batches: {self._get_max_inflight_tasks(len(dataframe_batches))}")

        batch_results = self._run_ray_batches(
            apply_log_transform_batch_remote_ray,
            [(batch, self.light_column) for batch in dataframe_batches],
            "Applying log1p (Ray batches)",
        )

        updated = []
        for updated_batch in batch_results:
            updated.extend(updated_batch)
        
        self.logger.subsection(f"Completed log transform for {len(updated)} dataframes \n")
        return updated


__all__ = [
    "DataLoader",
    "DataProcessor",
]
