import sys
from pathlib import Path
from collections import deque
import os
from types import SimpleNamespace
import numpy as np
import torch
import ray
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data  
import pickle
import math
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")
sys.path.insert(0, str(PROJECT_ROOT))

from core.data import DataLoader, DataProcessor
from core.logger import Logger


class Graph:
    def __init__(self, config):
        self.config           = config
        self.bidirectional    = config.graph.bidirectional
        self.k_neighbors      = config.graph.k_neighbors
        self.knn_algorithm    = config.graph.knn_algorithm
        self.light_column     = config.data.light_column
        self.position_columns = config.data.position_columns

    def _create_nodes(self, event_dataframe):
        light_col     = self.light_column
        position_cols = list(self.position_columns)
        
        positions       = event_dataframe[position_cols].values.astype(np.float32)
        light_intensity = event_dataframe[light_col].fillna(0.0).values.astype(np.float32)
        light_intensity = np.maximum(light_intensity, 0.0)

        hit_flag = (light_intensity > 0.0).astype(np.float32)

        total_light    = light_intensity.sum() + 1e-8
        light_fraction = (light_intensity / total_light).astype(np.float32)

        centroid = (light_intensity[:, None] * positions).sum(axis=0) / total_light
        displacement_to_centroid = positions - centroid[None, :]
        dist_to_centroid = np.linalg.norm(displacement_to_centroid, axis=1).astype(np.float32)

        effective_k = max(1, min(int(self.k_neighbors), positions.shape[0] - 1))
        knn_local   = NearestNeighbors(n_neighbors=effective_k + 1, algorithm=self.knn_algorithm)
        knn_local.fit(positions)
        _, neighbor_idx = knn_local.kneighbors(positions)
        local_light_density = light_intensity[neighbor_idx[:, 1:]].sum(axis=1).astype(np.float32)

        node_features = np.column_stack([
            positions,
            light_intensity.reshape(-1, 1),
            hit_flag.reshape(-1, 1),
            light_fraction.reshape(-1, 1),
            dist_to_centroid.reshape(-1, 1),
            local_light_density.reshape(-1, 1),
        ])

        x = torch.tensor(node_features, dtype=torch.float32)
        return x, event_dataframe

    def _create_edges(self, node_dataframe):
        position_columns = list(self.position_columns)
        light_col        = self.light_column
        positions_array  = node_dataframe[position_columns].values.astype(np.float32)
        light_array      = node_dataframe[light_col].fillna(0.0).values.astype(np.float32)
        light_array      = np.maximum(light_array, 0.0)

        number_of_nodes = int(positions_array.shape[0])
        effective_k     = max(1, min(int(self.k_neighbors), number_of_nodes - 1))

        knn = NearestNeighbors(n_neighbors=effective_k + 1, algorithm=self.knn_algorithm)
        knn.fit(positions_array)
        neighbor_distances_array, neighbor_indices_array = knn.kneighbors(positions_array)

        neighbor_indices_without_self   = neighbor_indices_array[:, 1:]
        neighbor_distances_without_self = neighbor_distances_array[:, 1:]

        source_node_indices      = np.repeat(np.arange(number_of_nodes), neighbor_indices_without_self.shape[1])
        destination_node_indices = neighbor_indices_without_self.reshape(-1).astype(np.int64)
        distance                 = neighbor_distances_without_self.reshape(-1).astype(np.float64)

        source_positions      = positions_array[source_node_indices]
        destination_positions = positions_array[destination_node_indices]
        source_light          = light_array[source_node_indices].astype(np.float64)
        destination_light     = light_array[destination_node_indices].astype(np.float64)

        position_difference     = destination_positions - source_positions
        normalization           = np.maximum(distance, 1e-8)
        displacement            = position_difference / normalization[:, None]
        inverse_square_distance = 1.0 / (distance ** 2 + 1e-8)
        light_gradient          = destination_light - source_light
        light_similarity        = 2.0 * np.minimum(source_light, destination_light) / (source_light + destination_light + 1e-8)

        forward_features = np.column_stack([
            distance,
            displacement[:, 0],
            displacement[:, 1],
            displacement[:, 2],
            inverse_square_distance,
            light_gradient,
            light_similarity,
        ]).astype(np.float32)

        if self.bidirectional:
            reverse_features = np.column_stack([
                distance,
                -displacement[:, 0],
                -displacement[:, 1],
                -displacement[:, 2],
                inverse_square_distance,
                -light_gradient,
                light_similarity,
            ]).astype(np.float32)

            interleaved_source_node_indices      = np.empty(source_node_indices.shape[0] * 2, dtype=np.int64)
            interleaved_destination_node_indices = np.empty(destination_node_indices.shape[0] * 2, dtype=np.int64)
            interleaved_source_node_indices[0::2]      = source_node_indices
            interleaved_source_node_indices[1::2]      = destination_node_indices
            interleaved_destination_node_indices[0::2] = destination_node_indices
            interleaved_destination_node_indices[1::2] = source_node_indices

            interleaved_features = np.empty((forward_features.shape[0] * 2, forward_features.shape[1]), dtype=np.float32)
            interleaved_features[0::2] = forward_features
            interleaved_features[1::2] = reverse_features

            edge_index = torch.tensor(np.vstack([interleaved_source_node_indices, interleaved_destination_node_indices]), dtype=torch.long)
            edge_attr  = torch.tensor(interleaved_features, dtype=torch.float32)
            return edge_index, edge_attr

        edge_index = torch.tensor(np.vstack([source_node_indices.astype(np.int64), destination_node_indices]), dtype=torch.long)
        edge_attr  = torch.tensor(forward_features, dtype=torch.float32)

        return edge_index, edge_attr

    def build_graph(self, event_dataframe):
        x, node_df = self._create_nodes(event_dataframe)
        edge_index, edge_attr = self._create_edges(node_df)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class DatasetBuilder:
    def __init__(self, output_dir, logger: Logger, config):
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.zscore_threshold     = config.preprocessing.zscore_threshold
        self.radius               = config.preprocessing.radius
        self.scale_factor         = config.preprocessing.scale_factor
        self.detection_efficiency = config.preprocessing.detection_efficiency
        self.random_state         = config.split.random_state
        self.coordinate_columns   = config.data.coordinate_columns
    
        self.logger.section("[Dataset Builder]")
        self.logger.subsection(f"Graph bidirectional : {config.graph.bidirectional}")
        self.logger.subsection(f"Graph k_neighbors   : {config.graph.k_neighbors}")
        self.logger.subsection(f"Graph knn_algorithm : {config.graph.knn_algorithm}")

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

    def _get_ray_batch_size(self, total_items: int) -> int:
        configured_cpus = self._get_ray_cpu_count()
        target_tasks_per_cpu = 8
        max_batch_size = None
        if self.config is not None and hasattr(self.config, "ray"):
            target_tasks_per_cpu = self.config.ray.batch_tasks_per_cpu
            max_batch_size = self.config.ray.max_batch_size

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

    def _get_max_inflight_tasks(self, total_batches: int) -> int:
        inflight_per_cpu = 2
        if self.config is not None and hasattr(self.config, "ray"):
            inflight_per_cpu = self.config.ray.max_inflight_tasks_per_cpu

        max_inflight = self._get_ray_cpu_count() * int(inflight_per_cpu)
        return max(1, min(total_batches, max_inflight))

    def _run_ray_batches(self, remote_function, batch_arguments, progress_description: str):
        total_batches = len(batch_arguments)
        if total_batches == 0:
            return []

        pending_arguments = deque(enumerate(batch_arguments))
        in_flight = {}
        completed_results = {}
        max_inflight = self._get_max_inflight_tasks(total_batches)

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

    def _load_data(self):
        self.logger.subsection("Loading data")
        loader = DataLoader(logger=self.logger, config=self.config)
        dataframes, target_df = loader.load()
        self.logger.subsection(f"Loaded {len(dataframes)} events")
        return dataframes, target_df
    
    def _process_data(self, dataframes):
        self.logger.subsection("Processing data")
        preprocessor = DataProcessor(logger=self.logger, config=self.config)

        dataframes, outliers = preprocessor.detect_spatial_outliers(
            dataframes, 
            self.zscore_threshold,
            self.radius
        )

        total_outliers = sum(len(df) for df in outliers)
        self.logger.subsection(f"Total outliers removed: {total_outliers}")

        dataframes = preprocessor.scale_light_intensity(
            dataframes, 
            self.scale_factor
        )
        
        dataframes = preprocessor.apply_detection_efficiency(
            dataframes, 
            self.detection_efficiency, 
            seed=self.random_state
        )

        dataframes = preprocessor.apply_log_transform(dataframes)

        return dataframes

    def _build_graphs(self, dataframes):
        self.logger.subsection("Building graphs")
        self.logger.subsection(f"Building {len(dataframes)} graphs")

        if len(dataframes) == 0:
            self.logger.subsection("No events to build graphs from.\n")
            return []

        self._ensure_ray_initialized()

        runtime_config = SimpleNamespace(
            graph=SimpleNamespace(
                bidirectional = self.config.graph.bidirectional,
                k_neighbors   = self.config.graph.k_neighbors,
                knn_algorithm = self.config.graph.knn_algorithm,
            ),
            data=SimpleNamespace(
                light_column     = self.config.data.light_column,
                position_columns = self.config.data.position_columns,
            ),
        )

        def build_graph_batch_remote(event_batch, graph_config):
            graph_builder = Graph(config=graph_config)
            return [graph_builder.build_graph(event_dataframe) for event_dataframe in event_batch]

        batch_size = self._get_ray_batch_size(len(dataframes))
        self.logger.subsection(f"Ray batch size for graph building: {batch_size}")
        build_graph_batch_remote_ray = ray.remote(build_graph_batch_remote)
        dataframe_batches = [dataframes[i:i + batch_size] for i in range(0, len(dataframes), batch_size)]
        self.logger.subsection(f"Ray max in-flight graph batches: {self._get_max_inflight_tasks(len(dataframe_batches))}")

        batch_results = self._run_ray_batches(
            build_graph_batch_remote_ray,
            [(batch, runtime_config) for batch in dataframe_batches],
            "Building graphs (Ray batches)",
        )

        graphs = []
        for graph_batch in batch_results:
            graphs.extend(graph_batch)

        num_nodes = [graph.x.shape[0] for graph in graphs]
        num_edges = [graph.edge_index.shape[1] for graph in graphs]
        
        self.logger.subsection(f"Graphs : {len(graphs)}")
        self.logger.subsection(f"Nodes  : {np.mean(num_nodes):.0f}±{np.std(num_nodes):.0f}")
        self.logger.subsection(f"Edges  : {np.mean(num_edges):.0f}±{np.std(num_edges):.0f} \n")
        return graphs

    def _build_dataset(self, graphs, target_df):
        self.logger.subsection("Building dataset")

        if len(graphs) == 0:
            self.logger.subsection("No graphs available for target attachment.\n")
            return []

        coord_cols = list(self.coordinate_columns)
        target_values = target_df[coord_cols].values.astype(np.float32)

        for i, graph_object in enumerate(self.logger.track(graphs, "Attaching targets")):
            graph_object.y = torch.tensor(target_values[i], dtype=torch.float32).unsqueeze(0)

        self.logger.subsection(f"Dataset built with {len(graphs)} graphs \n")
        return graphs

    def _save_artifacts(self, dataset, target_df):
        self.logger.subsection("Saving artifacts")
        target_path = self.output_dir / "target.pkl"
        config_path = self.output_dir / "config.pkl"
        
        chunk_size = 100  
        self.logger.subsection(f"Saving dataset in chunks of size {chunk_size}")
        num_chunks = math.ceil(len(dataset) / chunk_size)
        self.logger.subsection(f"Saving dataset in {num_chunks} chunks")
        
        chunks_dir = self.output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        for chunk_idx in self.logger.track(range(num_chunks), "Saving chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(dataset))
            chunk = dataset[start_idx:end_idx]
            
            chunk_path = self.output_dir / "chunks" / f"dataset_chunk_{chunk_idx}.pt"
            torch.save(chunk, chunk_path)
        
        metadata = {
            'num_chunks'   : num_chunks,
            'chunk_size'   : chunk_size,
            'total_graphs' : len(dataset)
        }
        
        metadata_path = self.output_dir / "dataset_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        with open(target_path, 'wb') as f:
            pickle.dump(target_df, f)

        config_dict = {
            'data_config'          : self.config.data,
            'graph_config'         : self.config.graph,
            'preprocessing_config' : self.config.preprocessing,
            'split_config'         : self.config.split,
            'dataset_len'          : len(dataset),
            'num_targets'          : len(target_df),
        }
        
        with open(config_path, 'wb') as f:
            pickle.dump(config_dict, f)
        
        self.logger.subsection(f"Artifacts saved: {self.output_dir.name} ({num_chunks} chunks) \n")

    def run(self):
        self.logger.section("[Dataset Builder]")
        self.logger.subsection(f"Output directory: {self.output_dir}")

        self._ensure_ray_initialized()
        self.logger.subsection("Ray initialized for the full dataset build")

        try:
            dataframes, target_df = self._load_data()
            dataframes            = self._process_data(dataframes)
            graphs                = self._build_graphs(dataframes)
            del dataframes
            dataset               = self._build_dataset(graphs, target_df)
            self.logger.subsection(f"Dataset built: {len(dataset)} graphs \n")
            self._save_artifacts(dataset, target_df)
        finally:
            self._shutdown_ray()

        self.logger.subsection("Dataset building process completed successfully \n")


class DatasetNormalizer:
    def __init__(self, logger: Logger, source_split = "train"):
        self.logger       = logger
        self.source_split = source_split
        self.node_mean    = None
        self.node_std     = None
        self.edge_mean    = None
        self.edge_std     = None
        self.target_mean  = None
        self.target_std   = None

    def compute_stats(self, splits):
        source_graphs = splits.get(self.source_split, [])
        self._compute_node_statistics(source_graphs)
        self._compute_edge_statistics(source_graphs)
        self._compute_target_statistics(source_graphs)
        self.logger.subsection(f"Computed statistics from '{self.source_split}' split \n")

    def _compute_node_statistics(self, source_graphs):
        self.logger.section("[Node statistics]")
        running_sum   = None
        running_count = 0

        for graph in source_graphs:
            if graph.x.numel() > 0:
                features = graph.x.numpy().astype(np.float64)
                if running_sum is None:
                    running_sum = np.zeros(features.shape[1], dtype=np.float64)
                running_sum   += features.sum(axis=0)
                running_count += features.shape[0]

        self.node_mean = (running_sum / running_count).astype(np.float32)

        running_sq = np.zeros_like(running_sum)
        for graph in source_graphs:
            if graph.x.numel() > 0:
                diff = graph.x.numpy().astype(np.float64) - self.node_mean
                running_sq += (diff ** 2).sum(axis=0)

        self.node_std = np.sqrt(running_sq / running_count).astype(np.float32) + 1e-8

        labels = [f"F{i}" for i in range(len(self.node_mean))]
        self.logger.print_stats_table("Node Statistics", labels, self.node_mean, self.node_std)

    def _compute_edge_statistics(self, source_graphs):
        self.logger.section("[Edge statistics]")
        running_sum   = None
        running_count = 0

        for graph in source_graphs:
            if graph.edge_attr.numel() > 0:
                attrs = graph.edge_attr.numpy().astype(np.float64)
                if running_sum is None:
                    running_sum = np.zeros(attrs.shape[1], dtype=np.float64)
                running_sum   += attrs.sum(axis=0)
                running_count += attrs.shape[0]

        self.edge_mean = (running_sum / running_count).astype(np.float32)

        running_sq = np.zeros_like(running_sum)
        for graph in source_graphs:
            if graph.edge_attr.numel() > 0:
                diff = graph.edge_attr.numpy().astype(np.float64) - self.edge_mean
                running_sq += (diff ** 2).sum(axis=0)

        self.edge_std = np.sqrt(running_sq / running_count).astype(np.float32) + 1e-8

        labels = [f"F{i}" for i in range(len(self.edge_mean))]
        self.logger.print_stats_table("Edge Statistics", labels, self.edge_mean, self.edge_std)
        
    def _compute_target_statistics(self, source_graphs):
        self.logger.section("[Target statistics]")
        running_sum   = None
        running_count = 0

        for graph in source_graphs:
            if hasattr(graph, 'y') and graph.y is not None and graph.y.numel() > 0:
                tgt = graph.y.detach().cpu().numpy().astype(np.float64).reshape(1, -1)
                if running_sum is None:
                    running_sum = np.zeros(tgt.shape[1], dtype=np.float64)
                running_sum   += tgt.sum(axis=0)
                running_count += tgt.shape[0]

        self.target_mean = (running_sum / running_count).astype(np.float32)

        running_sq = np.zeros_like(running_sum)
        for graph in source_graphs:
            if hasattr(graph, 'y') and graph.y is not None and graph.y.numel() > 0:
                tgt  = graph.y.detach().cpu().numpy().astype(np.float64).reshape(1, -1)
                diff = tgt - self.target_mean
                running_sq += (diff ** 2).sum(axis=0)

        self.target_std = np.sqrt(running_sq / running_count).astype(np.float32) + 1e-8

        self.logger.print_stats_table("Target Statistics", ["x", "y", "z"], self.target_mean, self.target_std)
        
    def normalize(self, splits):
        self.logger.section("[Normalizing splits]")
        for split_name, graphs in splits.items():
            for graph in graphs:
                graph.x = (graph.x - torch.tensor(self.node_mean, dtype=torch.float32)) / torch.tensor(self.node_std, dtype=torch.float32)
                graph.edge_attr = (graph.edge_attr - torch.tensor(self.edge_mean, dtype=torch.float32)) / torch.tensor(self.edge_std, dtype=torch.float32)

                mean_t  = torch.tensor(self.target_mean, dtype=torch.float32)
                std_t   = torch.tensor(self.target_std,  dtype=torch.float32)
                graph.y = (graph.y - mean_t) / std_t

        train_graphs = splits[self.source_split]

        def _streaming_mean_std(graphs, attr):
            running_sum   = None
            running_count = 0
            for g in graphs:
                arr = getattr(g, attr).numpy().astype(np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if running_sum is None:
                    running_sum = np.zeros(arr.shape[1], dtype=np.float64)
                running_sum   += arr.sum(axis=0)
                running_count += arr.shape[0]
            mean = running_sum / running_count
            running_sq = np.zeros_like(running_sum)
            for g in graphs:
                arr = getattr(g, attr).numpy().astype(np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                running_sq += ((arr - mean) ** 2).sum(axis=0)
            std = np.sqrt(running_sq / running_count)
            return mean.astype(np.float32), std.astype(np.float32)

        node_after   = _streaming_mean_std(train_graphs, 'x')
        edge_after   = _streaming_mean_std(train_graphs, 'edge_attr')
        target_after = _streaming_mean_std(train_graphs, 'y')
        
        node_labels   = [f"F{i}" for i in range(len(self.node_mean))]
        edge_labels   = [f"F{i}" for i in range(len(self.edge_mean))]

        self.logger.print_normalization_table(
            "Node Normalization",
            node_labels, self.node_mean, self.node_std, node_after[0], node_after[1],
        )
        self.logger.print_normalization_table(
            "Edge Normalization",
            edge_labels, self.edge_mean, self.edge_std, edge_after[0], edge_after[1],
        )
        self.logger.print_normalization_table(
            "Target Normalization",
            ["x", "y", "z"], self.target_mean, self.target_std, target_after[0], target_after[1],
        )

        return splits


class DatasetLoader:
    def __init__(self, preprocessed_dir, logger: Logger, config=None):
        self.config               = config
        self.logger               = logger
        self.coordinate_columns   = config.data.coordinate_columns
        self.preprocessed_dir     = Path(preprocessed_dir)

    @staticmethod
    def _resolve_split_size(total_size, split_size):
        if isinstance(split_size, float):
            return int(np.ceil(total_size * split_size))

        return int(split_size)

    def _build_stratification_labels(self, Y, n_bins):
        if n_bins <= 1:
            return np.zeros(len(Y), dtype=np.int64)

        binned_coords = []
        for dim in range(Y.shape[1]):
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins        = np.percentile(Y[:, dim], percentiles)
            bins        = np.unique(bins)

            if len(bins) <= 1:
                binned = np.zeros(len(Y), dtype=np.int64)
            else:
                binned = np.digitize(Y[:, dim], bins[:-1]) - 1
                binned = np.clip(binned, 0, len(bins) - 2)

            binned_coords.append(binned)

        binned_coords = np.array(binned_coords).T
        n_bins_actual = max(len(np.unique(bc)) for bc in binned_coords.T)
        strat_labels  = (binned_coords[:, 0] * n_bins_actual**2 + binned_coords[:, 1] * n_bins_actual + binned_coords[:, 2])
        return strat_labels.astype(np.int64)

    def _find_valid_stratification(self, Y, split_config):
        N = len(Y)
        max_bins = max(1, min(10, N // 10))
        test_size_count = self._resolve_split_size(N, split_config.test_size)
        val_size_count  = self._resolve_split_size(N, split_config.val_ratio)

        for n_bins in range(max_bins, 0, -1):
            strat_labels = self._build_stratification_labels(Y, n_bins)
            unique_labels, label_counts = np.unique(strat_labels, return_counts=True)

            if len(unique_labels) > test_size_count or len(unique_labels) > val_size_count:
                continue

            if label_counts.min() < 2:
                continue

            sss_test = StratifiedShuffleSplit(
                n_splits=1,
                test_size=split_config.test_size,
                random_state=split_config.random_state,
            )

            try:
                train_val_idx, test_idx = next(sss_test.split(np.arange(N), strat_labels))
            except ValueError:
                continue

            val_ratio_adjusted = split_config.val_ratio / (1 - split_config.test_size)
            sss_val = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_ratio_adjusted,
                random_state=split_config.random_state,
            )

            try:
                train_idx, val_idx = next(sss_val.split(train_val_idx, strat_labels[train_val_idx]))
            except ValueError:
                continue

            return n_bins, strat_labels, train_val_idx, test_idx, train_idx, val_idx

        raise ValueError("Unable to construct a valid stratified split for the current dataset and split configuration.")

    def _balance_targets(self, target_df):   
        self.logger.section("[Balancing targets]")
        
        Y = target_df[list(self.coordinate_columns)].to_numpy(dtype=np.float32)
        N = len(Y)

        coord_names = list(self.coordinate_columns)

        before_data = {
            f"Full (N={N})": {
                col: (
                    float(Y[:, i].mean()),
                    float(Y[:, i].std()),
                    float(Y[:, i].min()),
                    float(Y[:, i].max()),
                )
                for i, col in enumerate(coord_names)
            }
        }
        self.logger.print_distribution_table("Target Distribution \u2014 Before Balancing", coord_names, before_data)

        self.logger.subsection("Performing stratified split")
        split_config = self.config.split

        n_bins, strat_labels, train_val_idx, test_idx, train_idx, val_idx = self._find_valid_stratification(Y, split_config)

        self.logger.subsection(f"Binning into {n_bins} bins per dimension | {len(np.unique(strat_labels))} unique stratification groups")

        train_idx = train_val_idx[train_idx]
        val_idx   = train_val_idx[val_idx]

        self.logger.subsection(f"Split sizes: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

        after_data = {}
        for split_name, indices in [
            (f"Train (N={len(train_idx)})", train_idx),
            (f"Val   (N={len(val_idx)})",   val_idx),
            (f"Test  (N={len(test_idx)})",  test_idx),
        ]:
            after_data[split_name] = {
                col: (
                    float(Y[indices, i].mean()),
                    float(Y[indices, i].std()),
                    float(Y[indices, i].min()),
                    float(Y[indices, i].max()),
                )
                for i, col in enumerate(coord_names)
            }
        self.logger.print_distribution_table("Target Distribution \u2014 After Balancing", coord_names, after_data)

        group_counts = " | ".join(
            f"{name}: {len(np.unique(strat_labels[idx]))}"
            for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]
        )
        self.logger.subsection(f"Stratification groups \u2014 {group_counts}")

        return train_idx, val_idx, test_idx, Y, N

    def _attach_targets(self, graph_splits, idx_splits, Y: np.ndarray):
        self.logger.section("[Attaching targets to graphs]")
        train_graphs, val_graphs, test_graphs = graph_splits
        train_idx,    val_idx,    test_idx    = idx_splits

        def _attach(graph_list, indices):
            for gi, ii in enumerate(indices):
                graph_list[gi].y = torch.tensor(Y[ii], dtype=torch.float32).unsqueeze(0)
            dataset = graph_list
            return dataset

        train_dataset = _attach(train_graphs, train_idx)
        val_dataset   = _attach(val_graphs,   val_idx)
        test_dataset  = _attach(test_graphs,  test_idx)

        self.logger.subsection(f"Targets attached to splits: train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)} \n")
        return train_dataset, val_dataset, test_dataset

    def _normalize(self, splits):
        self.logger.subsection("Normalizing dataset")
        normalizer = DatasetNormalizer(logger=self.logger, source_split="train")
        normalizer.compute_stats(splits)
        splits = normalizer.normalize(splits)

        metrics = {
            "node_mean"  : normalizer.node_mean,
            "node_std"   : normalizer.node_std,
            "edge_mean"  : normalizer.edge_mean,
            "edge_std"   : normalizer.edge_std,
            "target_mean": normalizer.target_mean,
            "target_std" : normalizer.target_std,
        }

        self.logger.subsection("Dataset normalization completed \n")
        return splits, metrics

    def _load_artifacts(self):
        self.logger.section("[Loading artifacts]")  
        tp = self.preprocessed_dir / "target.pkl"
        cp = self.preprocessed_dir / "config.pkl"
        
        metadata_path = self.preprocessed_dir / "dataset_metadata.pkl"
  
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        num_chunks = metadata['num_chunks']
        graphs = []
        
        self.logger.subsection(f"Loading dataset from {num_chunks} chunks")
        for chunk_idx in self.logger.track(range(num_chunks), "Loading chunks"):
            chunk_path = self.preprocessed_dir / "chunks" / f"dataset_chunk_{chunk_idx}.pt"
            chunk = torch.load(chunk_path, weights_only=False)
            graphs.extend(chunk)
     
        with open(tp, "rb") as f: target_df = pickle.load(f)
        with open(cp, "rb") as f: config    = pickle.load(f)
        
        subset_fraction = getattr(self.config.data, 'subset_fraction', 1.0)
        if 0.0 < subset_fraction < 1.0:
            n = max(1, int(len(graphs) * subset_fraction))
            rng = np.random.RandomState(self.config.split.random_state)
            indices = np.sort(rng.choice(len(graphs), size=n, replace=False))
            graphs    = [graphs[i] for i in indices]
            target_df = target_df.iloc[indices].reset_index(drop=True)
            self.logger.subsection(f"Subset fraction {subset_fraction}: using {n} graphs")

        self.logger.subsection(f"Loaded artifacts from: {self.preprocessed_dir} \n")
        return graphs, target_df, config

    def run(self):
        self.logger.section("[Dataset Loader]")
        graphs, target_df, saved_config = self._load_artifacts()
        
        train_idx, val_idx, test_idx, Y, N = self._balance_targets(target_df)
        idx = (train_idx, val_idx, test_idx)
        
        train_balanced_graphs = [graphs[i] for i in train_idx]
        val_balanced_graphs   = [graphs[i] for i in val_idx]
        test_balanced_graphs  = [graphs[i] for i in test_idx]
        
        balanced_graphs = (train_balanced_graphs, val_balanced_graphs, test_balanced_graphs)
        
        raw_train_dataset, raw_val_dataset, raw_test_dataset = self._attach_targets(balanced_graphs, idx, Y)
        raw_split = {"train": raw_train_dataset, "val": raw_val_dataset, "test": raw_test_dataset}
        norm_split, norm_metrics = self._normalize(raw_split)
        
        return norm_split, norm_metrics, saved_config
