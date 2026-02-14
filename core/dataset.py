import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from core.data import DataLoader, DataProcessor
from core.config import config
from core.logger import Logger
from tqdm import tqdm
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data  # type: ignore
import pickle
import math
from sklearn.model_selection import StratifiedShuffleSplit
import ray # type: ignore


class Graph:
    def __init__(self):
        self.bidirectional = config.graph.bidirectional

    def _create_nodes(self, event_dataframe):
        light_col     = config.data.light_column
        position_cols = list(config.data.position_columns)
        
        positions       = event_dataframe[position_cols].values.astype(np.float32)
        light_intensity = event_dataframe[light_col].fillna(0.0).values.astype(np.float32)
        light_intensity = np.maximum(light_intensity, 0.0)
        
        node_features = np.concatenate([positions, light_intensity.reshape(-1, 1)], axis=1)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        return x, event_dataframe

    def _create_edges(self, node_dataframe):
        position_columns = list(config.data.position_columns)
        positions_array  = node_dataframe[position_columns].values.astype(np.float32)

        number_of_nodes = int(positions_array.shape[0])
        effective_k     = max(1, min(int(config.graph.k_neighbors), number_of_nodes - 1))

        knn = NearestNeighbors(n_neighbors=effective_k + 1, algorithm=config.graph.knn_algorithm)
        knn.fit(positions_array)
        neighbor_distances_array, neighbor_indices_array = knn.kneighbors(positions_array)

        source_node_indices      = []
        destination_node_indices = []
        euclidean_distances_list = []

        for source_index in range(number_of_nodes):
            neighbor_indices_row   = neighbor_indices_array[source_index][1:]
            neighbor_distances_row = neighbor_distances_array[source_index][1:]

            for local_index, neighbor_index in enumerate(neighbor_indices_row):
                neighbor_index = int(neighbor_index)
                distance_value = float(neighbor_distances_row[local_index])

                source_node_indices.append(source_index)
                destination_node_indices.append(neighbor_index)
                euclidean_distances_list.append(distance_value)

                if self.bidirectional:
                    source_node_indices.append(neighbor_index)
                    destination_node_indices.append(source_index)
                    euclidean_distances_list.append(distance_value)

        edge_index = torch.tensor([source_node_indices, destination_node_indices], dtype=torch.long)
        edge_attr = torch.tensor(euclidean_distances_list, dtype=torch.float32).unsqueeze(1)
        
        return edge_index, edge_attr

    def build_graph(self, event_dataframe):
        x, node_df = self._create_nodes(event_dataframe)
        edge_index, edge_attr = self._create_edges(node_df)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class DatasetBuilder:
    def __init__(self, output_dir, logger: Logger):
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_data(self):
        self.logger.subsection("Loading data")
        loader = DataLoader(logger=self.logger)
        dataframes, target_df = loader.load()
        self.logger.info(f"Loaded {len(dataframes)} events")
        return dataframes, target_df
    
    def _process_data(self, dataframes):
        self.logger.subsection("Processing data")
        preprocessor = DataProcessor(logger=self.logger)

        dataframes, outliers = preprocessor.detect_spatial_outliers(
            dataframes, 
            config.preprocessing.zscore_threshold,
            config.preprocessing.radius
        )

        total_outliers = sum(len(df) for df in outliers)
        self.logger.info(f"Total outliers removed: {total_outliers}")

        dataframes = preprocessor.scale_light_intensity(dataframes, config.preprocessing.scale_factor)
        dataframes = preprocessor.apply_detection_efficiency(
            dataframes, 
            config.preprocessing.detection_efficiency, 
            seed=config.split.random_state
        )

        dataframes = preprocessor.apply_log_transform(dataframes)

        return dataframes

    def _build_graphs(self, dataframes):
        self.logger.subsection("Building graphs")
        
        @ray.remote
        def build_graph_remote(event_df):
            graph_builder = Graph()
            graph = graph_builder.build_graph(event_df)
            return graph
        
        self.logger.info(f"Building {len(dataframes)} graphs in parallel with Ray")
        futures = [build_graph_remote.remote(df) for idx, df in enumerate(dataframes)]

        graphs = []
        batch_size = min(50, max(10, len(futures) // 20))
        self.logger.info(f"Processing graph building with batch size {batch_size}")
        with tqdm(total=len(futures), desc="Building graphs") as pbar:
            while futures:
                ready, futures = ray.wait(futures, num_returns=min(batch_size, len(futures)))
                graphs.extend(ray.get(ready))
                pbar.update(len(ready))

        num_nodes = [graph.x.shape[0] for graph in graphs]
        num_edges = [graph.edge_index.shape[1] for graph in graphs]
        self.logger.info(f"Graphs: {len(graphs)} | Nodes: {np.mean(num_nodes):.0f}±{np.std(num_nodes):.0f} | Edges: {np.mean(num_edges):.0f}±{np.std(num_edges):.0f}")
        return graphs

    def _build_dataset(self, graphs, target_df):
        self.logger.subsection("Building dataset")
        dataset    = []
        coord_cols = list(config.data.coordinate_columns)
        
        for idx, graph in tqdm(enumerate(graphs), desc="Building dataset", total=len(graphs)):
            label   = torch.tensor(target_df.iloc[idx][coord_cols].values, dtype=torch.float32)
            graph.y = label.unsqueeze(0)
            dataset.append(graph)

        self.logger.info(f"Dataset built with {len(dataset)} graphs")
        return dataset

    def _save_artifacts(self, dataset, target_df):
        self.logger.subsection("Saving artifacts")
        target_path = self.output_dir / "target.pkl"
        config_path = self.output_dir / "config.pkl"
        
        chunk_size = 100  
        self.logger.info(f"Saving dataset in chunks of size {chunk_size}")
        num_chunks = math.ceil(len(dataset) / chunk_size)
        self.logger.info(f"Saving dataset in {num_chunks} chunks")
        
        for chunk_idx in tqdm(range(num_chunks), desc="Saving chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(dataset))
            chunk = dataset[start_idx:end_idx]
            
            chunk_path = self.output_dir / f"dataset_chunk_{chunk_idx}.pt"
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
            'data_config'          : config.data,
            'graph_config'         : config.graph,
            'preprocessing_config' : config.preprocessing,
            'dataset_len'          : len(dataset),
            'num_targets'          : len(target_df),
        }
        
        with open(config_path, 'wb') as f:
            pickle.dump(config_dict, f)
        
        self.logger.info(f"Artifacts saved: {self.output_dir.name} ({num_chunks} chunks)")

    def run(self):
        self.logger.section("[Dataset Builder]")
        self.logger.info(f"Output directory: {self.output_dir}")
        dataframes, target_df = self._load_data()
        dataframes            = self._process_data(dataframes)
        graphs                = self._build_graphs(dataframes)
        dataset               = self._build_dataset(graphs, target_df)
        self.logger.info(f"Dataset built: {len(dataset)} graphs")
        self._save_artifacts(dataset, target_df)
        self.logger.info("Dataset building process completed successfully")


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
        self.logger.info(f"Computed statistics from '{self.source_split}' split")

    def _compute_node_statistics(self, source_graphs):
        all_features = []
        for graph in source_graphs:
            if graph.x.numel() > 0:
                features = graph.x.numpy().astype(np.float32)
                all_features.append(features)

        all_features   = np.vstack(all_features)
        self.node_mean = all_features.mean(axis=0)
        self.node_std  = all_features.std(axis=0) + 1e-8
        self.logger.info(f"Node statistics: mean shape={self.node_mean.shape}, std shape={self.node_std.shape}")

    def _compute_edge_statistics(self, source_graphs):
        all_edge_attrs = []
        for graph in source_graphs:
            if graph.edge_attr.numel() > 0:
                edge_attr_np = graph.edge_attr.numpy().astype(np.float32)
                all_edge_attrs.append(edge_attr_np)

        all_edge_attrs = np.vstack(all_edge_attrs)
        self.edge_mean = all_edge_attrs.mean(axis=0)
        self.edge_std  = all_edge_attrs.std(axis=0) + 1e-8
        self.logger.info(f"Edge statistics: mean shape={self.edge_mean.shape}, std shape={self.edge_std.shape}")
        
    def _compute_target_statistics(self, source_graphs):
        all_targets = []
        for graph in source_graphs:
            if hasattr(graph, 'y') and graph.y is not None and graph.y.numel() > 0:
                tgt = graph.y.detach().cpu().numpy().astype(np.float32).reshape(-1)
                all_targets.append(tgt)

        all_targets      = np.vstack(all_targets)
        self.target_mean = all_targets.mean(axis=0)
        self.target_std  = all_targets.std(axis=0) + 1e-8
        self.logger.info(f"Target statistics: mean shape={self.target_mean.shape}, std shape={self.target_std.shape}")
        
    def normalize(self, splits):
        for split_name, graphs in splits.items():
            for graph in graphs:
                graph.x = (graph.x - torch.tensor(self.node_mean, dtype=torch.float32)) / torch.tensor(self.node_std, dtype=torch.float32)
                graph.edge_attr = (graph.edge_attr - torch.tensor(self.edge_mean, dtype=torch.float32)) / torch.tensor(self.edge_std, dtype=torch.float32)

                mean_t = torch.tensor(self.target_mean, dtype=torch.float32)
                std_t = torch.tensor(self.target_std, dtype=torch.float32)
                graph.y = (graph.y - mean_t) / std_t
        
        return splits


class DatasetLoader:
    def __init__(self, preprocessed_dir, logger: Logger):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.config           = None
        self.logger           = logger

    def _balance_targets(self, target_df):
        data_config = self.config.data
        coord_cols = list(data_config.coordinate_columns)
        
        Y = target_df[coord_cols].to_numpy(dtype=np.float32)
        N = len(Y)

        self.logger.subsection("Performing stratified split")
        n_bins = min(10, N // 10)  
        self.logger.info(f"Binning coordinates into {n_bins} bins per dimension")
        
        binned_coords = []
        for dim in range(Y.shape[1]):
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins        = np.percentile(Y[:, dim], percentiles)
            bins        = np.unique(bins)  
            binned      = np.digitize(Y[:, dim], bins[:-1]) - 1
            binned      = np.clip(binned, 0, len(bins) - 2)
            binned_coords.append(binned)
        
        binned_coords = np.array(binned_coords).T
        n_bins_actual = max(len(np.unique(bc)) for bc in binned_coords.T)
        strat_labels  = (binned_coords[:, 0] * n_bins_actual**2 + binned_coords[:, 1] * n_bins_actual + binned_coords[:, 2])
        
        self.logger.info(f"Created {len(np.unique(strat_labels))} unique stratification groups")
        
        sss_test                = StratifiedShuffleSplit(n_splits=1, test_size=config.split.test_size, random_state=config.split.random_state)
        train_val_idx, test_idx = next(sss_test.split(np.arange(N), strat_labels))
        
        val_ratio_adjusted = config.split.val_ratio / (1 - config.split.test_size)
        sss_val            = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=config.split.random_state)
        train_idx, val_idx = next(sss_val.split(train_val_idx, strat_labels[train_val_idx]))
        
        train_idx = train_val_idx[train_idx]
        val_idx   = train_val_idx[val_idx]

        self.logger.info(f"Split sizes: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
        return train_idx, val_idx, test_idx, Y, N

    def _attach_targets(self, graph_splits, idx_splits, Y: np.ndarray):
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

        self.logger.info(f"Targets attached to splits: train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)}")
        return train_dataset, val_dataset, test_dataset

    def _normalize(self, splits, data_config):
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

        self.logger.info("Dataset normalization completed")
        return splits, metrics

    def _load_artifacts(self):
        tp = self.preprocessed_dir / "target.pkl"
        cp = self.preprocessed_dir / "config.pkl"
        
        metadata_path = self.preprocessed_dir / "dataset_metadata.pkl"
  
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        num_chunks = metadata['num_chunks']
        graphs = []
        
        self.logger.info(f"Loading dataset from {num_chunks} chunks")
        for chunk_idx in tqdm(range(num_chunks), desc="Loading chunks"):
            chunk_path = self.preprocessed_dir / f"dataset_chunk_{chunk_idx}.pt"
            chunk = torch.load(chunk_path, weights_only=False)
            graphs.extend(chunk)
     
        with open(tp, "rb") as f: target_df = pickle.load(f)
        with open(cp, "rb") as f: config    = pickle.load(f)
        
        self.config = config
        self.logger.info(f"Loaded artifacts from: {self.preprocessed_dir}")
        return graphs, target_df, config
    
