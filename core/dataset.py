import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data  
import pickle
import math
from sklearn.model_selection import StratifiedShuffleSplit
import ray 

sys.path.insert(0, str(Path.cwd()))

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
        
        node_features = np.concatenate([positions, light_intensity.reshape(-1, 1)], axis=1)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        return x, event_dataframe

    def _create_edges(self, node_dataframe):
        position_columns = list(self.position_columns)
        positions_array  = node_dataframe[position_columns].values.astype(np.float32)

        number_of_nodes = int(positions_array.shape[0])
        effective_k     = max(1, min(int(self.k_neighbors), number_of_nodes - 1))

        knn = NearestNeighbors(n_neighbors=effective_k + 1, algorithm=self.knn_algorithm)
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
        
        @ray.remote
        def build_graph_remote(event_df):
            graph_builder = Graph(config=self.config)
            graph = graph_builder.build_graph(event_df)
            return graph
        
        self.logger.subsection(f"Building {len(dataframes)} graphs in parallel with Ray")
        futures = [build_graph_remote.remote(df) for idx, df in enumerate(dataframes)]

        graphs = []
        batch_size = min(50, max(10, len(futures) // 20))
        self.logger.subsection(f"Processing graph building with batch size {batch_size}")
        with tqdm(total=len(futures), desc="Building graphs") as pbar:
            while futures:
                ready, futures = ray.wait(futures, num_returns=min(batch_size, len(futures)))
                graphs.extend(ray.get(ready))
                pbar.update(len(ready))

        num_nodes = [graph.x.shape[0] for graph in graphs]
        num_edges = [graph.edge_index.shape[1] for graph in graphs]
        
        self.logger.subsection(f"Graphs: {len(graphs)}")
        self.logger.subsection(f"Nodes: {np.mean(num_nodes):.0f}±{np.std(num_nodes):.0f}")
        self.logger.subsection(f"Edges: {np.mean(num_edges):.0f}±{np.std(num_edges):.0f} \n")
        return graphs

    def _build_dataset(self, graphs, target_df):
        self.logger.subsection("Building dataset")
        dataset    = []
        coord_cols = list(self.coordinate_columns)
        
        for idx, graph in tqdm(enumerate(graphs), desc="Building dataset", total=len(graphs)):
            label   = torch.tensor(target_df.iloc[idx][coord_cols].values, dtype=torch.float32)
            graph.y = label.unsqueeze(0)
            dataset.append(graph)

        self.logger.subsection(f"Dataset built with {len(dataset)} graphs \n")
        return dataset

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
        
        for chunk_idx in tqdm(range(num_chunks), desc="Saving chunks"):
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
        dataframes, target_df = self._load_data()
        dataframes            = self._process_data(dataframes)
        graphs                = self._build_graphs(dataframes)
        dataset               = self._build_dataset(graphs, target_df)
        self.logger.subsection(f"Dataset built: {len(dataset)} graphs \n")
        self._save_artifacts(dataset, target_df)
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
        all_features = []
        for graph in source_graphs:
            if graph.x.numel() > 0:
                features = graph.x.numpy().astype(np.float32)
                all_features.append(features)

        all_features   = np.vstack(all_features)
        self.node_mean = all_features.mean(axis=0)
        self.node_std  = all_features.std(axis=0) + 1e-8
        
        self.logger.subsection(f"Before - Node mean: {self.node_mean}, std: {self.node_std} \n")

    def _compute_edge_statistics(self, source_graphs):
        self.logger.section("[Edge statistics]")
        all_edge_attrs = []
        for graph in source_graphs:
            if graph.edge_attr.numel() > 0:
                edge_attr_np = graph.edge_attr.numpy().astype(np.float32)
                all_edge_attrs.append(edge_attr_np)

        all_edge_attrs = np.vstack(all_edge_attrs)
        self.edge_mean = all_edge_attrs.mean(axis=0)
        self.edge_std  = all_edge_attrs.std(axis=0) + 1e-8
        self.logger.subsection(f"Before - Edge mean: {self.edge_mean}, std: {self.edge_std} \n")
        
    def _compute_target_statistics(self, source_graphs):
        self.logger.section("[Target statistics]")
        all_targets = []
        for graph in source_graphs:
            if hasattr(graph, 'y') and graph.y is not None and graph.y.numel() > 0:
                tgt = graph.y.detach().cpu().numpy().astype(np.float32).reshape(-1)
                all_targets.append(tgt)

        all_targets      = np.vstack(all_targets)
        self.target_mean = all_targets.mean(axis=0)
        self.target_std  = all_targets.std(axis=0) + 1e-8
        self.logger.subsection(f"Before - Target mean: {self.target_mean}, std: {self.target_std} \n")
        
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
        node_after   = np.vstack([g.x.numpy() for g in train_graphs]).mean(axis=0), np.vstack([g.x.numpy() for g in train_graphs]).std(axis=0)
        edge_after   = np.vstack([g.edge_attr.numpy() for g in train_graphs]).mean(axis=0), np.vstack([g.edge_attr.numpy() for g in train_graphs]).std(axis=0)
        target_after = np.vstack([g.y.numpy() for g in train_graphs]).mean(axis=0), np.vstack([g.y.numpy() for g in train_graphs]).std(axis=0)
        
        self.logger.subsection(f"After - Node mean   : {node_after[0]},   std : {node_after[1]}")
        self.logger.subsection(f"After - Edge mean   : {edge_after[0]},   std : {edge_after[1]}")
        self.logger.subsection(f"After - Target mean : {target_after[0]}, std : {target_after[1]} \n")
        
        return splits


class DatasetLoader:
    def __init__(self, preprocessed_dir, logger: Logger, config=None):
        self.config               = config
        self.logger               = logger
        self.coordinate_columns   = config.data.coordinate_columns
        self.preprocessed_dir     = Path(preprocessed_dir)

    def _balance_targets(self, target_df):   
        self.logger.section("[Balancing targets]")
        
        Y = target_df[list(self.coordinate_columns)].to_numpy(dtype=np.float32)
        N = len(Y)

        self.logger.subsection("Target distribution (before balancing)")
        self.logger.subsection(f"Full dataset (N={N}):")
        for dim_idx, col in enumerate(self.coordinate_columns):
            self.logger.subsection(f"  {col:10s}: mean={Y[:, dim_idx].mean():7.4f}, std={Y[:, dim_idx].std():7.4f}, "
                           f"min={Y[:, dim_idx].min():7.4f}, max={Y[:, dim_idx].max():7.4f}")

        n_bins = min(10, N // 10)  
        self.logger.subsection("Performing stratified split")
        self.logger.subsection(f"Binning coordinates into {n_bins} bins per dimension")
        
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
        
        self.logger.subsection(f"Created {len(np.unique(strat_labels))} unique stratification groups")
        split_config = self.config.split
       
        sss_test                = StratifiedShuffleSplit(n_splits=1, test_size=split_config.test_size, random_state=split_config.random_state)
        train_val_idx, test_idx = next(sss_test.split(np.arange(N), strat_labels))
        
        val_ratio_adjusted = split_config.val_ratio / (1 - split_config.test_size)
        sss_val            = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=split_config.random_state)
        train_idx, val_idx = next(sss_val.split(train_val_idx, strat_labels[train_val_idx]))
        
        train_idx = train_val_idx[train_idx]
        val_idx   = train_val_idx[val_idx]

        self.logger.subsection(f"Split sizes: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
        
        self.logger.subsection("Target distribution (after balancing)")
        for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            self.logger.subsection(f"{split_name} split (N={len(indices)}):")
            for dim_idx, col in enumerate(self.coordinate_columns):
                split_data = Y[indices, dim_idx]
                self.logger.subsection(f"  {col:10s}: mean={split_data.mean():7.4f}, std={split_data.std():7.4f}, "
                               f"min={split_data.min():7.4f}, max={split_data.max():7.4f}")
        
        self.logger.subsection("Stratification group distribution")
        for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            unique_groups = np.unique(strat_labels[indices])
            self.logger.subsection(f"{split_name}: {len(unique_groups)} unique groups represented")
        
        self.logger.subsection("")
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
        for chunk_idx in tqdm(range(num_chunks), desc="Loading chunks"):
            chunk_path = self.preprocessed_dir / "chunks" / f"dataset_chunk_{chunk_idx}.pt"
            chunk = torch.load(chunk_path, weights_only=False)
            graphs.extend(chunk)
     
        with open(tp, "rb") as f: target_df = pickle.load(f)
        with open(cp, "rb") as f: config    = pickle.load(f)
        
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
