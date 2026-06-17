from __future__ import annotations

import math
import pickle
from pathlib import Path
from types   import SimpleNamespace

import numpy as np
import ray
import torch

from pipelines.dataset.graph        import Graph
from pipelines.dataset.loading      import DataLoader
from pipelines.dataset.preprocessing import DataProcessor
from pipelines.dataset.ray_executor import RayExecutor


class DatasetBuilder:
    CHUNK_SIZE = 100

    def __init__(self, output_dir, logger, config):
        self.logger     = logger
        self.config     = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor   = RayExecutor(logger, config.ray)

        self.zscore_threshold     = config.preprocessing.zscore_threshold
        self.radius               = config.preprocessing.radius
        self.scale_factor         = config.preprocessing.scale_factor
        self.detection_efficiency = config.preprocessing.detection_efficiency
        self.random_state         = config.split.random_state
        self.coordinate_columns   = config.data.coordinate_columns

    def _load_data(self):
        loader = DataLoader(logger=self.logger, config=self.config)
        return loader.load()

    def _process_data(self, dataframes):
        processor = DataProcessor(logger=self.logger, config=self.config)

        dataframes, outliers = processor.detect_spatial_outliers(dataframes, self.zscore_threshold, self.radius)
        dataframes           = processor.scale_light_intensity(dataframes, self.scale_factor)
        dataframes           = processor.apply_detection_efficiency(dataframes, self.detection_efficiency, seed=self.random_state)
        dataframes           = processor.apply_log_transform(dataframes)

        return dataframes

    def _build_graphs(self, dataframes):
        self.logger.section("[Building Graphs]")

        if len(dataframes) == 0:
            return []

        self.executor.ensure_initialized()

        graph_config = SimpleNamespace(
            graph = SimpleNamespace(
                bidirectional = self.config.graph.bidirectional,
                k_neighbors   = self.config.graph.k_neighbors,
                knn_algorithm = self.config.graph.knn_algorithm,
            ),
            data = SimpleNamespace(
                light_column     = self.config.data.light_column,
                position_columns = self.config.data.position_columns,
            ),
        )

        def build_graph_batch(event_batch, builder_config):
            builder = Graph(config=builder_config)
            return [builder.build_graph(event_dataframe) for event_dataframe in event_batch]

        remote_build      = ray.remote(build_graph_batch)
        dataframe_batches = self.executor.make_batches(dataframes)
        batch_results     = self.executor.run_batches(
            remote_build,
            [(batch, graph_config) for batch in dataframe_batches],
            "Building graphs (Ray batches)",
        )

        graphs = []
        for graph_batch in batch_results:
            graphs.extend(graph_batch)

        node_counts = [graph.x.shape[0] for graph in graphs]
        edge_counts = [graph.edge_index.shape[1] for graph in graphs]
        self.logger.subsection(f"Graphs: {len(graphs)} | Nodes: {np.mean(node_counts):.0f}+-{np.std(node_counts):.0f} | Edges: {np.mean(edge_counts):.0f}+-{np.std(edge_counts):.0f}")
        return graphs

    def _attach_targets(self, graphs, target_dataframe):
        if len(graphs) == 0:
            return []

        target_values = target_dataframe[list(self.coordinate_columns)].values.astype(np.float32)
        for index, graph in enumerate(graphs):
            graph.y = torch.tensor(target_values[index], dtype=torch.float32).unsqueeze(0)

        return graphs

    def _save_artifacts(self, dataset, target_dataframe):
        self.logger.section("[Saving Artifacts]")
        chunks_directory = self.output_dir / "chunks"
        chunks_directory.mkdir(parents=True, exist_ok=True)

        number_of_chunks = math.ceil(len(dataset) / self.CHUNK_SIZE)

        with self.logger.track() as progress:
            task_id = progress.add_task("Saving chunks", total=number_of_chunks)
            for chunk_index in range(number_of_chunks):
                start = chunk_index * self.CHUNK_SIZE
                end   = min((chunk_index + 1) * self.CHUNK_SIZE, len(dataset))
                torch.save(dataset[start:end], chunks_directory / f"dataset_chunk_{chunk_index}.pt")
                progress.advance(task_id)

        metadata = {"num_chunks": number_of_chunks, "chunk_size": self.CHUNK_SIZE, "total_graphs": len(dataset)}
        with open(self.output_dir / "dataset_metadata.pkl", "wb") as handle:
            pickle.dump(metadata, handle)

        with open(self.output_dir / "target.pkl", "wb") as handle:
            pickle.dump(target_dataframe, handle)

        config_payload = {
            "data_config"          : self.config.data,
            "graph_config"         : self.config.graph,
            "preprocessing_config" : self.config.preprocessing,
            "split_config"         : self.config.split,
            "dataset_len"          : len(dataset),
            "num_targets"          : len(target_dataframe),
        }
        with open(self.output_dir / "config.pkl", "wb") as handle:
            pickle.dump(config_payload, handle)

        self.logger.subsection(f"Artifacts saved to {self.output_dir} ({number_of_chunks} chunks)")

    def run(self):
        self.logger.section("[Dataset Builder]")
        self.logger.subsection(f"Output directory: {self.output_dir}")
        self.executor.ensure_initialized()

        try:
            dataframes, target_dataframe = self._load_data()
            dataframes                   = self._process_data(dataframes)
            graphs                       = self._build_graphs(dataframes)
            del dataframes
            dataset                      = self._attach_targets(graphs, target_dataframe)
            self._save_artifacts(dataset, target_dataframe)
        finally:
            self.executor.shutdown()

        self.logger.subsection("Dataset build complete")
