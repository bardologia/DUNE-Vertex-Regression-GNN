from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors      import NearestNeighbors
from torch_geometric.data   import Data


class Graph:
    def __init__(self, config):
        self.config           = config
        self.bidirectional    = config.graph.bidirectional
        self.k_neighbors      = config.graph.k_neighbors
        self.knn_algorithm    = config.graph.knn_algorithm
        self.light_column     = config.data.light_column
        self.position_columns = config.data.position_columns

    def _create_nodes(self, event_dataframe):
        position_columns = list(self.position_columns)

        positions       = event_dataframe[position_columns].values.astype(np.float32)
        light_intensity = event_dataframe[self.light_column].fillna(0.0).values.astype(np.float32)
        light_intensity = np.maximum(light_intensity, 0.0)

        hit_flag = (light_intensity > 0.0).astype(np.float32)

        total_light    = light_intensity.sum() + 1e-8
        light_fraction = (light_intensity / total_light).astype(np.float32)

        centroid                  = (light_intensity[:, None] * positions).sum(axis=0) / total_light
        displacement_to_centroid  = positions - centroid[None, :]
        distance_to_centroid      = np.linalg.norm(displacement_to_centroid, axis=1).astype(np.float32)

        effective_k     = max(1, min(int(self.k_neighbors), positions.shape[0] - 1))
        nearest_neighbors = NearestNeighbors(n_neighbors=effective_k + 1, algorithm=self.knn_algorithm)
        nearest_neighbors.fit(positions)
        _, neighbor_indices = nearest_neighbors.kneighbors(positions)
        local_light_density = light_intensity[neighbor_indices[:, 1:]].sum(axis=1).astype(np.float32)

        node_features = np.column_stack([
            positions,
            light_intensity.reshape(-1, 1),
            hit_flag.reshape(-1, 1),
            light_fraction.reshape(-1, 1),
            distance_to_centroid.reshape(-1, 1),
            local_light_density.reshape(-1, 1),
        ])

        return torch.tensor(node_features, dtype=torch.float32)

    def _create_edges(self, event_dataframe):
        position_columns = list(self.position_columns)
        positions_array  = event_dataframe[position_columns].values.astype(np.float32)
        light_array      = event_dataframe[self.light_column].fillna(0.0).values.astype(np.float32)
        light_array      = np.maximum(light_array, 0.0)

        number_of_nodes = int(positions_array.shape[0])
        effective_k     = max(1, min(int(self.k_neighbors), number_of_nodes - 1))

        nearest_neighbors = NearestNeighbors(n_neighbors=effective_k + 1, algorithm=self.knn_algorithm)
        nearest_neighbors.fit(positions_array)
        neighbor_distances_array, neighbor_indices_array = nearest_neighbors.kneighbors(positions_array)

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

            interleaved_source      = np.empty(source_node_indices.shape[0] * 2, dtype=np.int64)
            interleaved_destination = np.empty(destination_node_indices.shape[0] * 2, dtype=np.int64)
            interleaved_source[0::2]      = source_node_indices
            interleaved_source[1::2]      = destination_node_indices
            interleaved_destination[0::2] = destination_node_indices
            interleaved_destination[1::2] = source_node_indices

            interleaved_features        = np.empty((forward_features.shape[0] * 2, forward_features.shape[1]), dtype=np.float32)
            interleaved_features[0::2]  = forward_features
            interleaved_features[1::2]  = reverse_features

            edge_index = torch.tensor(np.vstack([interleaved_source, interleaved_destination]), dtype=torch.long)
            edge_attr  = torch.tensor(interleaved_features, dtype=torch.float32)
            return edge_index, edge_attr

        edge_index = torch.tensor(np.vstack([source_node_indices.astype(np.int64), destination_node_indices]), dtype=torch.long)
        edge_attr  = torch.tensor(forward_features, dtype=torch.float32)
        return edge_index, edge_attr

    def build_graph(self, event_dataframe):
        node_features         = self._create_nodes(event_dataframe)
        edge_index, edge_attr = self._create_edges(event_dataframe)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
