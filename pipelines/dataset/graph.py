from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors    import NearestNeighbors
from torch_geometric.data import Data


class Graph:
    EPSILON = 1e-8

    def __init__(self, config):
        self.config           = config
        self.bidirectional    = config.graph.bidirectional
        self.k_neighbors      = config.graph.k_neighbors
        self.knn_algorithm    = config.graph.knn_algorithm
        self.light_column     = config.data.light_column
        self.position_columns = config.data.position_columns

    def _effective_neighbors(self, number_of_nodes):
        return max(1, min(int(self.k_neighbors), number_of_nodes - 1))

    def _nearest_neighbors(self, positions):
        effective_neighbors = self._effective_neighbors(positions.shape[0])
        estimator           = NearestNeighbors(n_neighbors=effective_neighbors + 1, algorithm=self.knn_algorithm)
        estimator.fit(positions)
        return estimator.kneighbors(positions)

    def _node_position(self, positions):
        return positions

    def _node_light_intensity(self, light):
        return light.reshape(-1, 1)

    def _node_hit_flag(self, light):
        return (light > 0.0).astype(np.float32).reshape(-1, 1)

    def _node_light_fraction(self, light, total_light):
        return (light / total_light).astype(np.float32).reshape(-1, 1)

    def _node_distance_to_centroid(self, positions, light, total_light):
        centroid     = (light[:, None] * positions).sum(axis=0) / total_light
        displacement = positions - centroid[None, :]
        return np.linalg.norm(displacement, axis=1).astype(np.float32).reshape(-1, 1)

    def _node_local_light_density(self, light, neighbor_indices):
        return light[neighbor_indices[:, 1:]].sum(axis=1).astype(np.float32).reshape(-1, 1)

    def _build_nodes(self, event_dataframe):
        positions   = event_dataframe[list(self.position_columns)].values.astype(np.float32)
        light       = np.maximum(event_dataframe[self.light_column].fillna(0.0).values.astype(np.float32), 0.0)
        total_light = light.sum() + self.EPSILON

        _, neighbor_indices = self._nearest_neighbors(positions)

        node_features = np.column_stack([
            self._node_position(positions),
            self._node_light_intensity(light),
            self._node_hit_flag(light),
            self._node_light_fraction(light, total_light),
            self._node_distance_to_centroid(positions, light, total_light),
            self._node_local_light_density(light, neighbor_indices),
        ])

        return torch.tensor(node_features, dtype=torch.float32)

    def _edge_distance(self, distance):
        return distance.reshape(-1, 1)

    def _edge_displacement(self, source_positions, destination_positions, distance):
        normalization = np.maximum(distance, self.EPSILON)
        return (destination_positions - source_positions) / normalization[:, None]

    def _edge_inverse_square_distance(self, distance):
        return (1.0 / (distance ** 2 + self.EPSILON)).reshape(-1, 1)

    def _edge_light_gradient(self, source_light, destination_light):
        return (destination_light - source_light).reshape(-1, 1)

    def _edge_light_similarity(self, source_light, destination_light):
        return (2.0 * np.minimum(source_light, destination_light) / (source_light + destination_light + self.EPSILON)).reshape(-1, 1)

    def _assemble_edge_features(self, displacement, distance, inverse_square_distance, light_gradient, light_similarity):
        return np.column_stack([distance, displacement, inverse_square_distance, light_gradient, light_similarity]).astype(np.float32)

    def _build_edges(self, event_dataframe):
        positions = event_dataframe[list(self.position_columns)].values.astype(np.float32)
        light     = np.maximum(event_dataframe[self.light_column].fillna(0.0).values.astype(np.float32), 0.0)

        number_of_nodes                                  = int(positions.shape[0])
        neighbor_distances, neighbor_indices             = self._nearest_neighbors(positions)
        neighbor_indices_without_self                    = neighbor_indices[:, 1:]
        neighbor_distances_without_self                  = neighbor_distances[:, 1:]

        source_node_indices      = np.repeat(np.arange(number_of_nodes), neighbor_indices_without_self.shape[1])
        destination_node_indices = neighbor_indices_without_self.reshape(-1).astype(np.int64)
        distance                 = neighbor_distances_without_self.reshape(-1).astype(np.float64)

        source_positions      = positions[source_node_indices]
        destination_positions = positions[destination_node_indices]
        source_light          = light[source_node_indices].astype(np.float64)
        destination_light     = light[destination_node_indices].astype(np.float64)

        displacement            = self._edge_displacement(source_positions, destination_positions, distance)
        distance_column         = self._edge_distance(distance)
        inverse_square_distance = self._edge_inverse_square_distance(distance)
        light_gradient          = self._edge_light_gradient(source_light, destination_light)
        light_similarity        = self._edge_light_similarity(source_light, destination_light)

        forward_features = self._assemble_edge_features(displacement, distance_column, inverse_square_distance, light_gradient, light_similarity)

        if self.bidirectional:
            reverse_features = self._assemble_edge_features(-displacement, distance_column, inverse_square_distance, -light_gradient, light_similarity)

            interleaved_source      = np.empty(source_node_indices.shape[0] * 2, dtype=np.int64)
            interleaved_destination = np.empty(destination_node_indices.shape[0] * 2, dtype=np.int64)
            interleaved_source[0::2]      = source_node_indices
            interleaved_source[1::2]      = destination_node_indices
            interleaved_destination[0::2] = destination_node_indices
            interleaved_destination[1::2] = source_node_indices

            interleaved_features       = np.empty((forward_features.shape[0] * 2, forward_features.shape[1]), dtype=np.float32)
            interleaved_features[0::2] = forward_features
            interleaved_features[1::2] = reverse_features

            edge_index = torch.tensor(np.vstack([interleaved_source, interleaved_destination]), dtype=torch.long)
            edge_attr  = torch.tensor(interleaved_features, dtype=torch.float32)
            return edge_index, edge_attr

        edge_index = torch.tensor(np.vstack([source_node_indices.astype(np.int64), destination_node_indices]), dtype=torch.long)
        edge_attr  = torch.tensor(forward_features, dtype=torch.float32)
        return edge_index, edge_attr

    def build_graph(self, event_dataframe):
        node_features         = self._build_nodes(event_dataframe)
        edge_index, edge_attr = self._build_edges(event_dataframe)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
