from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors    import NearestNeighbors
from torch_geometric.data import Data


class GraphAssembler:
    EPSILON = 1e-8

    def __init__(self, config):
        self.bidirectional      = config.graph.bidirectional
        self.k_neighbors        = config.graph.k_neighbors
        self.knn_algorithm      = config.graph.knn_algorithm

        self.direction_features = config.graph.direction_features
        self.inertia_features   = config.graph.inertia_features
        self.rank_features      = config.graph.rank_features
        self.edge_rbf_count     = int(config.graph.edge_rbf_count)
        self.edge_rbf_max       = float(config.graph.edge_rbf_max)

    def _effective_neighbors(self, number_of_nodes):
        return max(1, min(int(self.k_neighbors), number_of_nodes - 1))

    def _nearest_neighbors(self, positions):
        effective_neighbors = self._effective_neighbors(positions.shape[0])
        estimator           = NearestNeighbors(n_neighbors=effective_neighbors + 1, algorithm=self.knn_algorithm)
        estimator.fit(positions)
        return estimator.kneighbors(positions)


class NodeFeatures(GraphAssembler):
    def _position(self, positions):
        return positions

    def _light_intensity(self, light):
        return light.reshape(-1, 1)

    def _light_fraction(self, light, total_light):
        return (light / total_light).astype(np.float32).reshape(-1, 1)

    def _distance_to_centroid(self, positions, light, total_light):
        centroid     = (light[:, None] * positions).sum(axis=0) / total_light
        displacement = positions - centroid[None, :]
        return np.linalg.norm(displacement, axis=1).astype(np.float32).reshape(-1, 1)

    def _local_light_density(self, light, neighbor_indices):
        return light[neighbor_indices[:, 1:]].sum(axis=1).astype(np.float32).reshape(-1, 1)

    def _direction_to_centroid(self, positions, light, total_light):
        centroid     = (light[:, None] * positions).sum(axis=0) / total_light
        displacement = centroid[None, :] - positions
        norm         = np.maximum(np.linalg.norm(displacement, axis=1, keepdims=True), self.EPSILON)
        return (displacement / norm).astype(np.float32)

    def _direction_to_brightest(self, positions, light):
        brightest    = positions[int(np.argmax(light))]
        displacement = brightest[None, :] - positions
        norm         = np.maximum(np.linalg.norm(displacement, axis=1, keepdims=True), self.EPSILON)
        return (displacement / norm).astype(np.float32)

    def _inertia_invariants(self, positions, light, total_light, number_of_nodes):
        centroid    = (light[:, None] * positions).sum(axis=0) / total_light
        centered    = positions - centroid[None, :]
        covariance  = (centered * light[:, None]).T @ centered / total_light
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)
        normalized  = eigenvalues / (eigenvalues.sum() + self.EPSILON)
        return np.tile(normalized.astype(np.float32), (number_of_nodes, 1))

    def _intensity_rank(self, light, number_of_nodes):
        ordinal = np.argsort(np.argsort(light))
        return (ordinal.astype(np.float32) / max(number_of_nodes - 1, 1)).reshape(-1, 1)

    def build(self, positions, light, neighbor_indices):
        number_of_nodes = positions.shape[0]
        total_light     = light.sum() + self.EPSILON

        feature_columns = [
            self._position(positions),
            self._light_intensity(light),
            self._light_fraction(light, total_light),
            self._distance_to_centroid(positions, light, total_light),
            self._local_light_density(light, neighbor_indices),
        ]

        if self.direction_features:
            feature_columns.append(self._direction_to_centroid(positions, light, total_light))
            feature_columns.append(self._direction_to_brightest(positions, light))

        if self.inertia_features:
            feature_columns.append(self._inertia_invariants(positions, light, total_light, number_of_nodes))

        if self.rank_features:
            feature_columns.append(self._intensity_rank(light, number_of_nodes))

        node_features = np.column_stack(feature_columns)
        return torch.tensor(node_features, dtype=torch.float32)


class EdgeFeatures(GraphAssembler):
    def _distance(self, distance):
        return distance.reshape(-1, 1)

    def _displacement(self, source_positions, destination_positions, distance):
        normalization = np.maximum(distance, self.EPSILON)
        return (destination_positions - source_positions) / normalization[:, None]

    def _inverse_square_distance(self, distance):
        return (1.0 / (distance ** 2 + self.EPSILON)).reshape(-1, 1)

    def _light_gradient(self, source_light, destination_light):
        return (destination_light - source_light).reshape(-1, 1)

    def _light_similarity(self, source_light, destination_light):
        return (2.0 * np.minimum(source_light, destination_light) / (source_light + destination_light + self.EPSILON)).reshape(-1, 1)

    def _distance_rbf(self, distance):
        centers = np.linspace(0.0, self.edge_rbf_max, self.edge_rbf_count)
        spacing = self.edge_rbf_max / max(self.edge_rbf_count - 1, 1)
        gamma   = 1.0 / (spacing ** 2 + self.EPSILON)
        return np.exp(-gamma * (distance[:, None] - centers[None, :]) ** 2)

    def _assemble(self, displacement, distance, inverse_square_distance, light_gradient, light_similarity, distance_rbf):
        columns = [distance, displacement, inverse_square_distance, light_gradient, light_similarity]
        if distance_rbf is not None:
            columns.append(distance_rbf)
        return np.column_stack(columns).astype(np.float32)

    def _interleave_bidirectional(self, source_node_indices, destination_node_indices, forward_features, reverse_features):
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

    def build(self, positions, light, neighbor_distances, neighbor_indices):
        number_of_nodes                 = int(positions.shape[0])
        neighbor_indices_without_self   = neighbor_indices[:, 1:]
        neighbor_distances_without_self = neighbor_distances[:, 1:]

        source_node_indices      = np.repeat(np.arange(number_of_nodes), neighbor_indices_without_self.shape[1])
        destination_node_indices = neighbor_indices_without_self.reshape(-1).astype(np.int64)
        distance                 = neighbor_distances_without_self.reshape(-1).astype(np.float64)

        source_positions      = positions[source_node_indices]
        destination_positions = positions[destination_node_indices]
        source_light          = light[source_node_indices].astype(np.float64)
        destination_light     = light[destination_node_indices].astype(np.float64)

        displacement            = self._displacement(source_positions, destination_positions, distance)
        distance_column         = self._distance(distance)
        inverse_square_distance = self._inverse_square_distance(distance)
        light_gradient          = self._light_gradient(source_light, destination_light)
        light_similarity        = self._light_similarity(source_light, destination_light)
        distance_rbf            = self._distance_rbf(distance) if self.edge_rbf_count > 0 else None

        forward_features = self._assemble(displacement, distance_column, inverse_square_distance, light_gradient, light_similarity, distance_rbf)

        if self.bidirectional:
            reverse_features = self._assemble(-displacement, distance_column, inverse_square_distance, -light_gradient, light_similarity, distance_rbf)
            return self._interleave_bidirectional(source_node_indices, destination_node_indices, forward_features, reverse_features)

        edge_index = torch.tensor(np.vstack([source_node_indices.astype(np.int64), destination_node_indices]), dtype=torch.long)
        edge_attr  = torch.tensor(forward_features, dtype=torch.float32)
        return edge_index, edge_attr


class ActiveNodeSelector:
    MINIMUM_NODES = 2

    def __init__(self, config):
        self.active_only      = config.graph.active_only
        self.max_active_nodes = int(config.graph.max_active_nodes)

    def _active_indices(self, light):
        if self.active_only:
            return np.nonzero(light > 0.0)[0]
        return np.arange(light.shape[0])

    def _apply_budget(self, indices, light):
        if self.max_active_nodes > 0 and indices.shape[0] > self.max_active_nodes:
            brightest_first = np.argsort(light[indices])[::-1]
            indices         = indices[brightest_first[: self.max_active_nodes]]
        return indices

    def _enforce_floor(self, indices, light):
        if indices.shape[0] >= self.MINIMUM_NODES:
            return indices
        return np.argsort(light)[::-1][: self.MINIMUM_NODES]

    def select(self, positions, light):
        indices = self._active_indices(light)
        indices = self._apply_budget(indices, light)
        indices = self._enforce_floor(indices, light)

        indices = np.sort(indices)
        return positions[indices], light[indices]


class Graph:
    def __init__(self, config):
        self.config        = config
        self.node_selector = ActiveNodeSelector(config)
        self.node_features = NodeFeatures(config)
        self.edge_features = EdgeFeatures(config)

    def build_from_arrays(self, positions, light):
        positions = np.asarray(positions, dtype=np.float32)
        light     = np.maximum(np.asarray(light, dtype=np.float32), 0.0)

        positions, light = self.node_selector.select(positions, light)

        neighbor_distances, neighbor_indices = self.node_features._nearest_neighbors(positions)

        node_features         = self.node_features.build(positions, light, neighbor_indices)
        edge_index, edge_attr = self.edge_features.build(positions, light, neighbor_distances, neighbor_indices)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


class FeatureSchema:
    PROBE_NODES = 8

    def __init__(self, config):
        self.builder = Graph(config)

    def dimensions(self):
        generator = np.random.default_rng(0)
        positions = generator.normal(size=(self.PROBE_NODES, 3)).astype(np.float32)
        light     = (generator.random(self.PROBE_NODES) + 0.1).astype(np.float32)
        data      = self.builder.build_from_arrays(positions, light)
        return int(data.x.shape[1]), int(data.edge_attr.shape[1])
