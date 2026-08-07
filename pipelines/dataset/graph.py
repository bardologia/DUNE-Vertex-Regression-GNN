from __future__ import annotations

import numpy as np
import torch
from scipy.spatial        import cKDTree
from scipy.stats          import rankdata
from torch_geometric.data import Data


class GraphAssembler:
    EPSILON = 1e-8

    def __init__(self, config):
        self.bidirectional      = config.graph.bidirectional
        self.k_neighbors        = config.graph.k_neighbors

        self.direction_features = config.graph.direction_features
        self.inertia_features   = config.graph.inertia_features
        self.rank_features      = config.graph.rank_features
        self.edge_rbf_count     = int(config.graph.edge_rbf_count)
        self.edge_rbf_max       = float(config.graph.edge_rbf_max)

    def _effective_neighbors(self, number_of_nodes):
        return max(1, min(int(self.k_neighbors), number_of_nodes - 1))

    def _nearest_neighbors(self, positions):
        effective_neighbors = self._effective_neighbors(positions.shape[0])
        tree                = cKDTree(positions)
        distances, indices  = tree.query(positions, k=effective_neighbors + 1)
        return distances.astype(np.float64), indices.astype(np.int64)

    def _light_centroid(self, positions, light, total_light):
        return (light[:, None] * positions).sum(axis=0) / total_light

    def _light_covariance_eigenvalues(self, positions, light, total_light):
        centroid   = self._light_centroid(positions, light, total_light)
        centered   = positions - centroid[None, :]
        covariance = (centered * light[:, None]).T @ centered / total_light
        return np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)

    def _sharp_light_centroid(self, positions, light):
        sharp_weights = light.astype(np.float64) ** 2
        return (sharp_weights[:, None] * positions).sum(axis=0) / (sharp_weights.sum() + self.EPSILON)

    def _axis_moments(self, positions, light, total_light, centroid):
        centered = positions - centroid[None, :]
        weights  = light.astype(np.float64)[:, None]

        variance = (weights * centered ** 2).sum(axis=0) / total_light
        spread   = np.sqrt(np.clip(variance, 0.0, None))
        third    = (weights * centered ** 3).sum(axis=0) / total_light
        skew     = third / (spread ** 3 + self.EPSILON)

        return spread, skew

    def _weighted_quantiles(self, positions, light, quantiles):
        blocks = []
        for axis_index in range(positions.shape[1]):
            order             = np.argsort(positions[:, axis_index], kind="stable")
            cumulative_weight = np.cumsum(light[order].astype(np.float64))
            targets           = np.asarray(quantiles) * cumulative_weight[-1]
            indices           = np.minimum(np.searchsorted(cumulative_weight, targets), positions.shape[0] - 1)
            blocks.append(positions[order][indices, axis_index])
        return np.stack(blocks, axis=1).reshape(-1)


class NodeFeatures(GraphAssembler):
    def _position(self, positions):
        return positions

    def _light_intensity(self, light):
        return light.reshape(-1, 1)

    def _light_fraction(self, light, total_light):
        return (light / total_light).astype(np.float32).reshape(-1, 1)

    def _distance_to_centroid(self, positions, light, total_light):
        centroid     = self._light_centroid(positions, light, total_light)
        displacement = positions - centroid[None, :]
        return np.linalg.norm(displacement, axis=1).astype(np.float32).reshape(-1, 1)

    def _local_light_density(self, light, neighbor_indices):
        return light[neighbor_indices[:, 1:]].sum(axis=1).astype(np.float32).reshape(-1, 1)

    def _direction_to_centroid(self, positions, light, total_light):
        centroid     = self._light_centroid(positions, light, total_light)
        displacement = centroid[None, :] - positions
        norm         = np.maximum(np.linalg.norm(displacement, axis=1, keepdims=True), self.EPSILON)
        return (displacement / norm).astype(np.float32)

    def _direction_to_brightest(self, positions, light):
        brightest    = positions[int(np.argmax(light))]
        displacement = brightest[None, :] - positions
        norm         = np.maximum(np.linalg.norm(displacement, axis=1, keepdims=True), self.EPSILON)
        return (displacement / norm).astype(np.float32)

    def _inertia_eigenvalues(self, positions, light, total_light, number_of_nodes):
        eigenvalues = self._light_covariance_eigenvalues(positions, light, total_light)
        return np.tile(eigenvalues.astype(np.float32), (number_of_nodes, 1))

    def _intensity_rank(self, light, number_of_nodes):
        ordinal = rankdata(light, method="average") - 1.0
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
            feature_columns.append(self._inertia_eigenvalues(positions, light, total_light, number_of_nodes))

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

    def _neighbor_pairs(self, number_of_nodes, neighbor_distances, neighbor_indices):
        neighbor_indices_without_self   = neighbor_indices[:, 1:]
        neighbor_distances_without_self = neighbor_distances[:, 1:]

        destination = np.repeat(np.arange(number_of_nodes), neighbor_indices_without_self.shape[1]).astype(np.int64)
        source      = neighbor_indices_without_self.reshape(-1).astype(np.int64)
        distance    = neighbor_distances_without_self.reshape(-1).astype(np.float64)
        return source, destination, distance

    def _append_pairs(self, positions, pairs, source_node_indices, destination_node_indices, distance):
        extra_source, extra_destination = pairs
        extra_distance                  = np.linalg.norm(positions[extra_destination] - positions[extra_source], axis=1).astype(np.float64)

        return (
            np.concatenate([source_node_indices,      extra_source]),
            np.concatenate([destination_node_indices, extra_destination]),
            np.concatenate([distance,                 extra_distance]),
        )

    def build(self, positions, light, neighbor_distances, neighbor_indices, extra_pairs=None):
        number_of_nodes = int(neighbor_indices.shape[0])

        source_node_indices, destination_node_indices, distance = self._neighbor_pairs(number_of_nodes, neighbor_distances, neighbor_indices)

        if extra_pairs is not None:
            source_node_indices, destination_node_indices, distance = self._append_pairs(positions, extra_pairs, source_node_indices, destination_node_indices, distance)

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

        edge_index = torch.tensor(np.vstack([source_node_indices, destination_node_indices]), dtype=torch.long)
        edge_attr  = torch.tensor(forward_features, dtype=torch.float32)
        return edge_index, edge_attr


class GraphFeatures(GraphAssembler):
    def __init__(self, config):
        super().__init__(config)
        self.moment_features = config.graph.graph_moment_features
        self.quantiles       = tuple(float(quantile) for quantile in config.graph.graph_quantiles)

    def _light_scale(self, light, number_of_nodes):
        return np.array([light.sum(), float(number_of_nodes), light.max()], dtype=np.float64)

    def _spread_scales(self, positions, light, total_light):
        return np.sqrt(self._light_covariance_eigenvalues(positions, light, total_light))

    def build(self, positions, light):
        number_of_nodes = int(positions.shape[0])
        total_light     = light.sum() + self.EPSILON

        centroid      = self._light_centroid(positions, light, total_light)
        light_scale   = self._light_scale(light, number_of_nodes)
        spread_scales = self._spread_scales(positions, light, total_light)

        blocks = [centroid, light_scale, spread_scales]

        if self.moment_features:
            axis_spread, axis_skew = self._axis_moments(positions, light, total_light, centroid)
            blocks.append(self._sharp_light_centroid(positions, light))
            blocks.append(axis_spread)
            blocks.append(axis_skew)

        if self.quantiles:
            blocks.append(self._weighted_quantiles(positions, light, self.quantiles))

        graph_features = np.concatenate(blocks).astype(np.float32)
        return torch.tensor(graph_features, dtype=torch.float32).unsqueeze(0)


class GlobalNode(GraphAssembler):
    def __init__(self, config):
        super().__init__(config)
        self.enabled = config.graph.global_node

    def append(self, positions, light):
        total_light = light.sum() + self.EPSILON
        centroid    = self._light_centroid(positions, light, total_light)

        positions = np.vstack([positions, centroid[None, :]]).astype(np.float32)
        light     = np.append(light, light.mean()).astype(np.float32)
        return positions, light

    def extend_neighbors(self, positions, light, neighbor_distances, neighbor_indices):
        neighbor_count = neighbor_indices.shape[1] - 1
        global_index   = positions.shape[0] - 1

        brightest = np.argsort(light[:global_index])[::-1][:neighbor_count]
        indices   = np.concatenate([[global_index], brightest]).astype(np.int64)
        distances = np.linalg.norm(positions[indices] - positions[global_index][None, :], axis=1).astype(np.float64)

        return np.vstack([neighbor_distances, distances[None, :]]), np.vstack([neighbor_indices, indices[None, :]])

    def pairs(self, number_of_nodes):
        real         = np.arange(number_of_nodes - 1, dtype=np.int64)
        global_index = np.full(number_of_nodes - 1, number_of_nodes - 1, dtype=np.int64)

        if self.bidirectional:
            return real, global_index

        return np.concatenate([real, global_index]), np.concatenate([global_index, real])


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

    def _enforce_floor(self, indices):
        if indices.shape[0] < self.MINIMUM_NODES:
            raise ValueError(f"Only {indices.shape[0]} sensor(s) carry light; at least {self.MINIMUM_NODES} are required to build k-nearest-neighbour edges. Raise physics.detection_efficiency or physics.scale_factor, or set graph.active_only=false.")
        return indices

    def select(self, positions, light):
        indices = self._active_indices(light)
        indices = self._apply_budget(indices, light)
        indices = self._enforce_floor(indices)

        indices = np.sort(indices)
        return positions[indices], light[indices]


class Graph:
    def __init__(self, config):
        self.config         = config
        self.node_selector  = ActiveNodeSelector(config)
        self.global_node    = GlobalNode(config)
        self.node_features  = NodeFeatures(config)
        self.edge_features  = EdgeFeatures(config)
        self.graph_features = GraphFeatures(config)
        self.graph_scalars  = config.graph.graph_scalar_features

    def build_from_arrays(self, positions, light):
        positions = np.asarray(positions, dtype=np.float32)
        light     = np.maximum(np.asarray(light, dtype=np.float32), 0.0)

        positions, light = self.node_selector.select(positions, light)

        neighbor_distances, neighbor_indices = self.node_features._nearest_neighbors(positions)
        extra_pairs                          = None
        scalar_positions, scalar_light       = positions, light

        if self.global_node.enabled:
            positions, light                     = self.global_node.append(positions, light)
            neighbor_distances, neighbor_indices = self.global_node.extend_neighbors(positions, light, neighbor_distances, neighbor_indices)
            extra_pairs                          = self.global_node.pairs(positions.shape[0])

        real_count            = scalar_positions.shape[0]
        node_features         = self.node_features.build(positions, light, neighbor_indices)
        edge_index, edge_attr = self.edge_features.build(positions, light, neighbor_distances[:real_count], neighbor_indices[:real_count], extra_pairs)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        if self.graph_scalars:
            data.graph_attr = self.graph_features.build(scalar_positions, scalar_light)
        return data


class FeatureSchema:
    PROBE_NODES = 8

    def __init__(self, config):
        self.builder = Graph(config)

    def dimensions(self):
        generator = np.random.default_rng(0)
        positions = generator.normal(size=(self.PROBE_NODES, 3)).astype(np.float32)
        light     = (generator.random(self.PROBE_NODES) + 0.1).astype(np.float32)
        data      = self.builder.build_from_arrays(positions, light)

        graph_dimension = int(data.graph_attr.shape[1]) if "graph_attr" in data else 0
        return int(data.x.shape[1]), int(data.edge_attr.shape[1]), graph_dimension


class FeatureLayout:
    AXES = ("x", "y", "z")

    def __init__(self, config):
        self.direction_features    = config.graph.direction_features
        self.inertia_features      = config.graph.inertia_features
        self.rank_features         = config.graph.rank_features
        self.edge_rbf_count        = int(config.graph.edge_rbf_count)
        self.graph_scalar_features = config.graph.graph_scalar_features
        self.graph_moment_features = config.graph.graph_moment_features
        self.graph_quantiles       = tuple(float(quantile) for quantile in config.graph.graph_quantiles)

    def node_features(self):
        layout  = [(f"position_{axis}", "position") for axis in self.AXES]
        layout += [("light_intensity",      "light_intensity")]
        layout += [("light_fraction",       "light_fraction")]
        layout += [("distance_to_centroid", "distance_to_centroid")]
        layout += [("local_light_density",  "local_light_density")]

        if self.direction_features:
            layout += [(f"direction_to_centroid_{axis}",  "direction_to_centroid")  for axis in self.AXES]
            layout += [(f"direction_to_brightest_{axis}", "direction_to_brightest") for axis in self.AXES]

        if self.inertia_features:
            layout += [(f"inertia_eigenvalue_{index}", "inertia_eigenvalues") for index in (1, 2, 3)]

        if self.rank_features:
            layout += [("intensity_rank", "intensity_rank")]

        return layout

    def edge_features(self):
        layout  = [("edge_distance", "edge_distance")]
        layout += [(f"displacement_{axis}", "displacement") for axis in self.AXES]
        layout += [("inverse_square_distance", "inverse_square_distance")]
        layout += [("light_gradient",   "light_gradient")]
        layout += [("light_similarity", "light_similarity")]

        if self.edge_rbf_count > 0:
            layout += [(f"distance_rbf_{index:02d}", "distance_rbf") for index in range(self.edge_rbf_count)]

        return layout

    def graph_features(self):
        if not self.graph_scalar_features:
            return []

        layout  = [(f"light_centroid_{axis}", "light_centroid") for axis in self.AXES]
        layout += [("total_light",       "light_scale")]
        layout += [("active_node_count", "light_scale")]
        layout += [("max_channel_light", "light_scale")]
        layout += [(f"light_spread_{index}", "light_spread") for index in (1, 2, 3)]

        if self.graph_moment_features:
            layout += [(f"sharp_centroid_{axis}", "sharp_centroid") for axis in self.AXES]
            layout += [(f"axis_spread_{axis}",    "axis_spread")    for axis in self.AXES]
            layout += [(f"axis_skew_{axis}",      "axis_skew")      for axis in self.AXES]

        for quantile in self.graph_quantiles:
            layout += [(f"light_quantile_{axis}_q{int(round(quantile * 100)):02d}", "light_quantile") for axis in self.AXES]

        return layout

    @staticmethod
    def group_index(layout):
        groups = {}
        for index, (_, group) in enumerate(layout):
            groups.setdefault(group, []).append(index)
        return groups
