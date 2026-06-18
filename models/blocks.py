from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn    import SAGPooling, Set2Set, GATv2Conv
from torch_geometric.utils import scatter, to_dense_batch


class ModuleFactory:
    ACTIVATIONS = {
        "gelu"       : nn.GELU,
        "relu"       : nn.ReLU,
        "elu"        : nn.ELU,
        "leaky_relu" : nn.LeakyReLU,
        "tanh"       : nn.Tanh,
    }

    @classmethod
    def activation(cls, name: str) -> nn.Module:
        return cls.ACTIVATIONS[name]()

    @classmethod
    def norm(cls, name: str, dimension: int) -> nn.Module:
        if name == "layer":
            return nn.LayerNorm(dimension)
        if name == "batch":
            return nn.BatchNorm1d(dimension)
        if name == "none":
            return nn.Identity()
        raise ValueError(f"Unknown normalization '{name}'")

    @classmethod
    def pooling(cls, config, hidden_dim: int) -> nn.Module:
        if config.pooling == "mean_max":
            return MeanMaxPool(hidden_dim)
        if config.pooling == "sagpool_multiscale":
            return MultiScalePool(hidden_dim, num_levels=config.pool_num_levels, ratio=config.sag_ratio)
        if config.pooling == "set2set":
            return Set2SetPool(hidden_dim, processing_steps=config.set2set_steps)
        raise ValueError(f"Unknown pooling '{config.pooling}'")

    @classmethod
    def head(cls, config, input_dim: int) -> nn.Module:
        if config.head_type == "hierarchical":
            return HierarchicalRegressionHead(input_dim, config)
        if config.head_type == "cascade":
            return CascadeRegressionHead(input_dim, config)
        if config.head_type == "mlp":
            return MLPRegressionHead(input_dim, config)
        raise ValueError(f"Unknown head type '{config.head_type}'")


class DropPath(nn.Module):
    def __init__(self, drop_probability: float = 0.0):
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        if not self.training or self.drop_probability == 0.0:
            return x

        keep_probability = 1.0 - self.drop_probability

        if batch is None:
            mask_shape     = (x.shape[0],) + (1,) * (x.ndim - 1)
            bernoulli_mask = (torch.rand(mask_shape, dtype=x.dtype, device=x.device) < keep_probability).to(x.dtype)
            return x * bernoulli_mask / keep_probability

        number_of_graphs = int(batch.max().item()) + 1
        graph_mask       = (torch.rand(number_of_graphs, dtype=x.dtype, device=x.device) < keep_probability).to(x.dtype)
        node_mask        = graph_mask[batch].view((-1,) + (1,) * (x.ndim - 1))
        return x * node_mask / keep_probability


class FeedForward(nn.Module):
    def __init__(self, dimension: int, ffn_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden   = int(dimension * ffn_ratio)
        self.net = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanPooling(nn.Module):
    def forward(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return scatter(features, batch, dim=0, reduce="mean")


class MaxPooling(nn.Module):
    def forward(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return scatter(features, batch, dim=0, reduce="max")


class MeanMaxPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mean_pool = MeanPooling()
        self.max_pool  = MaxPooling()
        self.out_dim   = 2 * hidden_dim

    def forward(self, node_features, edge_index, batch, edge_attr=None) -> torch.Tensor:
        return torch.cat([self.mean_pool(node_features, batch), self.max_pool(node_features, batch)], dim=-1)


class MultiScalePool(nn.Module):
    def __init__(self, hidden_dim: int, num_levels: int = 3, ratio: float = 0.5):
        super().__init__()
        self.sag_pools = nn.ModuleList([SAGPooling(hidden_dim, ratio=ratio, GNN=GATv2Conv) for _ in range(num_levels - 1)])
        self.mean_pool = MeanPooling()
        self.max_pool  = MaxPooling()
        self.out_dim   = 2 * hidden_dim * num_levels

    def forward(self, node_features, edge_index, batch, edge_attr=None) -> torch.Tensor:
        readouts = [torch.cat([self.mean_pool(node_features, batch), self.max_pool(node_features, batch)], dim=-1)]

        current_features, current_edge_index, current_edge_attr, current_batch = node_features, edge_index, edge_attr, batch
        for pool in self.sag_pools:
            current_features, current_edge_index, current_edge_attr, current_batch, _, _ = pool(current_features, current_edge_index, edge_attr=current_edge_attr, batch=current_batch)
            readouts.append(torch.cat([self.mean_pool(current_features, current_batch), self.max_pool(current_features, current_batch)], dim=-1))

        return torch.cat(readouts, dim=-1)


class Set2SetPool(nn.Module):
    def __init__(self, hidden_dim: int, processing_steps: int = 3):
        super().__init__()
        self.set2set = Set2Set(hidden_dim, processing_steps=processing_steps)
        self.out_dim = 2 * hidden_dim

    def forward(self, node_features, edge_index, batch, edge_attr=None) -> torch.Tensor:
        return self.set2set(node_features, batch)


class GPSLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim=7, heads=4, dropout=0.1, attention_dropout=0.1, ffn_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm_local  = nn.LayerNorm(hidden_dim)
        self.norm_global = nn.LayerNorm(hidden_dim)
        self.norm_ffn    = nn.LayerNorm(hidden_dim)

        self.local_conv  = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout, edge_dim=edge_dim, add_self_loops=True, share_weights=False)
        self.global_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=attention_dropout, batch_first=True)

        self.gate        = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.ffn         = FeedForward(hidden_dim, ffn_ratio, dropout)
        self.drop_path   = DropPath(drop_path)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for norm_module in (self.norm_local, self.norm_global, self.norm_ffn):
            nn.init.constant_(norm_module.weight, 1.0)
            nn.init.constant_(norm_module.bias, 0.0)

    def forward(self, node_features, edge_index, batch, edge_attr=None) -> torch.Tensor:
        local_representation = self.local_conv(self.norm_local(node_features), edge_index, edge_attr=edge_attr)

        normalized_features              = self.norm_global(node_features)
        dense_batch_features, valid_mask = to_dense_batch(normalized_features, batch)

        batch_size, max_nodes, dimension = dense_batch_features.shape
        number_of_heads                  = self.global_attn.num_heads
        head_dimension                   = dimension // number_of_heads

        projected                = F.linear(dense_batch_features, self.global_attn.in_proj_weight, self.global_attn.in_proj_bias)
        queries, keys, values    = projected.chunk(3, dim=-1)
        queries = queries.reshape(batch_size, max_nodes, number_of_heads, head_dimension).transpose(1, 2)
        keys    = keys.reshape(batch_size, max_nodes, number_of_heads, head_dimension).transpose(1, 2)
        values  = values.reshape(batch_size, max_nodes, number_of_heads, head_dimension).transpose(1, 2)

        attention_mask = None
        if not valid_mask.all():
            attention_mask = valid_mask[:, None, None, :]

        dropout_probability = self.global_attn.dropout if self.training else 0.0
        attention_output    = F.scaled_dot_product_attention(queries, keys, values, attn_mask=attention_mask, dropout_p=dropout_probability)
        attention_output    = attention_output.transpose(1, 2).reshape(batch_size, max_nodes, dimension)
        attention_output    = self.global_attn.out_proj(attention_output)

        global_representation = attention_output[valid_mask]

        gate_values          = self.gate(torch.cat([local_representation, global_representation], dim=-1))
        fused_representation  = gate_values * local_representation + (1.0 - gate_values) * global_representation
        node_features         = node_features + self.drop_path(fused_representation, batch)

        node_features = node_features + self.drop_path(self.ffn(self.norm_ffn(node_features)), batch)
        return node_features


class GPSEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim))
        self._initialize_projection()

        drop_path_rates = [rate.item() for rate in torch.linspace(0, config.drop_path_rate, config.num_layers)]
        self.layers     = nn.ModuleList([
            GPSLayer(
                hidden_dim        = config.hidden_dim,
                edge_dim          = config.edge_dim,
                heads             = config.heads,
                dropout           = config.dropout,
                attention_dropout = config.attention_dropout,
                ffn_ratio         = config.ffn_ratio,
                drop_path         = drop_path_rates[index],
            )
            for index in range(config.num_layers)
        ])
        self.out_dim = config.hidden_dim

    def _initialize_projection(self):
        for module in self.input_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, node_features, edge_index, batch, edge_attr=None) -> torch.Tensor:
        node_features = self.input_proj(node_features)
        for layer in self.layers:
            node_features = layer(node_features, edge_index, batch, edge_attr=edge_attr)
        return node_features


class MessagePassingEncoder(nn.Module):
    def __init__(self, config, convolution_factory, edge_mode: str = "attr"):
        super().__init__()
        self.edge_mode  = edge_mode
        self.input_proj = nn.Sequential(nn.Linear(config.input_dim, config.hidden_dim), nn.LayerNorm(config.hidden_dim))

        self.convolutions = nn.ModuleList([convolution_factory(config.hidden_dim, config.hidden_dim, index) for index in range(config.num_layers)])
        self.norms        = nn.ModuleList([ModuleFactory.norm(config.normalization, config.hidden_dim) for _ in range(config.num_layers)])
        self.activation   = ModuleFactory.activation(config.activation)
        self.dropout      = nn.Dropout(config.dropout)

        drop_path_rates = [rate.item() for rate in torch.linspace(0, config.drop_path_rate, config.num_layers)]
        self.drop_paths = nn.ModuleList([DropPath(rate) for rate in drop_path_rates])

        self.out_dim = config.hidden_dim
        self._initialize_projection()

    def _initialize_projection(self):
        for module in self.input_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def _apply_convolution(self, convolution, node_features, edge_index, edge_attr):
        if self.edge_mode == "attr" and edge_attr is not None:
            return convolution(node_features, edge_index, edge_attr)
        return convolution(node_features, edge_index)

    def forward(self, node_features, edge_index, batch, edge_attr=None) -> torch.Tensor:
        node_features = self.input_proj(node_features)

        for convolution, norm, drop_path in zip(self.convolutions, self.norms, self.drop_paths):
            normalized = norm(node_features)
            updated    = self._apply_convolution(convolution, normalized, edge_index, edge_attr)
            node_features = node_features + drop_path(self.dropout(self.activation(updated)), batch)

        return node_features


class FiLMConditioner(nn.Module):
    def __init__(self, condition_dim: int, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        with torch.no_grad():
            last_linear.bias[: last_linear.out_features // 2] = 1.0

    def forward(self, condition: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return gamma * features + beta


class CoordinateHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple, dropout: float = 0.1):
        super().__init__()
        dimensions = [input_dim] + list(hidden_dims) + [1]
        layers     = []
        for index in range(len(dimensions) - 2):
            layers += [nn.Linear(dimensions[index], dimensions[index + 1]), nn.LayerNorm(dimensions[index + 1]), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        last_linear = self.net[-1]
        nn.init.xavier_uniform_(last_linear.weight, gain=0.01)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalRegressionHead(nn.Module):
    def __init__(self, input_dim: int, config):
        super().__init__()
        feature_dim     = config.hierarchical_feature_dim
        hidden_dims     = config.regression_hidden_dims
        dropout         = config.regression_dropout
        coordinate_embedding_dim = config.coord_embed_dim

        self.feature_proj = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU(), nn.Dropout(dropout))
        self._initialize_projection()

        self.x_head = CoordinateHead(feature_dim, hidden_dims, dropout)
        self.film_y = FiLMConditioner(1, feature_dim, coordinate_embedding_dim)
        self.y_head = CoordinateHead(feature_dim, hidden_dims, dropout)
        self.film_z = FiLMConditioner(2, feature_dim, coordinate_embedding_dim)
        self.z_head = CoordinateHead(feature_dim, hidden_dims, dropout)

    def _initialize_projection(self):
        for module in self.feature_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected_features = self.feature_proj(features)

        prediction_x   = self.x_head(projected_features)

        features_for_y = self.film_y(prediction_x, projected_features)
        prediction_y   = self.y_head(features_for_y)

        condition_for_z = torch.cat([prediction_x, prediction_y], dim=-1)
        features_for_z  = self.film_z(condition_for_z, projected_features)
        prediction_z    = self.z_head(features_for_z)

        return torch.cat([prediction_x, prediction_y, prediction_z], dim=-1)


class MLPRegressionHead(nn.Module):
    def __init__(self, input_dim: int, config):
        super().__init__()
        dimensions = [input_dim] + list(config.regression_hidden_dims)
        layers     = []
        for index in range(len(dimensions) - 1):
            layers += [nn.Linear(dimensions[index], dimensions[index + 1]), nn.LayerNorm(dimensions[index + 1]), nn.GELU(), nn.Dropout(config.regression_dropout)]
        layers.append(nn.Linear(dimensions[-1], config.output_dim))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CascadeRegressionHead(nn.Module):
    def __init__(self, input_dim: int, config):
        super().__init__()
        feature_dim = config.hierarchical_feature_dim
        hidden_dims = config.regression_hidden_dims
        dropout     = config.regression_dropout

        self.feature_proj = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU(), nn.Dropout(dropout))
        self._initialize_projection()

        self.x_head = CoordinateHead(feature_dim,     hidden_dims, dropout)
        self.y_head = CoordinateHead(feature_dim + 1, hidden_dims, dropout)
        self.z_head = CoordinateHead(feature_dim + 2, hidden_dims, dropout)

    def _initialize_projection(self):
        for module in self.feature_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected_features = self.feature_proj(features)

        prediction_x = self.x_head(projected_features)
        prediction_y = self.y_head(torch.cat([projected_features, prediction_x], dim=-1))
        prediction_z = self.z_head(torch.cat([projected_features, prediction_x, prediction_y], dim=-1))

        return torch.cat([prediction_x, prediction_y, prediction_z], dim=-1)


class GraphRegressor(nn.Module):
    def __init__(self, config, encoder: nn.Module):
        super().__init__()
        self.config          = config
        self.encoder         = encoder
        self.pool            = ModuleFactory.pooling(config, encoder.out_dim)
        self.norm            = nn.LayerNorm(self.pool.out_dim)
        self.regression_head = ModuleFactory.head(config, self.pool.out_dim)

    def _prepare_edge_attributes(self, data, reference):
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        return edge_attr.to(dtype=reference.dtype)

    def forward(self, data, return_features: bool = False):
        node_features = data.x
        edge_index    = data.edge_index
        batch         = data.batch
        edge_attr     = self._prepare_edge_attributes(data, node_features)

        node_embeddings  = self.encoder(node_features, edge_index, batch, edge_attr=edge_attr)
        graph_embedding  = self.pool(node_embeddings, edge_index, batch, edge_attr=edge_attr)
        graph_embedding  = self.norm(graph_embedding)
        predictions      = self.regression_head(graph_embedding)

        if return_features:
            return predictions, {"graph_embedding": graph_embedding, "predictions": predictions}
        return predictions
