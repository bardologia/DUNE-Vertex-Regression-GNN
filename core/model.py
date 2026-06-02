import warnings
warnings.filterwarnings('ignore', message='.*torch-scatter.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.nn import GATv2Conv, SAGPooling as PyGSAGPooling


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().clamp_(0, 1)
        bernoulli_mask = (torch.rand(shape, dtype=x.dtype, device=x.device) < keep).to(x.dtype)
        return x * bernoulli_mask / keep


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanPooling(nn.Module):
    def forward(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return scatter(features, batch, dim=0, reduce='mean')


class MaxPooling(nn.Module):
    def forward(self, features: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return scatter(features, batch, dim=0, reduce='max')


class GPSLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 7,
        heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.norm_local  = nn.LayerNorm(hidden_dim)
        self.norm_global = nn.LayerNorm(hidden_dim)
        self.norm_ffn    = nn.LayerNorm(hidden_dim)

        self.local_conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
            share_weights=False,
        )

        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.ffn       = FeedForward(hidden_dim, ffn_ratio, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for norm_module in (self.norm_local, self.norm_global, self.norm_ffn):
            nn.init.constant_(norm_module.weight, 1.0)
            nn.init.constant_(norm_module.bias, 0.0)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:

        local_representation = self.local_conv(self.norm_local(node_features), edge_index, edge_attr=edge_attr)

        normed_features                  = self.norm_global(node_features)
        dense_batch_features, valid_mask = to_dense_batch(normed_features, batch)

        B, N, D   = dense_batch_features.shape
        num_heads = self.global_attn.num_heads
        head_dim  = D // num_heads

        qkv    = F.linear(dense_batch_features, self.global_attn.in_proj_weight, self.global_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, N, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, N, num_heads, head_dim).transpose(1, 2)

        attn_mask = None
        if not valid_mask.all():
            attn_mask = valid_mask[:, None, None, :]

        dropout_p   = self.global_attn.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        attn_output = self.global_attn.out_proj(attn_output)

        global_representation = attn_output[valid_mask]

        gate_values          = self.gate(torch.cat([local_representation, global_representation], dim=-1))
        fused_representation = gate_values * local_representation + (1.0 - gate_values) * global_representation
        node_features        = node_features + self.drop_path(fused_representation)

        node_features = node_features + self.drop_path(self.ffn(self.norm_ffn(node_features)))
        return node_features


class MultiScalePool(nn.Module):
    def __init__(self, hidden_dim: int, num_levels: int = 3, ratio: float = 0.5):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_levels  = num_levels

        self.sag_pools = nn.ModuleList([
            PyGSAGPooling(hidden_dim, ratio=ratio, GNN=GATv2Conv)
            for _ in range(num_levels - 1)
        ])

        self.mean_pool = MeanPooling()
        self.max_pool  = MaxPooling()

        self.out_dim = 2 * hidden_dim * num_levels

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        readouts = []

        readouts.append(torch.cat([
            self.mean_pool(node_features, batch),
            self.max_pool(node_features, batch),
        ], dim=-1))

        node_features_h, edge_index_h, edge_attr_h, batch_h = node_features, edge_index, edge_attr, batch
        for pool in self.sag_pools:
            node_features_h, edge_index_h, edge_attr_h, batch_h, _, _ = pool(node_features_h, edge_index_h, edge_attr=edge_attr_h, batch=batch_h)
            readouts.append(torch.cat([
                self.mean_pool(node_features_h, batch_h),
                self.max_pool(node_features_h, batch_h),
            ], dim=-1))

        return torch.cat(readouts, dim=-1)


class GraphBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim      = config.model.input_dim
        hidden_dim     = config.model.gps_hidden_dim
        num_layers     = config.model.gps_num_layers
        edge_dim       = config.model.edge_dim
        heads          = config.model.gps_heads
        dropout        = config.model.gps_dropout
        attn_dropout   = config.model.gps_attn_dropout
        ffn_ratio      = config.model.gps_ffn_ratio
        drop_path_rate = config.model.drop_path_rate
        pool_levels    = config.model.pool_num_levels
        pool_ratio     = config.model.sag_ratio

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self._init_linear(self.input_proj)

        dp_rates = [r.item() for r in torch.linspace(0, drop_path_rate, num_layers)]
        self.gps_layers = nn.ModuleList([
            GPSLayer(
                hidden_dim   = hidden_dim,
                edge_dim     = edge_dim,
                heads        = heads,
                dropout      = dropout,
                attn_dropout = attn_dropout,
                ffn_ratio    = ffn_ratio,
                drop_path    = dp_rates[i],
            )
            for i in range(num_layers)
        ])

        self.pool = MultiScalePool(
            hidden_dim = hidden_dim,
            num_levels = pool_levels,
            ratio      = pool_ratio,
        )

    @staticmethod
    def _init_linear(module):
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                nn.init.kaiming_uniform_(submodule.weight, nonlinearity='relu')
                nn.init.zeros_(submodule.bias)
            elif isinstance(submodule, nn.LayerNorm):
                nn.init.constant_(submodule.weight, 1.0)
                nn.init.constant_(submodule.bias, 0.0)

    @property
    def out_dim(self) -> int:
        return self.pool.out_dim

    def forward(self, data) -> torch.Tensor:
        node_features, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = edge_attr.to(dtype=node_features.dtype)

        node_features = self.input_proj(node_features)

        for layer in self.gps_layers:
            node_features = layer(node_features, edge_index, batch, edge_attr=edge_attr)

        return self.pool(node_features, edge_index, batch, edge_attr)


class FiLMConditioner(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2),   
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
        last_linear = self.net[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        with torch.no_grad():
            last_linear.bias[: last_linear.out_features // 2] = 1.0  

    def forward(self, cond: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        params = self.net(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * features + beta


class CoordinateHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple, dropout: float = 0.1):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
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
        feature_dim     = config.model.hierarchical_feature_dim
        hidden_dims     = config.model.regression_hidden_dims
        dropout         = config.model.regression_dropout
        coord_embed_dim = config.model.coord_embed_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self._init_linear(self.feature_proj)

        self.x_head = CoordinateHead(feature_dim, hidden_dims, dropout)
        
        self.film_y = FiLMConditioner(1, feature_dim, coord_embed_dim)
        self.y_head = CoordinateHead(feature_dim, hidden_dims, dropout)

        self.film_z = FiLMConditioner(2, feature_dim, coord_embed_dim)
        self.z_head = CoordinateHead(feature_dim, hidden_dims, dropout)

    @staticmethod
    def _init_linear(module):
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                nn.init.kaiming_uniform_(submodule.weight, nonlinearity='relu')
                nn.init.zeros_(submodule.bias)
            elif isinstance(submodule, nn.LayerNorm):
                nn.init.constant_(submodule.weight, 1.0)
                nn.init.constant_(submodule.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected_features = self.feature_proj(features)

        prediction_x   = self.x_head(projected_features)                          

        features_for_y = self.film_y(prediction_x, projected_features)
        prediction_y   = self.y_head(features_for_y)                        

        cond_for_z     = torch.cat([prediction_x, prediction_y], dim=-1)     
        features_for_z = self.film_z(cond_for_z, projected_features)
        prediction_z   = self.z_head(features_for_z)                        

        return torch.cat([prediction_x, prediction_y, prediction_z], dim=-1)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone    = GraphBackbone(config)
        self.feature_dim = self.backbone.out_dim
        self.norm        = nn.LayerNorm(self.feature_dim)

        self.regression_head = HierarchicalRegressionHead(
            input_dim = self.feature_dim,
            config    = config,
        )

    def forward(self, data, return_features: bool = False):
        graph_features = self.backbone(data)
        graph_features = self.norm(graph_features)

        predictions = self.regression_head(graph_features)

        if return_features:
            features = {
                'backbone': graph_features,
                'predictions': predictions,
            }
            return predictions, features

        return predictions


__all__ = [
    "DropPath",
    "FeedForward",
    "MeanPooling",
    "MaxPooling",
    "GPSLayer",
    "MultiScalePool",
    "GraphBackbone",
    "FiLMConditioner",
    "CoordinateHead",
    "HierarchicalRegressionHead",
    "Model",
]
