import warnings
warnings.filterwarnings('ignore', message='.*torch-scatter.*')

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import GATv2Conv, SAGPooling as PyGSAGPooling


class GATBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 4,
        dropout: float = 0.2,
        use_batch_norm: bool = False,
        is_last: bool = False,
        heads: int = 4,
    ):
        super().__init__()
        self.is_last = is_last
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=int(heads),
            concat=False,
            dropout=dropout,
            edge_dim=int(edge_dim),
            add_self_loops=True,
            share_weights=False,
        )
        self.skip_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

        if not self.is_last:
            self.batch_norm = nn.LayerNorm(out_dim) if use_batch_norm else nn.Identity()
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        if self.skip_proj is not None:
            self._initialize_weights(self.skip_proj)

    @staticmethod
    def _initialize_weights(module):
        for layer in module.modules() if hasattr(module, "modules") else [module]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = edge_attr.to(dtype=x.dtype)
        x_in = x
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        if self.skip_proj is not None:
            x_in = self.skip_proj(x_in)
        x = x + x_in
        if not self.is_last:
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class MeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean = scatter(x, batch, dim=0, reduce='mean')
        return mean


class MaxPooling(nn.Module):
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        max_val = scatter(x, batch, dim=0, reduce='max')
        return max_val


class SAGPooling(nn.Module):
    def __init__(self, hidden_dim: int, ratio: float = 0.5, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.sag_pools = nn.ModuleList([PyGSAGPooling(hidden_dim, ratio=ratio, GNN=GATv2Conv) for _ in range(num_layers)])
        
        self.mean_pool = MeanPooling()
        self.max_pool  = MaxPooling()
        
        self.out_dim = hidden_dim * 2
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        xs = []
        
        for i, pool in enumerate(self.sag_pools):
            x, edge_index, edge_attr, batch, perm, score = pool(x, edge_index, edge_attr=edge_attr, batch=batch)
            xs.append(x)
        
        x_final = xs[-1]
        
        mean_pool = self.mean_pool(x_final, batch)
        max_pool = self.max_pool(x_final, batch)
        
        return torch.cat([mean_pool, max_pool], dim=-1)


class GraphBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim       = config.model.input_dim
        self.gcn_hidden_dims = list(config.model.gcn_hidden_dims)
        edge_dim             = config.model.edge_dim
        gat_heads            = config.model.gat_heads
        use_batch_norm       = config.model.gat_use_batch_norm
        dropout              = config.model.gat_dropout
        sag_ratio            = config.model.sag_ratio
        sag_layers           = config.model.sag_layers

        self.gcn_layers = nn.ModuleList()
        gcn_dims = [self.input_dim] + self.gcn_hidden_dims
        
        for i in range(len(gcn_dims) - 1):
            is_last = i == len(gcn_dims) - 2
            
            self.gcn_layers.append(
                GATBlock(
                    in_dim         = gcn_dims[i],
                    out_dim        = gcn_dims[i + 1],
                    edge_dim       = edge_dim,
                    dropout        = dropout,
                    use_batch_norm = use_batch_norm,
                    is_last        = is_last,
                    heads          = gat_heads,
                )
            )
           
        self.pool = SAGPooling(
            hidden_dim = self.gcn_hidden_dims[-1],
            ratio      = sag_ratio,
            num_layers = sag_layers
        )

    @property
    def out_dim(self) -> int:
        return self.pool.out_dim

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        
        for layer in self.gcn_layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
        
        return self.pool(x, edge_index, batch, edge_attr)


class RegressionHead(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        hidden_dims = config.model.regression_hidden_dims
        output_dim  = config.model.regression_output_dim
        dropout     = config.model.regression_dropout
        
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.network = nn.Sequential(*layers)
        
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config   = config
        
        self.backbone = GraphBackbone(self.config)
        
        self.feature_dim     = self.backbone.out_dim
        self.regression_head = RegressionHead(
            input_dim   = self.feature_dim,
            config      = self.config
        )
        
        self.norm = nn.LayerNorm(self.feature_dim)

    def forward(self, data, return_features: bool = False):
        graph_features = self.backbone(data)
        graph_features = self.norm(graph_features)
        
        predictions = self.regression_head(graph_features)
        
        if return_features:
            features = {
                'backbone': graph_features,
                'predictions': predictions
            }
            return predictions, features
        
        return predictions


__all__ = [
    "GATBlock",
    "MeanPooling",
    "MaxPooling",
    "SAGPooling",
    "GraphBackbone",
    "RegressionHead",
    "Model",
    "MeanPooling",
    "MaxPooling",
]
