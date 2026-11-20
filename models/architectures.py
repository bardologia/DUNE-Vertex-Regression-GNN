from __future__ import annotations

from dataclasses import fields

import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, GATv2Conv, GCNConv, GeneralConv, GINEConv, PNAConv, ResGatedGraphConv, SAGEConv, TransformerConv

from configuration.architectures import ModelConfigRegistry
from models.blocks               import GPSEncoder, GraphRegressor, MessagePassingEncoder, ModuleFactory


class GPS(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, GPSEncoder(config))


class GPSLite(GPS):
    pass


class GATv2(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="attr"))

    @staticmethod
    def convolution(config):
        return GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.heads, concat=False, dropout=config.attention_dropout, edge_dim=config.edge_dim, add_self_loops=True)


class GraphTransformer(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="attr"))

    @staticmethod
    def convolution(config):
        return TransformerConv(config.hidden_dim, config.hidden_dim, heads=config.heads, concat=False, beta=config.beta, dropout=config.attention_dropout, edge_dim=config.edge_dim)


class GCN(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="none"))

    @staticmethod
    def convolution(config):
        return GCNConv(config.hidden_dim, config.hidden_dim, improved=config.improved, add_self_loops=config.add_self_loops)


class GraphSAGE(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="none"))

    @staticmethod
    def convolution(config):
        return SAGEConv(config.hidden_dim, config.hidden_dim, aggr=config.aggregator)


class GINE(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="attr"))

    @staticmethod
    def convolution(config):
        multilayer_perceptron = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            ModuleFactory.activation(config.activation),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        return GINEConv(multilayer_perceptron, train_eps=config.train_eps, edge_dim=config.edge_dim)


class GeneralGNN(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="attr"))

    @staticmethod
    def convolution(config):
        return GeneralConv(config.hidden_dim, config.hidden_dim, in_edge_channels=config.edge_dim, heads=config.heads, attention=config.attention)


class ResGatedGNN(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="none"))

    @staticmethod
    def convolution(config):
        return ResGatedGraphConv(config.hidden_dim, config.hidden_dim)


class PNA(GraphRegressor):
    def __init__(self, config):
        if not config.degree_histogram:
            raise ValueError("PNA requires a degree histogram computed from the training graphs; none was provided in the config.")

        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="attr"))

    @staticmethod
    def convolution(config):
        return PNAConv(
            config.hidden_dim,
            config.hidden_dim,
            aggregators = list(config.aggregators),
            scalers     = list(config.scalers),
            deg         = torch.tensor(config.degree_histogram, dtype=torch.long),
            edge_dim    = config.edge_dim,
            towers      = config.towers,
            pre_layers  = config.pre_layers,
            post_layers = config.post_layers,
        )


class DynamicEdgeConv(GraphRegressor):
    def __init__(self, config):
        super().__init__(config, MessagePassingEncoder(config, self.convolution, edge_mode="none"))

    @staticmethod
    def convolution(config):
        multilayer_perceptron = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.edge_mlp_hidden),
            ModuleFactory.activation(config.activation),
            nn.Linear(config.edge_mlp_hidden, config.hidden_dim),
        )
        return EdgeConv(multilayer_perceptron, aggr=config.aggregator)


class ModelFactory:
    @staticmethod
    def by_name() -> dict:
        return {
            "edgeconv"      : DynamicEdgeConv,
            "gine"          : GINE,
            "gatv2_cascade" : GATv2,
            "gine_cascade"  : GINE,
            "gatv2"         : GATv2,
            "graphsage"     : GraphSAGE,
            "pna"           : PNA,
            "res_gated"     : ResGatedGNN,
            "gps_cascade"   : GPS,
            "general_conv"  : GeneralGNN,
            "gps"           : GPS,
            "gps_lite"      : GPSLite,
            "gcn"           : GCN,
            "transformer"   : GraphTransformer,
        }

    @classmethod
    def create(cls, name: str, config=None, **overrides):
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key not in cls.by_name():
            raise ValueError(f"Unknown model '{name}'. Available: {list(cls.by_name())}")

        if config is None:
            config = ModelConfigRegistry.by_name()[key](**overrides)
        elif overrides:
            valid_field_names = {field_definition.name for field_definition in fields(config)}
            unknown_keys      = [key_name for key_name in overrides if key_name not in valid_field_names]
            if unknown_keys:
                raise ValueError(f"Unknown override key(s) {unknown_keys} for {type(config).__name__}")
            for field_name, value in overrides.items():
                setattr(config, field_name, value)

        return cls.by_name()[key](config), config
