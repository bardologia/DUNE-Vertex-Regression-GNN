from __future__ import annotations

from torch_geometric.nn import SuperGATConv

from models.blocks import GraphRegressor, MessagePassingEncoder


class SuperGAT(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            return SuperGATConv(input_dim, output_dim, heads=config.heads, concat=False, dropout=config.attention_dropout, attention_type=config.attention_type, add_self_loops=True)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="none"))
