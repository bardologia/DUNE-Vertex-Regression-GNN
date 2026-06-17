from __future__ import annotations

from torch_geometric.nn import TransformerConv

from models.blocks import GraphRegressor, MessagePassingEncoder


class GraphTransformer(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            return TransformerConv(input_dim, output_dim, heads=config.heads, concat=False, beta=config.beta, dropout=config.attention_dropout, edge_dim=config.edge_dim)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="attr"))
