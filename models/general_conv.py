from __future__ import annotations

from torch_geometric.nn import GeneralConv

from models.blocks import GraphRegressor, MessagePassingEncoder


class GeneralGNN(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            return GeneralConv(input_dim, output_dim, in_edge_channels=config.edge_dim, heads=config.heads, attention=config.attention)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="attr"))
