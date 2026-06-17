from __future__ import annotations

from torch_geometric.nn import ResGatedGraphConv

from models.blocks import GraphRegressor, MessagePassingEncoder


class ResGatedGNN(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            return ResGatedGraphConv(input_dim, output_dim)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="none"))
