from __future__ import annotations

from torch_geometric.nn import GCNConv

from models.blocks import GraphRegressor, MessagePassingEncoder


class GCN(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            return GCNConv(input_dim, output_dim, improved=config.improved, add_self_loops=config.add_self_loops)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="none"))
