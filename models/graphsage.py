from __future__ import annotations

from torch_geometric.nn import SAGEConv

from models.blocks import GraphRegressor, MessagePassingEncoder


class GraphSAGE(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            return SAGEConv(input_dim, output_dim, aggr=config.aggregator)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="none"))
