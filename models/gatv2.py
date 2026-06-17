from __future__ import annotations

from torch_geometric.nn import GATv2Conv

from models.blocks import GraphRegressor, MessagePassingEncoder


class GATv2(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            return GATv2Conv(input_dim, output_dim, heads=config.heads, concat=False, dropout=config.attention_dropout, edge_dim=config.edge_dim, add_self_loops=True)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="attr"))
