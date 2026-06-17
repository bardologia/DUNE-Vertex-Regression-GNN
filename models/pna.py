from __future__ import annotations

import torch
from torch_geometric.nn import PNAConv

from models.blocks import GraphRegressor, MessagePassingEncoder


class PNA(GraphRegressor):
    def __init__(self, config):
        degree_histogram = torch.ones(config.max_degree + 1, dtype=torch.long)

        def convolution_factory(input_dim, output_dim, layer_index):
            return PNAConv(
                input_dim,
                output_dim,
                aggregators = list(config.aggregators),
                scalers     = list(config.scalers),
                deg         = degree_histogram,
                edge_dim    = config.edge_dim,
                towers      = config.towers,
                pre_layers  = config.pre_layers,
                post_layers = config.post_layers,
            )

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="attr"))
