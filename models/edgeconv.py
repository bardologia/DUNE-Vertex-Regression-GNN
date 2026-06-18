from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import EdgeConv

from models.blocks import GraphRegressor, MessagePassingEncoder, ModuleFactory


class DynamicEdgeConv(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            multilayer_perceptron = nn.Sequential(
                nn.Linear(2 * input_dim, config.edge_mlp_hidden),
                ModuleFactory.activation(config.activation),
                nn.Linear(config.edge_mlp_hidden, output_dim),
            )
            return EdgeConv(multilayer_perceptron, aggr=config.aggregator)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="none"))
