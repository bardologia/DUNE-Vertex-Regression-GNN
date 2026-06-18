from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import GINEConv

from models.blocks import GraphRegressor, MessagePassingEncoder, build_activation


class GINE(GraphRegressor):
    def __init__(self, config):
        def convolution_factory(input_dim, output_dim, layer_index):
            multilayer_perceptron = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                build_activation(config.activation),
                nn.Linear(output_dim, output_dim),
            )
            return GINEConv(multilayer_perceptron, train_eps=config.train_eps, edge_dim=config.edge_dim)

        super().__init__(config, MessagePassingEncoder(config, convolution_factory, edge_mode="attr"))
