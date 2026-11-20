from configuration.architectures import ModelConfigRegistry

from .architectures import (
    DynamicEdgeConv,
    GATv2,
    GCN,
    GeneralGNN,
    GINE,
    GPS,
    GPSLite,
    GraphSAGE,
    GraphTransformer,
    ModelFactory,
    PNA,
    ResGatedGNN,
)
from .blocks        import GraphRegressor


get_model = ModelFactory.create


__all__ = [
    "GraphRegressor",
    "DynamicEdgeConv",
    "GATv2",
    "GCN",
    "GeneralGNN",
    "GINE",
    "GPS",
    "GPSLite",
    "GraphSAGE",
    "PNA",
    "ResGatedGNN",
    "GraphTransformer",
    "ModelFactory",
    "ModelConfigRegistry",
    "get_model",
]
