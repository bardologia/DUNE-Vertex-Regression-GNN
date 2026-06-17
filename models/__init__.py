from configuration.architectures import MODEL_CONFIG_REGISTRY

from .blocks          import GraphRegressor
from .edgeconv        import DynamicEdgeConv
from .gatv2           import GATv2
from .gcn             import GCN
from .general_conv    import GeneralGNN
from .gine            import GINE
from .gps             import GPS
from .gps_lite        import GPSLite
from .graphsage       import GraphSAGE
from .pna             import PNA
from .res_gated       import ResGatedGNN
from .supergat        import SuperGAT
from .transformer_conv import GraphTransformer

MODEL_REGISTRY: dict[str, type] = {
    "gps"          : GPS,
    "gps_lite"     : GPSLite,
    "gatv2"        : GATv2,
    "gine"         : GINE,
    "graphsage"    : GraphSAGE,
    "pna"          : PNA,
    "gcn"          : GCN,
    "transformer"  : GraphTransformer,
    "edgeconv"     : DynamicEdgeConv,
    "general_conv" : GeneralGNN,
    "res_gated"    : ResGatedGNN,
    "supergat"     : SuperGAT,
}


def get_model(name: str, config=None, **overrides):
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    if config is None:
        config = MODEL_CONFIG_REGISTRY[key](**overrides)
    elif overrides:
        for field_name, value in overrides.items():
            if hasattr(config, field_name):
                setattr(config, field_name, value)

    return MODEL_REGISTRY[key](config), config


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
    "SuperGAT",
    "GraphTransformer",
    "MODEL_REGISTRY",
    "MODEL_CONFIG_REGISTRY",
    "get_model",
]
