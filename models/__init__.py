from dataclasses import fields

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
from .transformer_conv import GraphTransformer

MODEL_REGISTRY: dict[str, type] = {
    "gps"           : GPS,
    "gps_lite"      : GPSLite,
    "gps_cascade"   : GPS,
    "gatv2"         : GATv2,
    "gatv2_cascade" : GATv2,
    "gine"          : GINE,
    "gine_cascade"  : GINE,
    "graphsage"     : GraphSAGE,
    "pna"           : PNA,
    "gcn"           : GCN,
    "transformer"   : GraphTransformer,
    "edgeconv"      : DynamicEdgeConv,
    "general_conv"  : GeneralGNN,
    "res_gated"     : ResGatedGNN,
}


class ModelFactory:
    @classmethod
    def create(cls, name: str, config=None, **overrides):
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

        if config is None:
            config = MODEL_CONFIG_REGISTRY[key](**overrides)
        elif overrides:
            valid_field_names = {field_definition.name for field_definition in fields(config)}
            unknown_keys      = [key_name for key_name in overrides if key_name not in valid_field_names]
            if unknown_keys:
                raise ValueError(f"Unknown override key(s) {unknown_keys} for {type(config).__name__}")
            for field_name, value in overrides.items():
                setattr(config, field_name, value)

        return MODEL_REGISTRY[key](config), config


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
    "MODEL_REGISTRY",
    "MODEL_CONFIG_REGISTRY",
    "get_model",
]
