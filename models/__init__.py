from dataclasses import fields

from configuration.architectures import ModelConfigRegistry

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



class ModelFactory:
    @staticmethod
    def by_name() -> dict:
        return {
            "edgeconv"      : DynamicEdgeConv,
            "gine"          : GINE,
            "gatv2_cascade" : GATv2,
            "gine_cascade"  : GINE,
            "gatv2"         : GATv2,
            "graphsage"     : GraphSAGE,
            "pna"           : PNA,
            "res_gated"     : ResGatedGNN,
            "gps_cascade"   : GPS,
            "general_conv"  : GeneralGNN,
            "gps"           : GPS,
            "gps_lite"      : GPSLite,
            "gcn"           : GCN,
            "transformer"   : GraphTransformer,
        }

    @classmethod
    def create(cls, name: str, config=None, **overrides):
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key not in cls.by_name():
            raise ValueError(f"Unknown model '{name}'. Available: {list(cls.by_name())}")

        if config is None:
            config = ModelConfigRegistry.by_name()[key](**overrides)
        elif overrides:
            valid_field_names = {field_definition.name for field_definition in fields(config)}
            unknown_keys      = [key_name for key_name in overrides if key_name not in valid_field_names]
            if unknown_keys:
                raise ValueError(f"Unknown override key(s) {unknown_keys} for {type(config).__name__}")
            for field_name, value in overrides.items():
                setattr(config, field_name, value)

        return cls.by_name()[key](config), config


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
