from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class BaseGNNConfig:
    name              : str   = "base"

    input_dim         : int   = 8
    edge_dim          : int   = 7
    output_dim        : int   = 3

    hidden_dim        : int   = 128
    num_layers        : int   = 4
    dropout           : float = 0.1
    activation        : str   = "gelu"
    normalization     : str   = "layer"
    drop_path_rate    : float = 0.1

    pooling           : str   = "mean_max"
    pool_num_levels   : int   = 3
    sag_ratio         : float = 0.5
    set2set_steps     : int   = 3

    head_type                : str   = "hierarchical"
    hierarchical_feature_dim : int   = 256
    coord_embed_dim          : int   = 64
    regression_hidden_dims   : tuple = (128, 64)
    regression_dropout       : float = 0.1

    SECTIONS = {
        "dimensions" : ("input_dim", "edge_dim", "output_dim"),
        "encoder"    : ("hidden_dim", "num_layers", "dropout", "activation", "normalization", "drop_path_rate", "heads", "attention_dropout", "ffn_ratio"),
        "pooling"    : ("pooling", "pool_num_levels", "sag_ratio", "set2set_steps"),
        "head"       : ("head_type", "hierarchical_feature_dim", "coord_embed_dim", "regression_hidden_dims", "regression_dropout"),
    }

    @classmethod
    def _field_default(cls, field_name):
        for field_definition in fields(cls):
            if field_definition.name == field_name:
                return field_definition.default
        return None

    @classmethod
    def tunable_params(cls) -> dict:
        field_names = {field_definition.name for field_definition in fields(cls)}
        head_type   = cls._field_default("head_type")
        pooling     = cls._field_default("pooling")

        space = {
            "hidden_dim"         : {"type": "categorical", "choices": [64, 128, 192, 256]},
            "num_layers"         : {"type": "int",         "low": 2, "high": 6},
            "dropout"            : {"type": "float",       "low": 0.0, "high": 0.2, "step": 0.05},
            "drop_path_rate"     : {"type": "float",       "low": 0.0, "high": 0.2, "step": 0.05},
            "regression_dropout" : {"type": "float",       "low": 0.0, "high": 0.2, "step": 0.05},
        }

        if head_type in ("hierarchical", "cascade"):
            space["hierarchical_feature_dim"] = {"type": "categorical", "choices": [128, 256, 384]}
        if head_type == "hierarchical":
            space["coord_embed_dim"]          = {"type": "categorical", "choices": [32, 64, 128]}

        if "heads" in field_names:
            space["heads"]             = {"type": "categorical", "choices": [2, 4, 8]}
        if "attention_dropout" in field_names:
            space["attention_dropout"] = {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05}
        if pooling == "sagpool_multiscale":
            space["pool_num_levels"]   = {"type": "int",   "low": 2, "high": 4}
            space["sag_ratio"]         = {"type": "float", "low": 0.3, "high": 0.7, "step": 0.1}

        return space


@dataclass
class GPSConfig(BaseGNNConfig):
    name              : str   = "gps"
    heads             : int   = 4
    attention_dropout : float = 0.1
    ffn_ratio         : float = 4.0
    pooling           : str   = "sagpool_multiscale"


@dataclass
class GPSLiteConfig(BaseGNNConfig):
    name              : str   = "gps_lite"
    hidden_dim        : int   = 96
    num_layers        : int   = 3
    heads             : int   = 4
    attention_dropout : float = 0.1
    ffn_ratio         : float = 2.0
    pooling           : str   = "mean_max"
    pool_num_levels   : int   = 1
    head_type         : str   = "mlp"


@dataclass
class GATv2Config(BaseGNNConfig):
    name              : str   = "gatv2"
    heads             : int   = 4
    attention_dropout : float = 0.1


@dataclass
class GINEConfig(BaseGNNConfig):
    name      : str  = "gine"
    train_eps : bool = True


@dataclass
class GraphSAGEConfig(BaseGNNConfig):
    name       : str = "graphsage"
    aggregator : str = "mean"


@dataclass
class PNAConfig(BaseGNNConfig):
    name             : str   = "pna"
    aggregators      : tuple = ("mean", "min", "max", "std")
    scalers          : tuple = ("identity", "amplification", "attenuation")
    towers           : int   = 1
    pre_layers       : int   = 1
    post_layers      : int   = 1
    degree_histogram : tuple = ()


@dataclass
class GCNConfig(BaseGNNConfig):
    name           : str  = "gcn"
    improved       : bool = False
    add_self_loops : bool = True


@dataclass
class TransformerConvConfig(BaseGNNConfig):
    name              : str   = "transformer"
    heads             : int   = 4
    attention_dropout : float = 0.1
    beta              : bool  = True


@dataclass
class EdgeConvConfig(BaseGNNConfig):
    name            : str = "edgeconv"
    aggregator      : str = "max"
    edge_mlp_hidden : int = 128


@dataclass
class GeneralConvConfig(BaseGNNConfig):
    name      : str  = "general_conv"
    heads     : int  = 4
    attention : bool = True


@dataclass
class ResGatedConfig(BaseGNNConfig):
    name : str = "res_gated"


@dataclass
class SuperGATConfig(BaseGNNConfig):
    name              : str   = "supergat"
    heads             : int   = 4
    attention_dropout : float = 0.1
    attention_type    : str   = "MX"


@dataclass
class GPSCascadeConfig(GPSConfig):
    name      : str = "gps_cascade"
    head_type : str = "cascade"


@dataclass
class GATv2CascadeConfig(GATv2Config):
    name      : str = "gatv2_cascade"
    head_type : str = "cascade"


@dataclass
class GINECascadeConfig(GINEConfig):
    name      : str = "gine_cascade"
    head_type : str = "cascade"


MODEL_CONFIG_REGISTRY: dict[str, type] = {
    "gps"           : GPSConfig,
    "gps_lite"      : GPSLiteConfig,
    "gps_cascade"   : GPSCascadeConfig,
    "gatv2"         : GATv2Config,
    "gatv2_cascade" : GATv2CascadeConfig,
    "gine"          : GINEConfig,
    "gine_cascade"  : GINECascadeConfig,
    "graphsage"     : GraphSAGEConfig,
    "pna"           : PNAConfig,
    "gcn"           : GCNConfig,
    "transformer"   : TransformerConvConfig,
    "edgeconv"      : EdgeConvConfig,
    "general_conv"  : GeneralConvConfig,
    "res_gated"     : ResGatedConfig,
    "supergat"      : SuperGATConfig,
}
