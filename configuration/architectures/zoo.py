from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class BaseGNNConfig:
    name              : str   = "base"

    input_dim         : int   = 17
    edge_dim          : int   = 23
    graph_dim         : int   = 33
    output_dim        : int   = 3

    output_clamp      : bool  = True
    clamp_leak        : float = 0.01
    clamp_lower       : tuple = ()
    clamp_upper       : tuple = ()

    hidden_dim        : int   = 128
    num_layers        : int   = 4
    dropout           : float = 0.0
    activation        : str   = "gelu"
    normalization     : str   = "layer"
    drop_path_rate    : float = 0.0

    pooling           : str   = "mean_max_sum"
    pool_num_levels   : int   = 3
    sag_ratio         : float = 0.5
    set2set_steps     : int   = 3
    graph_embed_dim   : int   = 128

    head_type                : str   = "hierarchical"
    hierarchical_feature_dim : int   = 256
    coord_embed_dim          : int   = 64
    regression_hidden_dims   : tuple = (128, 64)
    regression_dropout       : float = 0.0

    optimizer_overrides      : dict  = field(default_factory=dict)

    @classmethod
    def tunable_params(cls) -> dict:
        defaults = {field_definition.name: field_definition.default for field_definition in fields(cls)}

        space = {
            "hidden_dim"         : {"type": "categorical", "choices": list(ArchitectureSearchSpace.hidden_dims)},
            "num_layers"         : {"type": "int",         "low": ArchitectureSearchSpace.layers_low, "high": ArchitectureSearchSpace.layers_high},
            "dropout"            : {"type": "float",       "low": ArchitectureSearchSpace.rate_low, "high": ArchitectureSearchSpace.rate_high, "step": ArchitectureSearchSpace.rate_step},
            "drop_path_rate"     : {"type": "float",       "low": ArchitectureSearchSpace.rate_low, "high": ArchitectureSearchSpace.rate_high, "step": ArchitectureSearchSpace.rate_step},
            "regression_dropout" : {"type": "float",       "low": ArchitectureSearchSpace.rate_low, "high": ArchitectureSearchSpace.rate_high, "step": ArchitectureSearchSpace.rate_step},
        }

        if defaults["head_type"] in ("hierarchical", "cascade"):
            space["hierarchical_feature_dim"] = {"type": "categorical", "choices": list(ArchitectureSearchSpace.hierarchical_feature_dims)}
        if defaults["head_type"] == "hierarchical":
            space["coord_embed_dim"]          = {"type": "categorical", "choices": list(ArchitectureSearchSpace.coord_embed_dims)}

        if "heads" in defaults:
            space["heads"]             = {"type": "categorical", "choices": list(ArchitectureSearchSpace.heads)}
        if "attention_dropout" in defaults:
            space["attention_dropout"] = {"type": "float", "low": ArchitectureSearchSpace.rate_low, "high": ArchitectureSearchSpace.rate_high, "step": ArchitectureSearchSpace.rate_step}
        if defaults["pooling"] == "sagpool_multiscale":
            space["pool_num_levels"]   = {"type": "int",   "low": ArchitectureSearchSpace.pool_levels_low, "high": ArchitectureSearchSpace.pool_levels_high}
            space["sag_ratio"]         = {"type": "float", "low": ArchitectureSearchSpace.sag_ratio_low, "high": ArchitectureSearchSpace.sag_ratio_high, "step": ArchitectureSearchSpace.sag_ratio_step}

        return space


@dataclass
class GPSConfig(BaseGNNConfig):
    name              : str   = "gps"
    heads             : int   = 4
    attention_dropout : float = 0.0
    ffn_ratio         : float = 4.0
    pooling           : str   = "mean_max_sum"
    drop_path_rate    : float = 0.0

    optimizer_overrides : dict = field(default_factory=lambda: {
        "learning_rate_encoder"         : 1.5e-4,
        "learning_rate_pool"            : 1.5e-4,
        "learning_rate_regression_head" : 1.5e-3,
    })


@dataclass
class GPSLiteConfig(BaseGNNConfig):
    name              : str   = "gps_lite"
    hidden_dim        : int   = 96
    num_layers        : int   = 3
    heads             : int   = 4
    attention_dropout : float = 0.0
    ffn_ratio         : float = 2.0
    pooling           : str   = "mean_max_sum"
    pool_num_levels   : int   = 1
    head_type         : str   = "mlp"


@dataclass
class GATv2Config(BaseGNNConfig):
    name              : str   = "gatv2"
    heads             : int   = 4
    attention_dropout : float = 0.0


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
    attention_dropout : float = 0.0
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


class ModelConfigRegistry:
    @staticmethod
    def by_name() -> dict:
        configs = (EdgeConvConfig, GINEConfig, GATv2CascadeConfig, GINECascadeConfig, GATv2Config, GraphSAGEConfig, PNAConfig, ResGatedConfig, GPSCascadeConfig, GeneralConvConfig, GPSConfig, GPSLiteConfig, GCNConfig, TransformerConvConfig)
        return {config.name: config for config in configs}


@dataclass(frozen=True)
class ArchitectureSettings:
    activations      : tuple = (("gelu", "GELU"), ("relu", "ReLU"), ("elu", "ELU"), ("leaky_relu", "LeakyReLU"), ("tanh", "Tanh"))
    coordinate_count : int   = 3
    film_gain        : float = 0.1
    output_gain      : float = 0.01


@dataclass(frozen=True)
class ArchitectureSearchSpace:
    hidden_dims               : tuple = (64, 128, 192, 256)
    layers_low                : int   = 2
    layers_high               : int   = 6
    rate_low                  : float = 0.0
    rate_high                 : float = 0.2
    rate_step                 : float = 0.05
    hierarchical_feature_dims : tuple = (128, 256, 384)
    coord_embed_dims          : tuple = (32, 64, 128)
    heads                     : tuple = (2, 4, 8)
    pool_levels_low           : int   = 2
    pool_levels_high          : int   = 4
    sag_ratio_low             : float = 0.3
    sag_ratio_high            : float = 0.7
    sag_ratio_step            : float = 0.1
