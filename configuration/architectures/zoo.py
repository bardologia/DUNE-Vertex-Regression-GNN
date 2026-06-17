from __future__ import annotations

from dataclasses import dataclass, field


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
    use_edge_features : bool  = True
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
    name        : str   = "pna"
    aggregators : tuple = ("mean", "min", "max", "std")
    scalers     : tuple = ("identity", "amplification", "attenuation")
    towers      : int   = 1
    max_degree  : int   = 8
    pre_layers  : int   = 1
    post_layers : int   = 1


@dataclass
class GCNConfig(BaseGNNConfig):
    name              : str  = "gcn"
    use_edge_features : bool = True
    improved          : bool = False
    add_self_loops    : bool = True


@dataclass
class TransformerConvConfig(BaseGNNConfig):
    name              : str   = "transformer"
    heads             : int   = 4
    attention_dropout : float = 0.1
    beta              : bool  = True


@dataclass
class EdgeConvConfig(BaseGNNConfig):
    name              : str  = "edgeconv"
    use_edge_features : bool = False
    aggregator        : str  = "max"
    edge_mlp_hidden   : int  = 128


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
    use_edge_features : bool  = False
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
