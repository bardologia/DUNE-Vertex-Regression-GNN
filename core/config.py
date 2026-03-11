from dataclasses import dataclass, field
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torch


@dataclass
class IOConfig:
    logdir   : str = f"runs/run_{datetime.now():%Y%m%d-%H%M%S}"
    tunerdir : str = f"tuning/tune_{datetime.now():%Y%m%d-%H%M%S}"
    writer = None

    def __post_init__(self):
        self.writer = SummaryWriter(log_dir=Path(self.logdir) / "tensorboard")
        self.writer.flush()


@dataclass
class DataConfig:
    input_dir             : Path = field(default_factory=lambda: Path("./data"))
    max_files             : int = 10000
    coordinate_columns    : tuple = ("event_x", "event_y", "event_z")
    feature_columns       : tuple = ("x", "y", "z", "light_amount", "hit_flag", "light_fraction", "dist_to_centroid", "local_light_density")
    position_columns      : tuple = ("x", "y", "z")
    light_column          : str = "light_amount"
    header_rows_to_skip   : int = 0
    coordinate_line_index : int = 2


@dataclass
class GraphConfig:
    k_neighbors   : int = 4
    knn_algorithm : str = "kd_tree"
    bidirectional : bool = False


@dataclass
class ModelConfig:
    input_dim       : int = 8   # x, y, z, light, hit_flag, light_fraction, dist_to_centroid, local_light_density
    output_dim      : int = 3
    
    gps_hidden_dim  : int   = 128
    gps_num_layers  : int   = 4
    gps_heads       : int   = 4
    gps_dropout     : float = 0.1
    gps_attn_dropout: float = 0.1
    gps_ffn_ratio   : float = 4.0
    drop_path_rate  : float = 0.1
    
    edge_dim        : int = 7   # dist, dx, dy, dz, inv_sq_dist, light_gradient, light_similarity
    
    pool_num_levels : int   = 3
    sag_ratio       : float = 0.5
    
    hierarchical_feature_dim : int   = 256
    coord_embed_dim          : int   = 64
    regression_hidden_dims   : tuple = (128, 64)
    regression_output_dim    : int   = 3
    regression_dropout       : float = 0.1


@dataclass
class EMAConfig:
    use_ema   : bool = False
    ema_decay : float = 0.9995


@dataclass
class WarmupConfig:
    warmup_enabled      : bool = True
    warmup_steps        : int = 100
    warmup_start_factor : float = 0.1


@dataclass
class TrainingConfig:
    lr                          : float = 0.001
    weight_decay                : float = 1e-5
    batch_size                  : int   = 4
    epochs                      : int   = 10
    device                      : str   = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp                     : bool  = True
    gradient_accumulation_steps : int   = 1
    pin_memory                  : bool  = False 
    num_workers                 : int   = 0
    validation_frequency        : int   = 5
    max_grad_norm               : float = 1.0
    verbose                     : bool  = True
    profile                     : bool  = True


@dataclass
class EarlyStoppingConfig:
    patience     : int = 15
    min_delta    : float = 0.0
    restore_best : bool = True


@dataclass
class SchedulerConfig:
    epochs   : int   = 250
    eta_min  : float = 1e-6
    factor   : float = 0.5
    patience : int   = 8
    mode     : str   = "min"


@dataclass
class SplitConfig:
    test_size    : float = 0.1
    val_ratio    : float = 0.1
    random_state : int = 42


@dataclass
class OptimizerConfig:
    lr_regression_head : float = 0.01
    lr_pool            : float = 0.001
    lr_gcn             : float = 0.001
    
    weight_decay_regression_head : float = 1e-5
    weight_decay_pool            : float = 1e-5
    weight_decay_gcn             : float = 1e-4
    
    betas : tuple = (0.9, 0.999)
    eps   : float = 1e-8


@dataclass
class PreprocessingConfig:
    zscore_threshold     : float = 9.0
    radius               : float = 1.0
    scale_factor         : float = 0.1
    detection_efficiency : float = 0.02
    efficiency_seed      : int   = 42


@dataclass
class OverfitConfig:
    enabled: bool = True


@dataclass
class TuningConfig:
    epochs        : int = 1
    n_trials      : int = 5
    warmup_trials : int = 1
    warmup_steps  : int = 1


@dataclass
class GlobalConfig:
    io             : IOConfig            = field(default_factory=IOConfig)
    data           : DataConfig          = field(default_factory=DataConfig)
    graph          : GraphConfig         = field(default_factory=GraphConfig)
    model          : ModelConfig         = field(default_factory=ModelConfig)
    training       : TrainingConfig      = field(default_factory=TrainingConfig)
    optimizer      : OptimizerConfig     = field(default_factory=OptimizerConfig)
    ema            : EMAConfig           = field(default_factory=EMAConfig)
    warmup         : WarmupConfig        = field(default_factory=WarmupConfig)
    early_stopping : EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    scheduler      : SchedulerConfig     = field(default_factory=SchedulerConfig)
    split          : SplitConfig         = field(default_factory=SplitConfig)
    preprocessing  : PreprocessingConfig = field(default_factory=PreprocessingConfig)
    overfit        : OverfitConfig       = field(default_factory=OverfitConfig)
    tuning         : TuningConfig        = field(default_factory=TuningConfig)


config = GlobalConfig()

    
__all__ = [
    "IOConfig",
    "DataConfig",
    "GraphConfig",
    "ModelConfig",
    "EMAConfig",
    "WarmupConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "EarlyStoppingConfig",
    "SchedulerConfig",
    "SplitConfig",
    "PreprocessingConfig",
    "OverfitConfig",
    "TuningConfig",
    "GlobalConfig",
    "config",
]
