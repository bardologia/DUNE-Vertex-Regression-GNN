from dataclasses import dataclass, field
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torch


@dataclass
class IOConfig:
    logdir: str = f"runs/run_{datetime.now():%Y%m%d-%H%M%S}"
    writer = None

    def __post_init__(self):
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.writer.flush()


@dataclass
class DataConfig:
    input_dir             : Path = field(default_factory=lambda: Path("./data"))
    max_files             : int = 10000000
    coordinate_columns    : tuple = ("event_x", "event_y", "event_z")
    feature_columns       : tuple = ("x", "y", "z", "light_amount")
    position_columns      : tuple = ("x", "y", "z")
    light_column          : str = "light_amount"
    header_rows_to_skip   : int = 0
    coordinate_line_index : int = 2


@dataclass
class GraphConfig:
    k_neighbors   : int = 4
    knn_algorithm : str = "kd_tree"
    bidirectional : bool = True


@dataclass
class ModelConfig:
    input_dim       : int = 4
    output_dim      : int = 3
    gcn_hidden_dims : tuple = (192, 384, 256)
    
    regression_hidden_dims : tuple = (512, 256, 128)
    regression_output_dim  : int = 3
    regression_dropout     : float = 0.1
    
    gat_heads          : int = 6
    gat_use_batch_norm : bool = True
    gat_dropout        : float = 0.1
    
    use_batch_norm     : bool = True
    attention_heads    : int = 6
    edge_dim           : int = 1
    sag_ratio          : float = 0.5
    sag_layers         : int = 2


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
    batch_size                  : int = 4
    epochs                      : int = 250
    device                      : str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir                     : str = f"./runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer                      : any = None
    deep_log                    : bool = False
    use_amp                     : bool = True
    gradient_accumulation_steps : int = 4 
    pin_memory                  : bool = False 
    num_workers                 : int = 0
    validation_frequency        : int = 5
    max_grad_norm               : float = 1.0
    verbose                     : bool = True


@dataclass
class EarlyStoppingConfig:
    patience     : int = 15
    min_delta    : float = 0.0
    restore_best : bool = True


@dataclass
class SchedulerConfig:
    epochs   : int = 250
    eta_min  : float = 1e-6
    factor   : float = 0.5
    patience : int = 8
    mode     : str = "min"


@dataclass
class SplitConfig:
    test_size    : float = 0.1
    val_ratio    : float = 0.1
    random_state : int = 42


@dataclass
class OptimizerConfig:
    lr_regression_head : float = 0.001
    lr_pool            : float = 0.0005
    lr_gcn             : float = 0.0003
    
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
    efficiency_seed      : int = 42


@dataclass
class OverfitConfig:
    enabled: bool = False


@dataclass
class TuningConfig:
    epochs: int = 50


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
