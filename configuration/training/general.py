from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch


@dataclass
class IOConfig:
    log_base_dir   : Path = Path("runs")
    tuner_base_dir : Path = Path("tuning")
    checkpoint_name: str  = "best_model.pt"


@dataclass
class OptimizerConfig:
    learning_rate_regression_head : float = 0.005
    learning_rate_pool            : float = 0.0005
    learning_rate_encoder         : float = 0.0005

    reference_batch_size : int = 128

    weight_decay_regression_head : float = 1e-5
    weight_decay_pool            : float = 1e-5
    weight_decay_encoder         : float = 1e-4

    betas : tuple = (0.9, 0.999)
    eps   : float = 1e-8

    @classmethod
    def tunable_params(cls) -> dict:
        return {
            "learning_rate_regression_head" : {"type": "float", "low": 5e-5, "high": 2e-3, "log": True},
            "learning_rate_pool"            : {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            "learning_rate_encoder"         : {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
            "weight_decay_regression_head"  : {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
            "weight_decay_pool"             : {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
            "weight_decay_encoder"          : {"type": "float", "low": 1e-6, "high": 5e-4, "log": True},
        }


@dataclass
class SchedulerConfig:
    type    : str   = "reduce_on_plateau"
    eta_min : float = 1e-6

    power     : float = 2.0
    step_size : int   = 30
    gamma     : float = 0.1

    mode      : str   = "min"
    factor    : float = 0.5
    patience  : int   = 5
    threshold : float = 1e-4
    cooldown  : int   = 0
    min_lr    : float = 1e-6


@dataclass
class WarmupConfig:
    warmup_enabled      : bool  = True
    warmup_steps        : int   = 100
    warmup_start_factor : float = 0.1
    warmup_mode         : str   = "linear"
    warmup_poly_power   : float = 2.0


@dataclass
class EarlyStoppingConfig:
    patience     : int   = 15
    min_delta    : float = 0.0
    restore_best : bool  = True


@dataclass
class GradientClipperConfig:
    clip_mode           : str   = "fixed"
    max_grad_norm       : float = 1.0
    adaptive_window     : int   = 100
    adaptive_percentile : float = 75.0
    adaptive_mean_std_k : float = 2.0
    clip_epsilon        : float = 1e-6
    log_histogram_freq  : int   = 100


@dataclass
class LossConfig:
    data_term   : str   = "mse"
    huber_delta : float = 1.0
    data_weight : float = 1.0

    euclidean_weight : float = 0.0

    light_falloff_weight  : float = 0.0
    light_falloff_epsilon : float = 1.0


@dataclass
class TrainingLoopConfig:
    epochs                      : int   = 100
    batch_size                  : int   = 15
    device                      : str   = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp                     : bool  = True
    gradient_accumulation_steps : int   = 1
    validation_frequency        : int   = 1

    num_workers                 : int   = 4
    prefetch_factor             : int   = 2
    pin_memory                  : bool  = True
    persistent_workers          : bool  = True

    use_ema                     : bool  = True
    ema_decay                   : float = 0.999
    resume                      : bool  = False
    abort_on_nonfinite_loss     : bool  = True
    log_debug                   : bool  = False
    tuning_mode                 : bool  = False


@dataclass
class MemoryConfig:
    clear_cache_every_n_steps : int  = 0
    clear_cache_after_eval    : bool = False
    clear_cache_after_epoch   : bool = False

    reserve_vram      : bool  = False
    vram_keep_free_gb : float = 1.0


@dataclass
class ResourceConfig:
    enabled            : bool  = True
    poll_interval_sec  : float = 5.0
    log_to_tensorboard : bool  = True
    warn_ram_pct       : float = 90.0
    warn_vram_pct      : float = 90.0
    warn_swap_pct      : float = 50.0
    warn_shm_pct       : float = 80.0
    warn_cooldown_sec  : float = 30.0


@dataclass
class OverfitCheckConfig:
    enabled         : bool  = False
    n_examples      : int   = 2
    max_steps       : int   = 300
    steps_per_epoch : int   = 25
    pass_loss_ratio : float = 0.05
    stop_threshold  : float = 1e-6


@dataclass
class PretrainConfig:
    find_batch_size : bool = False
    tune_loader     : bool = False

    vram_budget_gb : float = 3.5
    max_batch      : int   = 512
    measure_steps  : int   = 8

    worker_counts    : tuple = (0, 2, 4, 6, 8)
    prefetch_factors : tuple = (2, 4, 8)
    warmup_batches   : int   = 8
    timed_batches    : int   = 60
    data_wait_target : float = 0.05

    seed : int = 42


@dataclass
class TrainingConfig:
    io               : IOConfig              = field(default_factory=IOConfig)
    optimizer        : OptimizerConfig       = field(default_factory=OptimizerConfig)
    scheduler        : SchedulerConfig       = field(default_factory=SchedulerConfig)
    warmup           : WarmupConfig          = field(default_factory=WarmupConfig)
    early_stopping   : EarlyStoppingConfig   = field(default_factory=EarlyStoppingConfig)
    gradient_clipper : GradientClipperConfig = field(default_factory=GradientClipperConfig)
    loss             : LossConfig            = field(default_factory=LossConfig)
    loop             : TrainingLoopConfig    = field(default_factory=TrainingLoopConfig)
    memory           : MemoryConfig          = field(default_factory=MemoryConfig)
    resources        : ResourceConfig        = field(default_factory=ResourceConfig)
    overfit_check    : OverfitCheckConfig    = field(default_factory=OverfitCheckConfig)
    pretrain         : PretrainConfig        = field(default_factory=PretrainConfig)
