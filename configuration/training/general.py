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
    learning_rate_regression_head : float = 0.01
    learning_rate_pool            : float = 0.001
    learning_rate_encoder         : float = 0.001

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
    type    : str   = "cosine_annealing"
    epochs  : int   = 250
    eta_min : float = 1e-6


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

    containment_weight : float = 0.0
    containment_radius : float = 4.0

    light_falloff_weight  : float = 0.0
    light_falloff_epsilon : float = 1.0


@dataclass
class TrainingLoopConfig:
    epochs                      : int   = 100
    batch_size                  : int   = 15
    device                      : str   = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp                     : bool  = True
    gradient_accumulation_steps : int   = 1
    num_workers                 : int   = 4
    pin_memory                  : bool  = True
    persistent_workers          : bool  = True
    verbose                     : bool  = True
    profile                     : bool  = False
    tuning_mode                 : bool  = False


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
