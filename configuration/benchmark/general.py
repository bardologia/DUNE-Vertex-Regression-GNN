from __future__ import annotations

from dataclasses import dataclass, field

from configuration.data.general     import DatasetConfig
from configuration.training.general import TrainingConfig


@dataclass
class SizeMatchConfig:
    enabled          : bool  = True
    reference_model  : str   = "gps"
    scaled_attribute : str   = "hidden_dim"
    tolerance        : float = 0.05
    max_iterations   : int   = 24
    minimum_width    : int   = 8
    maximum_width    : int   = 1024


@dataclass
class OverfitGateConfig:
    enabled             : bool  = True
    sample_count        : int   = 16
    epochs              : int   = 200
    batch_size          : int   = 8
    stop_threshold      : float = 0.05
    require_convergence : bool  = True
    abort_on_fail       : bool  = True


@dataclass
class MaxBatchConfig:
    enabled        : bool  = True
    vram_budget_gb : float = 3.0
    maximum_batch  : int   = 512
    measure_steps  : int   = 3
    sample_count   : int   = 256


@dataclass
class BenchmarkTrainingConfig:
    epochs               : int  = 50
    use_measured_batch   : bool = True


@dataclass
class ComparisonConfig:
    rank_metric : str = "euclidean_mean"


@dataclass
class BenchmarkConfig:
    seed        : int        = 42
    run_name    : str        = ""
    resume      : bool       = True
    skip_models : tuple      = ()

    size_match         : SizeMatchConfig        = field(default_factory=SizeMatchConfig)
    overfit            : OverfitGateConfig      = field(default_factory=OverfitGateConfig)
    max_batch          : MaxBatchConfig         = field(default_factory=MaxBatchConfig)
    benchmark_training : BenchmarkTrainingConfig = field(default_factory=BenchmarkTrainingConfig)
    comparison         : ComparisonConfig       = field(default_factory=ComparisonConfig)

    dataset  : DatasetConfig  = field(default_factory=DatasetConfig)
    training : TrainingConfig = field(default_factory=TrainingConfig)
