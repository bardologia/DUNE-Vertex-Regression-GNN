from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch

from configuration.cross_validation.general import CrossValidationConfig
from configuration.data.general             import DatasetConfig
from configuration.training.general         import TrainingConfig
from configuration.tuning.general           import TuningConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainEntryConfig:
    model_name      : str            = "gps"
    model_overrides : dict           = field(default_factory=dict)
    seed            : int            = 42
    gpu             : int            = 0
    run_name        : str            = ""
    tensorboard     : bool           = False
    infer_after     : bool           = False
    dataset         : DatasetConfig  = field(default_factory=DatasetConfig)
    training        : TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class InferenceEntryConfig:
    run_directory : Path  = field(default_factory=lambda: PROJECT_ROOT / "runs")
    gpu           : int   = 0
    device        : str   = "cuda" if torch.cuda.is_available() else "cpu"
    splits        : tuple = ("test", "val", "train")
    batch_size    : int   = 16


@dataclass
class TuneEntryConfig:
    model_name : str            = "gps"
    seed       : int            = 42
    gpu        : int            = 0
    dataset    : DatasetConfig  = field(default_factory=DatasetConfig)
    tuning     : TuningConfig   = field(default_factory=TuningConfig)
    training   : TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class CrossValidationEntryConfig:
    model_name       : str                   = "gps"
    model_overrides  : dict                  = field(default_factory=dict)
    seed             : int                   = 42
    gpu              : int                   = 0
    run_name         : str                   = ""
    dataset          : DatasetConfig         = field(default_factory=DatasetConfig)
    training         : TrainingConfig        = field(default_factory=TrainingConfig)
    cross_validation : CrossValidationConfig = field(default_factory=CrossValidationConfig)
