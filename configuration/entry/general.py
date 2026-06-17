from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch

from configuration.data.general     import DatasetConfig, SplitConfig
from configuration.training.general import TrainingConfig
from configuration.tuning.general   import TuningConfig


PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIRECTORY = PROJECT_ROOT / "datasets" / "E_10.0MeV_K_4"


@dataclass
class DatasetEntryConfig:
    dataset            : DatasetConfig = field(default_factory=DatasetConfig)
    output_base_dir    : Path          = field(default_factory=lambda: PROJECT_ROOT / "datasets")
    profile            : bool          = False


@dataclass
class TrainEntryConfig:
    model_name      : str            = "gps"
    model_overrides : dict           = field(default_factory=dict)
    seed            : int            = 42
    gpu             : int            = 0
    run_name        : str            = ""
    dataset_dir     : Path           = field(default_factory=lambda: DEFAULT_DATASET_DIRECTORY)
    subset_fraction : float          = 1.0
    tensorboard     : bool           = False
    infer_after     : bool           = False
    split           : SplitConfig    = field(default_factory=SplitConfig)
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
    model_name      : str            = "gps"
    seed            : int            = 42
    gpu             : int            = 0
    dataset_dir     : Path           = field(default_factory=lambda: DEFAULT_DATASET_DIRECTORY)
    subset_fraction : float          = 1.0
    tuning          : TuningConfig   = field(default_factory=TuningConfig)
    split           : SplitConfig    = field(default_factory=SplitConfig)
    training        : TrainingConfig = field(default_factory=TrainingConfig)
