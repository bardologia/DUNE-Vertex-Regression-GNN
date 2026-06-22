from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FeatureImportanceConfig:
    run_directory       : Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
    device              : str  = "cuda" if torch.cuda.is_available() else "cpu"
    split               : str  = "test"
    batch_size          : int  = 16
    run_name            : str  = ""
    max_events          : int  = 0

    permutation         : bool = True
    occlusion           : bool = True
    gradient            : bool = True

    permutation_repeats : int  = 10
    permutation_seed    : int  = 0
    grouped             : bool = True

    primary_metric      : str  = "euclidean_mean"
    top_k               : int  = 15
