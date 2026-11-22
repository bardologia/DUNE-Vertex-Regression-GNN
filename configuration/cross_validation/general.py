from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossValidationConfig:
    n_folds      : int   = 6
    val_share    : float = 0.5
    random_state : int   = 42


@dataclass(frozen=True)
class CrossValidationReportFields:
    summary_metrics : tuple = ("rms_3d", "mae", "rmse", "median_error", "r2")
