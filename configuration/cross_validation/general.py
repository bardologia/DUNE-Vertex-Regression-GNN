from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossValidationConfig:
    n_folds             : int   = 5
    validation_fraction : float = 0.1
    stratified          : bool  = True
    shuffle             : bool  = True
    random_state        : int   = 42


@dataclass(frozen=True)
class CrossValidationReportFields:
    summary_metrics : tuple = ("euclidean_mean", "euclidean_median", "mae", "rmse", "median_error", "r2")
