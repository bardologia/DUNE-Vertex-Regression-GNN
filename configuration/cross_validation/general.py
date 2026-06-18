from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossValidationConfig:
    n_folds             : int   = 5
    validation_fraction : float = 0.1
    stratified          : bool  = True
    shuffle             : bool  = True
    random_state        : int   = 42
