from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TuningConfig:
    epochs        : int  = 15
    n_trials      : int  = 5
    warmup_trials : int  = 5
    warmup_steps  : int  = 3
    batch_size    : int  = 128
    random_state  : int  = 42
    final_epochs  : int  = 300
