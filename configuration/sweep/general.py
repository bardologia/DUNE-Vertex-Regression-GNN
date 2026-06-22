from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AxisSweepConfig:
    minimum : float = 0.0
    maximum : float = 1.0
    step    : float = 0.1


@dataclass
class EnergyEfficiencySweepConfig:
    run_directory : Path            = field(default_factory=lambda: PROJECT_ROOT / "runs")
    device        : str             = "cuda" if torch.cuda.is_available() else "cpu"
    split         : str             = "test"
    batch_size    : int             = 16
    run_name      : str             = ""
    annotate      : bool            = True

    scale         : AxisSweepConfig = field(default_factory=lambda: AxisSweepConfig(minimum=0.05, maximum=0.50, step=0.05))
    efficiency    : AxisSweepConfig = field(default_factory=lambda: AxisSweepConfig(minimum=0.01, maximum=0.10, step=0.01))
