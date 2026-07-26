from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BaselineFeatureConfig:
    top_channel_count : int   = 5
    quantiles         : tuple = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass
class BaselineSearchConfig:
    n_trials       : int = 40
    startup_trials : int = 12
    random_state   : int = 42


@dataclass
class BaselineBoosterConfig:
    max_estimators        : int = 3000
    early_stopping_rounds : int = 100
    thread_count          : int = 10
    device                : str = "cpu"


@dataclass
class BaselineIOConfig:
    log_base_dir : Path = field(default_factory=lambda: PROJECT_ROOT / "runs")


@dataclass
class BaselineConfig:
    features : BaselineFeatureConfig = field(default_factory=BaselineFeatureConfig)
    search   : BaselineSearchConfig  = field(default_factory=BaselineSearchConfig)
    booster  : BaselineBoosterConfig = field(default_factory=BaselineBoosterConfig)
    io       : BaselineIOConfig      = field(default_factory=BaselineIOConfig)
