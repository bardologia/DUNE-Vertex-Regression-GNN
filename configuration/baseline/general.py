from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.shared import IoNames


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
    log_base_dir : Path = field(default_factory=lambda: IoNames.project_root / IoNames.runs_dir)


@dataclass
class BaselineConfig:
    features : BaselineFeatureConfig = field(default_factory=BaselineFeatureConfig)
    search   : BaselineSearchConfig  = field(default_factory=BaselineSearchConfig)
    booster  : BaselineBoosterConfig = field(default_factory=BaselineBoosterConfig)
    io       : BaselineIOConfig      = field(default_factory=BaselineIOConfig)


@dataclass(frozen=True)
class BaselineLabels:
    model_name     : str = "baseline_{family}"
    lightgbm_key   : str = "lightgbm"
    lightgbm_label : str = "LightGBM"
    xgboost_key    : str = "xgboost"
    xgboost_label  : str = "XGBoost"


@dataclass(frozen=True)
class LightGbmSearchSpace:
    learning_rate     : tuple = (0.01, 0.3)
    num_leaves        : tuple = (16, 512)
    min_child_samples : tuple = (5, 200)
    subsample         : tuple = (0.5, 1.0)
    colsample_bytree  : tuple = (0.5, 1.0)
    reg_alpha         : tuple = (1e-8, 10.0)
    reg_lambda        : tuple = (1e-8, 10.0)


@dataclass(frozen=True)
class XgBoostSearchSpace:
    learning_rate    : tuple = (0.01, 0.3)
    max_depth        : tuple = (4, 12)
    min_child_weight : tuple = (0.5, 64.0)
    subsample        : tuple = (0.5, 1.0)
    colsample_bytree : tuple = (0.5, 1.0)
    reg_alpha        : tuple = (1e-8, 10.0)
    reg_lambda       : tuple = (1e-8, 10.0)
    gamma            : tuple = (1e-8, 1.0)


@dataclass(frozen=True)
class BaselineImportance:
    gain_epsilon : float = 1e-12
    top_count    : int   = 25
    figure       : tuple = (7, 8)
    bar_color    : str   = "#4C72B0"
