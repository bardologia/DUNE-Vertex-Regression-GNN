from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    source                : str   = "parquet"
    input_dir             : Path  = field(default_factory=lambda: PROJECT_ROOT / "data")
    parquet_store_dir     : Path  = field(default_factory=lambda: PROJECT_ROOT / "data_frames" / "parquet")
    augment_octants       : bool  = False
    max_files             : int   = 500000
    subset_fraction       : float = 1.0
    coordinate_columns    : tuple = ("event_x", "event_y", "event_z")
    feature_columns       : tuple = ("x", "y", "z")
    position_columns      : tuple = ("x", "y", "z")
    light_column          : str   = "light_amount"
    header_rows_to_skip   : int   = 0
    coordinate_line_index : int   = 2


@dataclass
class GraphConfig:
    k_neighbors   : int  = 4
    knn_algorithm : str  = "kd_tree"
    bidirectional : bool = False


@dataclass
class PreprocessingConfig:
    zscore_threshold     : float = 9.0
    radius               : float = 1.0
    scale_factor         : float = 0.1
    detection_efficiency : float = 0.02
    efficiency_seed      : int   = 42


@dataclass
class SplitConfig:
    test_size    : float = 0.1
    val_ratio    : float = 0.1
    random_state : int   = 42


@dataclass
class RayConfig:
    num_cpus                           : Optional[int] = 6
    batch_tasks_per_cpu                : int           = 8
    max_inflight_tasks_per_cpu         : int           = 1
    max_batch_size                     : int           = 8
    scaling_batch_tasks_per_cpu        : int           = 4
    scaling_max_inflight_tasks_per_cpu : int           = 2
    scaling_max_batch_size             : int           = 64


@dataclass
class DatasetConfig:
    data          : DataConfig          = field(default_factory=DataConfig)
    graph         : GraphConfig         = field(default_factory=GraphConfig)
    preprocessing : PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split         : SplitConfig         = field(default_factory=SplitConfig)
    ray           : RayConfig           = field(default_factory=RayConfig)
