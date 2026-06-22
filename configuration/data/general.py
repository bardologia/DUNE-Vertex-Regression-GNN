from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HotChannelConfig:
    enabled         : bool  = True
    active_fraction : float = 0.999
    median_factor   : float = 50.0
    neighbor_count  : int   = 8
    min_events      : int   = 100


@dataclass
class DataConfig:
    parquet_store_dir  : Path  = field(default_factory=lambda: PROJECT_ROOT / "data_frames" / "parquet")
    raw_input_dir      : Path  = field(default_factory=lambda: PROJECT_ROOT / "data")
    build_store        : bool  = False
    store_worker_count : int   = 10

    augment_octants    : bool  = False
    subset_fraction    : float = 1.0
    stats_sample_size  : int   = 8192

    coordinate_columns : tuple = ("event_x", "event_y", "event_z")
    position_columns   : tuple = ("x", "y", "z")

    hot_channels       : HotChannelConfig = field(default_factory=HotChannelConfig)


@dataclass
class GraphConfig:
    k_neighbors        : int   = 4
    bidirectional      : bool  = False

    active_only        : bool  = True
    max_active_nodes   : int   = 0

    direction_features : bool  = True
    inertia_features   : bool  = True
    rank_features      : bool  = True
    edge_rbf_count     : int   = 16
    edge_rbf_max       : float = 7.0


@dataclass
class PhysicsConfig:
    scale_factor         : float = 0.1
    detection_efficiency : float = 0.02
    efficiency_seed      : int   = 42


@dataclass
class SplitConfig:
    test_size    : float = 0.1
    val_ratio    : float = 0.1
    random_state : int   = 42


@dataclass
class AugmentationConfig:
    enabled : bool = False
    seed    : int  = 1234

    sensor_dropout_enabled     : bool  = True
    sensor_dropout_probability : float = 0.02

    spurious_activation_enabled     : bool  = True
    spurious_activation_probability : float = 0.003
    spurious_activation_max_light   : float = 0.5

    light_noise_enabled : bool  = True
    light_noise_sigma   : float = 0.03
    light_noise_mode    : str   = "multiplicative"

    photon_thinning_enabled  : bool  = True
    photon_thinning_survival : float = 0.97

    gain_jitter_enabled : bool  = True
    gain_jitter_sigma   : float = 0.02


@dataclass
class DatasetConfig:
    data         : DataConfig         = field(default_factory=DataConfig)
    graph        : GraphConfig        = field(default_factory=GraphConfig)
    physics      : PhysicsConfig      = field(default_factory=PhysicsConfig)
    split        : SplitConfig        = field(default_factory=SplitConfig)
    augmentation : AugmentationConfig = field(default_factory=AugmentationConfig)
