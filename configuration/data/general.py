from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

from configuration.shared import IoNames


@dataclass
class DataConfig:
    parquet_store_dir  : Path  = field(default_factory=lambda: IoNames.project_root / IoNames.data_frames_dir / IoNames.parquet_dir)
    raw_input_dir      : Path  = field(default_factory=lambda: IoNames.project_root / IoNames.data_dir)
    build_store        : bool  = False
    store_worker_count : int   = 10

    augment_octants    : bool  = True
    subset_fraction    : float = 1.0
    stats_sample_size  : int   = 8192
    target_frame       : str   = "axiswise"

    coordinate_columns : tuple = ("event_x", "event_y", "event_z")
    position_columns   : tuple = ("x", "y", "z")

    detector_half_extent : tuple = (6.03, 6.67, 29.93)


@dataclass
class GraphConfig:
    k_neighbors        : int   = 4
    bidirectional      : bool  = False

    active_only        : bool  = True
    max_active_nodes   : int   = 0
    global_node        : bool  = True

    direction_features    : bool  = True
    inertia_features      : bool  = True
    rank_features         : bool  = True
    graph_scalar_features : bool  = True
    graph_moment_features : bool  = True
    graph_quantiles       : tuple = (0.1, 0.25, 0.5, 0.75, 0.9)
    edge_rbf_count        : int   = 16
    edge_rbf_max          : float = 7.0


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


@dataclass(frozen=True)
class ApexModuleGeometry:
    batch_size       : int   = 48
    half_slots       : int   = 12
    drift_pitch      : float = 0.505
    drift_edge       : float = 6.0275
    wall_side        : float = 6.673
    wall_first_z     : float = -29.75
    wall_ring_pitch  : float = 0.5
    wall_modules     : int   = 5760
    endcap_face      : float = 29.9265
    endcap_first_row : float = -5.8075
    endcap_row_pitch : float = 0.505
    endcap_modules   : int   = 1152
    wall_faces       : tuple = ("right_wall", "left_wall")
    endcap_faces     : tuple = ("front_endcap", "back_endcap")
    blocks           : tuple = ("wall", "endcap")


@dataclass(frozen=True)
class ChannelBins:
    raw_columns          : tuple = ("bin", "x", "y", "z")
    lead_bin             : int   = 12
    lead_module          : int   = 1
    wall_bins            : tuple = (15, 5771)
    wall_offset          : int   = 12
    endcap_bins          : tuple = (5774, 6924)
    endcap_offset        : int   = 5773
    face_total_bins      : tuple = (13, 5772, 6925)
    absent_bins          : tuple = (14, 5773)
    empty_wall_modules   : tuple = (0, 2)
    empty_endcap_modules : tuple = (0,)
    face_total_block     : str   = "face_total"


@dataclass(frozen=True)
class ChannelMappingChecks:
    channel_count      : int   = 6909
    position_decimals  : int   = 6
    position_tolerance : float = 1e-6
    listed_bins        : int   = 10


@dataclass(frozen=True)
class StoreLayout:
    geometry_columns    : tuple = ("channel_index", "raw_channel_index", "bin", "module_index", "face", "x", "y", "z")
    geometry_types      : tuple = ("int32", "int32", "int32", "int32", "string", "float32", "float32", "float32")
    octant_columns      : tuple = ("base_event_id", "octant", "sign_x", "sign_y", "sign_z", "target_x", "target_y", "target_z")
    octant_signs        : tuple = (1, -1)
    octant_symbols      : tuple = ((1, "p"), (-1, "n"))
    reflection_decimals : int   = 6


@dataclass(frozen=True)
class CoordinateAxes:
    names : tuple = ("x", "y", "z")


@dataclass(frozen=True)
class DatasetSplits:
    names : tuple = ("train", "val", "test")


@dataclass(frozen=True)
class GraphSettings:
    epsilon           : float = 1e-8
    minimum_nodes     : int   = 2
    probe_nodes       : int   = 8
    probe_light_floor : float = 0.1


@dataclass(frozen=True)
class LightRealizationSettings:
    epoch_seed_stride : int = 1_000_003


@dataclass(frozen=True)
class NormalizationSettings:
    epsilon              : float = 1e-8
    skewness_log         : float = 2.0
    outlier_ratio        : float = 3.0
    robust_iqr_factor    : float = 1.349
    quartiles            : tuple = (75, 25)
    groups               : tuple = ("node", "edge", "graph", "target")
    target_frames        : tuple = ("isotropic", "axiswise")
    node_spatial_group   : str   = "position"
    graph_spatial_groups : tuple = ("light_centroid", "sharp_centroid", "light_quantile")


@dataclass(frozen=True)
class StratificationSettings:
    maximum_bins    : int = 10
    targets_per_bin : int = 10
    percentile_span : int = 100
