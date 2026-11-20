from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path

from configuration.shared import MeasurementUnits


@dataclass(frozen=True)
class CliSettings:
    supported_types  : tuple = (bool, int, float, str, Path, list, tuple, dict)
    bootstrap_flags  : tuple = ("-h", "--help", "--help-config", "--detach", "--nohup", "--gpu")
    help_config_flag : str   = "--help-config"
    detach_flags     : tuple = ("--detach", "--nohup")
    detach_env_flag  : str   = "DUNE_GNN_DETACHED"
    unbuffered_flag  : str   = "-u"
    true_words       : tuple = ("true", "1", "yes", "on")
    false_words      : tuple = ("false", "0", "no", "off")


@dataclass(frozen=True)
class ReproducibilitySettings:
    seed_modulus : int = 2 ** 32


@dataclass(frozen=True)
class MarkdownSettings:
    empty_cell     : str   = "n/a"
    min_rule_width : int   = 3
    precision      : int   = 5
    adaptive_upper : float = 1e4
    adaptive_lower : float = 1e-3


@dataclass(frozen=True)
class LightCentroidSettings:
    axes         : tuple = ("x", "y", "z")
    face_margin  : float = 0.5
    weight_floor : float = 1e-12


@dataclass(frozen=True)
class LoaderTuningSettings:
    sample_interval_s  : float = 0.05
    join_intervals     : int   = 4
    join_floor_s       : float = 1.0
    learning_rate      : float = 1e-4
    time_floor_s       : float = 1e-9
    reference_prefetch : int   = 4
    fallback_cores     : int   = 8
    wait_threshold     : float = 0.05


@dataclass(frozen=True)
class LoaderSweepPlotStyle:
    figure_size     : tuple = (7.0, 4.5)
    dpi             : int   = 300
    marker          : str   = "o"
    legend_anchor   : tuple = (1.02, 1.0)
    percent         : float = 100.0
    threshold_color : str   = "black"
    threshold_style : str   = "--"
    threshold_width : float = 1.0
    scatter_size    : int   = 55
    colormap        : str   = "viridis"


@dataclass(frozen=True)
class GradientSettings:
    exploding_norm : float = 100.0


@dataclass(frozen=True)
class SchedulerSettings:
    learning_rate_floor : float = 1e-12
    decay_ratio_floor   : float = 1e-8


@dataclass(frozen=True)
class VramReservationSettings:
    granularity_bytes : int = 2 * MeasurementUnits.bytes_per_mib
    chunk_floor_bytes : int = 16 * MeasurementUnits.bytes_per_mib
