from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch

from configuration.shared import IoNames


@dataclass
class AxisSweepConfig:
    minimum : float = 0.0
    maximum : float = 1.0
    step    : float = 0.1


@dataclass
class EnergyEfficiencySweepConfig:
    run_directory : Path            = field(default_factory=lambda: IoNames.project_root / IoNames.runs_dir)
    device        : str             = "cuda" if torch.cuda.is_available() else "cpu"
    split         : str             = "test"
    batch_size    : int             = 16
    run_name      : str             = ""
    annotate      : bool            = True

    scale         : AxisSweepConfig = field(default_factory=lambda: AxisSweepConfig(minimum=0.05, maximum=0.50, step=0.05))
    efficiency    : AxisSweepConfig = field(default_factory=lambda: AxisSweepConfig(minimum=0.01, maximum=0.10, step=0.01))


@dataclass(frozen=True)
class SweepMetrics:
    grid_keys   : tuple = ("euclidean_mean", "euclidean_median", "mae", "rmse", "r2")
    csv_columns : tuple = ("scale_factor", "detection_efficiency", "count", "mae", "rmse", "r2", "median_error", "bias", "euclidean_mean", "euclidean_median", "euclidean_p90", "euclidean_p95")


@dataclass(frozen=True)
class SweepGridSettings:
    rounding_decimals : int = 10


@dataclass(frozen=True)
class SweepPlotStyle:
    heatmaps                     : tuple = (("euclidean_mean", "Mean euclidean error [m]", "viridis_r"), ("euclidean_median", "Median euclidean error [m]", "viridis_r"), ("mae", "Mean absolute error [m]", "viridis_r"), ("rmse", "Root-mean-square error [m]", "viridis_r"), ("r2", "Coefficient of determination", "viridis"))
    scale_label                  : str   = "Light scale factor"
    efficiency_label             : str   = "Detection efficiency"
    heatmap_figure               : tuple = (6, 5)
    line_figure                  : tuple = (6, 4.5)
    tick_rotation                : int   = 45
    annotation_size              : int   = 6
    light_text_color             : str   = "white"
    dark_text_color              : str   = "black"
    baseline_marker_size         : int   = 160
    baseline_line_width          : float = 1.4
    legend_font_size             : int   = 7
    colorbar_shrink              : float = 0.85
    colormap_range               : tuple = (0.15, 0.9)
    lines_vs_scale_colormap      : str   = "viridis"
    lines_vs_efficiency_colormap : str   = "plasma"
    marker_size                  : int   = 3
    line_width                   : float = 1.2
