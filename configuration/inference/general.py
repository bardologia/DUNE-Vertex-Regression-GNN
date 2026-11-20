from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorQuantiles:
    p90 : float = 0.90
    p95 : float = 0.95


@dataclass(frozen=True)
class PlotStyle:
    rc              : tuple = (("figure.dpi", 300), ("savefig.dpi", 300), ("font.size", 10), ("font.family", "sans-serif"), ("axes.grid", True), ("grid.alpha", 0.3), ("axes.spines.top", False), ("axes.spines.right", False))
    primary_color   : str   = "#2c7fb8"
    secondary_color : str   = "#31a354"
    accent_color    : str   = "#d95f0e"


@dataclass(frozen=True)
class AnalysisPlotStyle:
    square_figure     : tuple = (5, 5)
    panel_figure      : tuple = (5, 4)
    volume_figure     : tuple = (6, 5)
    point_size        : int   = 6
    point_alpha       : float = 0.4
    volume_point_size : int   = 8
    volume_alpha      : float = 0.7
    bins              : int   = 60
    bar_alpha         : float = 0.85
    line_width        : float = 1.2
    cumulative_width  : float = 1.5
    colorbar_shrink   : float = 0.6
    colormap          : str   = "viridis"


@dataclass(frozen=True)
class AnalysisAnimationStyle:
    maximum_points    : int   = 2000
    frames            : int   = 48
    frames_per_second : int   = 15
    frame_dpi         : int   = 100
    point_size        : int   = 8
    point_alpha       : float = 0.5
    error_point_size  : int   = 10
    legend_anchor     : tuple = (1.02, 1.0)
    elevation         : int   = 20
    full_turn         : int   = 360


@dataclass(frozen=True)
class DatasetExplorerSources:
    raw                : str   = "raw"
    augmented          : str   = "augmented"
    kinds              : tuple = ("raw", "augmented")
    unreflected_octant : str   = "ppp"
