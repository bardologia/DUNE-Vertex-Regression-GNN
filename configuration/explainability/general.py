from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

import torch

from configuration.shared import IoNames


@dataclass
class FeatureImportanceConfig:
    run_directory       : Path = field(default_factory=lambda: IoNames.project_root / IoNames.runs_dir)
    device              : str  = "cuda" if torch.cuda.is_available() else "cpu"
    split               : str  = "test"
    batch_size          : int  = 16
    run_name            : str  = ""
    max_events          : int  = 0

    permutation         : bool = True
    occlusion           : bool = True
    gradient            : bool = True
    expected_gradients  : bool = True
    shap                : bool = True

    permutation_repeats : int  = 10
    permutation_seed    : int  = 0
    grouped             : bool = True

    eg_samples          : int  = 32
    eg_events           : int  = 64
    eg_seed             : int  = 0

    shap_nsamples       : int  = 256
    shap_events         : int  = 16
    shap_seed           : int  = 0

    primary_metric      : str  = "rms_3d"
    top_k               : int  = 15


@dataclass(frozen=True)
class ImportanceMetrics:
    keys        : tuple = ("rms_3d", "mae", "rmse", "r2")
    descriptors : tuple = (("rms_3d", ("m", False)), ("mae", ("m", False)), ("rmse", ("m", False)), ("r2", ("", True)))


@dataclass(frozen=True)
class PermutationSeeding:
    domain_offsets : tuple = (("node", 0), ("edge", 100_000))
    column_stride  : int   = 131
    repeat_stride  : int   = 7919


@dataclass(frozen=True)
class KernelShapSettings:
    chunk_size : int = 64


@dataclass(frozen=True)
class ImportancePlotStyle:
    figure_width   : float = 7.0
    minimum_height : float = 2.6
    row_height     : float = 0.34
    height_padding : float = 1.0
    bar_height     : float = 0.74
    line_width     : float = 0.8
    line_color     : str   = "#525252"


@dataclass(frozen=True)
class ImportanceReportColumns:
    csv : tuple = (
        "feature", "group",
        "permutation_delta", "permutation_std", "permutation_pct_increase",
        "permutation_delta_mae", "permutation_delta_rmse", "permutation_delta_r2",
        "occlusion_delta", "occlusion_delta_mae", "occlusion_delta_rmse", "occlusion_delta_r2",
        "gradient_saliency", "gradient_input_times_grad",
        "expected_gradient", "expected_gradient_x", "expected_gradient_y", "expected_gradient_z",
        "kernel_shap", "kernel_shap_x", "kernel_shap_y", "kernel_shap_z",
        "consensus_mean_rank",
    )
