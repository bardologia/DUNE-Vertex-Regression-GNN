from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path


@dataclass(frozen=True)
class IoNames:
    project_root : Path = Path(__file__).resolve().parents[1]

    data_dir           : str = "data"
    data_frames_dir    : str = "data_frames"
    parquet_dir        : str = "parquet"
    event_explorer_dir : str = "event_explorer"
    runs_dir           : str = "runs"
    tuning_dir         : str = "tuning"
    logs_dir           : str = "logs"

    default_log_name  : str = "experiment"
    log_suffix        : str = ".log"
    output_log_suffix : str = ".out"
    run_tag_format    : str = "%Y%m%d_%H%M%S"
    run_stamp_format  : str = "%Y%m%d-%H%M%S"
    json_indent       : int = 2

    run_dir                  : str = "{model}_{run}"
    tensorboard_dir          : str = "tensorboard"
    metadata_dir             : str = "metadata"
    checkpoints_dir          : str = "checkpoints"
    analysis_dir             : str = "analysis"
    best_model_file          : str = "best_model.pt"
    trainer_state_file       : str = "last.pt"
    resolved_config_file     : str = "resolved_config.json"
    split_base_ids_file      : str = "split_base_ids.json"
    metrics_file             : str = "metrics.json"
    normalization_stats_file : str = "normalization_stats.json"
    store_geometry_file      : str = "store_geometry.json"
    completion_file          : str = "complete.json"
    model_summary_file       : str = "model_summary.md"

    overfit_check_dir     : str = "overfit_check"
    overfit_report_file   : str = "overfit_report.json"
    pretrain_dir          : str = "pretrain"
    pretrain_log          : str = "pretrain"
    pretrain_report_file  : str = "pretrain_report.json"
    probe_dir             : str = "probe"
    probe_checkpoint_file : str = "probe.pt"
    trial_dir             : str = "trial_{number}"
    best_trial_file       : str = "best_trial.json"


@dataclass(frozen=True)
class LogSettings:
    level       : str   = "INFO"
    level_names : tuple = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    file_format         : str = "[%(asctime)s] [%(levelname)s] %(message)s"
    timestamp_format    : str = "%Y-%m-%d %H:%M:%S"
    console_time_format : str = "[%H:%M:%S]"
    file_rule_character : str = "="
    file_rule_width     : int = 100

    console_color_system        : str   = "truecolor"
    console_theme               : tuple = (("section", "bold cyan"), ("subsection", "white"), ("key", "bold magenta"), ("value", "bright_white"), ("ok", "bold green"), ("warn", "bold yellow"), ("err", "bold red"), ("muted", "white"), ("logging.level.debug", "white"), ("logging.level.info", "white"), ("logging.level.warning", "bold yellow"), ("logging.level.error", "bold red"), ("logging.level.critical", "bold red"))
    accent_style                : str   = "cyan"
    heading_style               : str   = "bold cyan"
    subsection_style            : str   = "bold white"
    elapsed_style               : str   = "dim"
    progress_description_format : str   = "[progress.description]{task.description}"
    live_refresh_per_second     : int   = 4
    progress_refresh_per_second : int   = 8

    terminal_columns  : int = 120
    terminal_lines    : str = "32"
    job_stream_buffer : int = 4000
    job_stream_queue  : int = 8000
