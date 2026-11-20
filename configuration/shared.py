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

    best_model_file : str = "best_model.pt"


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
