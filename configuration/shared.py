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

    best_model_file : str = "best_model.pt"
