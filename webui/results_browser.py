from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

from project_paths import ProjectPaths
from server_logger import ServerLogger


class ResultsBrowser:

    SKIPPED_DIRECTORIES = {"__pycache__", ".git", ".ipynb_checkpoints"}
    MAX_DEPTH           = 6
    BEST_METRIC_SPLITS  = ("test", "val", "train")
    BEST_METRIC_NAME    = "euclidean_mean"
    COORDINATE_NAMES    = ("x", "y", "z")
    ERROR_METRIC_NAMES  = ("mae", "rmse")

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths  = paths
        self.logger = logger

    def _run_directories(self) -> list[Path]:
        if not self.paths.runs_dir.is_dir():
            return []

        directories = [entry for entry in self.paths.runs_dir.iterdir() if entry.is_dir() and entry.name not in self.SKIPPED_DIRECTORIES]
        directories.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
        return directories

    def _model_name(self, run_directory: Path) -> str:
        config_path = run_directory / "metadata" / "resolved_config.json"
        if not config_path.is_file():
            return "unknown"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return str(config.get("model_name", "unknown"))

    def _metrics_payload(self, run_directory: Path):
        metrics_path = run_directory / "metadata" / "metrics.json"
        if not metrics_path.is_file():
            return None
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _best_metric(self, metrics):
        if metrics is None:
            return None

        splits = metrics.get("splits", {})
        for split_name in self.BEST_METRIC_SPLITS:
            split_metrics = splits.get(split_name, {})
            if self.BEST_METRIC_NAME in split_metrics:
                return {"name": f"{split_name}/{self.BEST_METRIC_NAME}", "value": split_metrics[self.BEST_METRIC_NAME], "unit": metrics.get("unit")}
        return None

    def _coordinate_errors(self, metrics):
        if metrics is None:
            return None

        splits        = metrics.get("splits", {})
        required_keys = [f"{metric}_{name}" for metric in self.ERROR_METRIC_NAMES for name in self.COORDINATE_NAMES]

        for split_name in self.BEST_METRIC_SPLITS:
            split_metrics = splits.get(split_name, {})
            if not all(key in split_metrics for key in required_keys):
                continue

            values = {metric: {name: split_metrics[f"{metric}_{name}"] for name in self.COORDINATE_NAMES} for metric in self.ERROR_METRIC_NAMES}
            return {"split": split_name, "unit": metrics.get("unit"), "values": values}
        return None

    def _timestamp(self, run_directory: Path) -> str:
        return datetime.fromtimestamp(run_directory.stat().st_mtime).isoformat(timespec="seconds")

    def _tree(self, directory: Path, depth: int) -> dict:
        children = []
        files    = []

        try:
            entries = sorted(directory.iterdir())
        except OSError:
            entries = []

        for entry in entries:
            if entry.is_dir():
                if entry.name in self.SKIPPED_DIRECTORIES or entry.name.startswith("."):
                    continue
                if depth < self.MAX_DEPTH:
                    children.append(self._tree(entry, depth + 1))
            elif entry.is_file():
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                files.append({"name": entry.name, "size": size})

        return {"name": directory.name, "files": files, "children": children}

    def _run(self, run_directory: Path) -> dict:
        metrics = self._metrics_payload(run_directory)

        return {
            "name"              : run_directory.name,
            "path"              : str(run_directory),
            "model"             : self._model_name(run_directory),
            "timestamp"         : self._timestamp(run_directory),
            "best_metric"       : self._best_metric(metrics),
            "coordinate_errors" : self._coordinate_errors(metrics),
            "tree"              : self._tree(run_directory, 0),
        }

    def list_runs(self) -> dict:
        runs = [self._run(directory) for directory in self._run_directories()]
        return {"ok": True, "runs_dir": str(self.paths.runs_dir), "runs": runs}
