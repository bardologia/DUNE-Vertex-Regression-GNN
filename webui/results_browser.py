from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

from project_paths import ProjectPaths
from server_logger import ServerLogger


class ResultsBrowser:

    SKIPPED_DIRECTORIES = {"__pycache__", ".git", ".ipynb_checkpoints"}
    MAX_DEPTH           = 6
    METADATA_FILES      = ("metadata/summary.json", "metadata/metrics.json", "metadata.json", "summary.json")
    METRIC_KEYS         = ("best_metric", "best_validation_loss", "best_val_loss", "best_loss", "euclidean_error", "rmse", "mae")

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths  = paths
        self.logger = logger

    def _run_directories(self) -> list[Path]:
        if not self.paths.runs_dir.is_dir():
            return []

        directories = [entry for entry in self.paths.runs_dir.iterdir() if entry.is_dir() and entry.name not in self.SKIPPED_DIRECTORIES]
        directories.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
        return directories

    def _metadata(self, run_directory: Path) -> dict:
        for relative in self.METADATA_FILES:
            candidate = run_directory / relative
            if candidate.is_file():
                try:
                    return json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
        return {}

    def _best_metric(self, metadata: dict):
        for key in self.METRIC_KEYS:
            if key in metadata:
                return {"name": key, "value": metadata[key]}
        return None

    def _model_name(self, metadata: dict, run_directory: Path) -> str:
        for key in ("model_name", "model", "backbone_name"):
            if metadata.get(key):
                return str(metadata[key])

        name  = run_directory.name
        first = name.split("_")[0]
        return first if first and not first.startswith("run") else "unknown"

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
        metadata = self._metadata(run_directory)
        return {
            "name"        : run_directory.name,
            "path"        : str(run_directory),
            "model"       : self._model_name(metadata, run_directory),
            "timestamp"   : self._timestamp(run_directory),
            "best_metric" : self._best_metric(metadata),
            "tree"        : self._tree(run_directory, 0),
        }

    def list_runs(self) -> dict:
        runs = [self._run(directory) for directory in self._run_directories()]
        return {"ok": True, "runs_dir": str(self.paths.runs_dir), "runs": runs}
