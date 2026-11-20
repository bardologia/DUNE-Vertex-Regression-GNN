from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

from configuration.shared        import IoNames
from configuration.webui.general import ResultsBrowsing

from project_paths import ProjectPaths
from server_logger import ServerLogger


class ResultsBrowser:

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths  = paths
        self.logger = logger

    def _is_run(self, directory: Path) -> bool:
        return (directory / IoNames.metadata_dir / IoNames.resolved_config_file).is_file()

    def _collect_runs(self, directory: Path, depth: int) -> list[Path]:
        if self._is_run(directory):
            return [directory]
        if depth >= ResultsBrowsing.nested_run_depth:
            return []

        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return []

        nested = []
        for entry in entries:
            if entry.is_dir() and entry.name not in IoNames.browser_skipped_dirs and not entry.name.startswith("."):
                nested.extend(self._collect_runs(entry, depth + 1))
        return nested

    def _run_directories(self) -> list[Path]:
        if not self.paths.runs_dir.is_dir():
            return []

        directories = []
        for entry in sorted(self.paths.runs_dir.iterdir()):
            if entry.is_dir() and entry.name not in IoNames.browser_skipped_dirs:
                directories.extend(self._collect_runs(entry, 0))

        directories.sort(key=lambda directory: directory.stat().st_mtime, reverse=True)
        return directories

    def _run_name(self, run_directory: Path) -> str:
        return str(run_directory.relative_to(self.paths.runs_dir))

    def _model_name(self, run_directory: Path) -> str:
        config_path = run_directory / IoNames.metadata_dir / IoNames.resolved_config_file
        if not config_path.is_file():
            return "unknown"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return str(config.get("model_name", "unknown"))

    def _metrics_payload(self, run_directory: Path):
        metrics_path = run_directory / IoNames.metadata_dir / IoNames.metrics_file
        if not metrics_path.is_file():
            return None
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _best_metric(self, metrics):
        if metrics is None:
            return None

        splits = metrics.get("splits", {})
        for split_name in ResultsBrowsing.best_metric_splits:
            split_metrics = splits.get(split_name, {})
            if ResultsBrowsing.best_metric_name in split_metrics:
                return {"name": f"{split_name}/{ResultsBrowsing.best_metric_name}", "value": split_metrics[ResultsBrowsing.best_metric_name], "unit": metrics.get("unit")}
        return None

    def _coordinate_errors(self, metrics):
        if metrics is None:
            return None

        splits        = metrics.get("splits", {})
        required_keys = [f"{metric}_{name}" for metric in ResultsBrowsing.error_metric_names for name in ResultsBrowsing.coordinate_names]

        for split_name in ResultsBrowsing.best_metric_splits:
            split_metrics = splits.get(split_name, {})
            if not all(key in split_metrics for key in required_keys):
                continue

            values = {metric: {name: split_metrics[f"{metric}_{name}"] for name in ResultsBrowsing.coordinate_names} for metric in ResultsBrowsing.error_metric_names}
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
                if entry.name in IoNames.browser_skipped_dirs or entry.name.startswith("."):
                    continue
                if depth < ResultsBrowsing.max_depth:
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
            "name"              : self._run_name(run_directory),
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
