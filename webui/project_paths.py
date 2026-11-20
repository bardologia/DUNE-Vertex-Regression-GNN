from __future__ import annotations

import sys
from pathlib import Path

from configuration.shared import IoNames


class ProjectPaths:

    PREFERRED_ENVIRONMENT = "conda:Dune"

    SCRIPTS = {
        "train": {
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "TrainEntryConfig",
        },
        "tune": {
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "TuneEntryConfig",
        },
        "cross_validate": {
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "CrossValidationEntryConfig",
        },
        "benchmark": {
            "entry_config_module" : "configuration.benchmark.general",
            "entry_config_class"  : "BenchmarkConfig",
        },
        "baseline": {
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "BaselineEntryConfig",
        },
        "infer": {
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "InferenceEntryConfig",
        },
        "explain": {
            "entry_config_module" : "configuration.explainability.general",
            "entry_config_class"  : "FeatureImportanceConfig",
        },
        "build_parquet_store": {
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "DatasetEntryConfig",
        },
    }

    def __init__(self) -> None:
        self.repo_root          = IoNames.project_root
        self.webui_root         = self.repo_root / IoNames.webui_dir
        self.main_dir           = self.repo_root / IoNames.main_dir
        self.static_dir         = self.webui_root / IoNames.static_dir
        self.runs_dir           = self.repo_root / IoNames.runs_dir
        self.logs_dir           = self.repo_root / IoNames.logs_dir
        self.webui_logs_dir     = self.logs_dir / IoNames.webui_dir
        self.explorer_cache_dir = self.repo_root / IoNames.data_frames_dir / IoNames.event_explorer_dir

    def script_keys(self) -> list[str]:
        return list(self.SCRIPTS.keys())

    def script_entry(self, key: str) -> dict | None:
        specification = self.SCRIPTS.get(key)
        if specification is None:
            return None

        file = f"{key}{IoNames.python_suffix}"

        return {
            "key"                 : key,
            "path"                : self.main_dir / file,
            "relative"            : f"{IoNames.main_dir}/{file}",
            "entry_config_module" : specification["entry_config_module"],
            "entry_config_class"  : specification["entry_config_class"],
            "has_config"          : specification["entry_config_class"] is not None,
        }

    def discover_interpreters(self) -> list[dict]:
        discovered = []
        seen       = set()
        current    = str(Path(sys.executable))

        home  = Path.home()
        bases = [home / base for base in IoNames.conda_bases]

        for base in bases:
            base_python = base / IoNames.interpreter_bin_dir / IoNames.python_executable
            if base_python.exists() and str(base_python) not in seen:
                discovered.append({"label": "conda:base", "path": str(base_python), "current": str(base_python) == current})
                seen.add(str(base_python))

            environments_directory = base / IoNames.conda_envs_dir
            if environments_directory.is_dir():
                for environment in sorted(environments_directory.iterdir()):
                    environment_python = environment / IoNames.interpreter_bin_dir / IoNames.python_executable
                    if environment_python.exists() and str(environment_python) not in seen:
                        discovered.append({"label": f"conda:{environment.name}", "path": str(environment_python), "current": str(environment_python) == current})
                        seen.add(str(environment_python))

        if current not in seen and Path(current).exists():
            discovered.insert(0, {"label": "current", "path": current, "current": True})

        return discovered

    def preferred_interpreter(self, interpreters: list[dict] | None = None) -> str:
        candidates = interpreters if interpreters is not None else self.discover_interpreters()

        for candidate in candidates:
            if candidate["label"] == self.PREFERRED_ENVIRONMENT:
                return candidate["path"]

        return candidates[0]["path"] if candidates else sys.executable
