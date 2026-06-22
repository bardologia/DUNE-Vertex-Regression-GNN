from __future__ import annotations

import sys
from pathlib import Path


class ProjectPaths:

    PREFERRED_ENVIRONMENT = "conda:Dune"

    SCRIPTS = {
        "train": {
            "file"                : "train.py",
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "TrainEntryConfig",
        },
        "tune": {
            "file"                : "tune.py",
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "TuneEntryConfig",
        },
        "cross_validate": {
            "file"                : "cross_validate.py",
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "CrossValidationEntryConfig",
        },
        "benchmark": {
            "file"                : "benchmark.py",
            "entry_config_module" : "configuration.benchmark.general",
            "entry_config_class"  : "BenchmarkConfig",
        },
        "infer": {
            "file"                : "infer.py",
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "InferenceEntryConfig",
        },
        "explain": {
            "file"                : "explain.py",
            "entry_config_module" : "configuration.explainability.general",
            "entry_config_class"  : "FeatureImportanceConfig",
        },
        "build_parquet_store": {
            "file"                : "build_parquet_store.py",
            "entry_config_module" : "configuration.entry.general",
            "entry_config_class"  : "DatasetEntryConfig",
        },
    }

    def __init__(self) -> None:
        self.webui_root = Path(__file__).resolve().parent
        self.repo_root  = self.webui_root.parent
        self.main_dir   = self.repo_root / "main"
        self.static_dir = self.webui_root / "static"
        self.runs_dir   = self.repo_root / "runs"
        self.logs_dir   = self.repo_root / "logs"
        self.webui_logs_dir      = self.logs_dir / "webui"
        self.explorer_cache_dir  = self.repo_root / "data_frames" / "event_explorer"

    def script_keys(self) -> list[str]:
        return list(self.SCRIPTS.keys())

    def script_entry(self, key: str) -> dict | None:
        specification = self.SCRIPTS.get(key)
        if specification is None:
            return None

        return {
            "key"                 : key,
            "path"                : self.main_dir / specification["file"],
            "relative"            : f"main/{specification['file']}",
            "entry_config_module" : specification["entry_config_module"],
            "entry_config_class"  : specification["entry_config_class"],
            "has_config"          : specification["entry_config_class"] is not None,
        }

    def discover_interpreters(self) -> list[dict]:
        discovered = []
        seen       = set()
        current    = str(Path(sys.executable))

        home  = Path.home()
        bases = [home / "miniconda3", home / "anaconda3", home / ".conda"]

        for base in bases:
            base_python = base / "bin" / "python"
            if base_python.exists() and str(base_python) not in seen:
                discovered.append({"label": "conda:base", "path": str(base_python), "current": str(base_python) == current})
                seen.add(str(base_python))

            environments_directory = base / "envs"
            if environments_directory.is_dir():
                for environment in sorted(environments_directory.iterdir()):
                    environment_python = environment / "bin" / "python"
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
