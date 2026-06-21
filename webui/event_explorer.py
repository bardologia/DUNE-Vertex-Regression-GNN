from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib  import Path

import numpy as np

from project_paths import ProjectPaths
from server_logger import ServerLogger


class EventExplorer:

    SPLITS    = ("test", "val", "train")
    DATASETS  = ({"name": "raw", "label": "raw events"}, {"name": "augmented", "label": "octant-augmented"})
    FILENAME  = "event_explorer.npz"

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths  = paths
        self.logger = logger
        self.lock   = threading.Lock()
        self.loaded = None
        self.status = {"state": "idle", "kind": None, "name": None, "split": None, "stage": "", "error": ""}

    def list_sources(self) -> dict:
        datasets = []
        for dataset in self.DATASETS:
            cached = (self.paths.explorer_cache_dir / f"{dataset['name']}.npz").is_file()
            datasets.append({"name": dataset["name"], "label": dataset["label"], "cached": cached})

        return {"ok": True, "datasets": datasets, "runs": self._list_runs()}

    def _list_runs(self) -> list:
        if not self.paths.runs_dir.is_dir():
            return []

        runs = []
        for directory in sorted(self.paths.runs_dir.iterdir(), key=lambda entry: entry.stat().st_mtime, reverse=True):
            if not directory.is_dir():
                continue
            if not (directory / "checkpoints" / "best_model.pt").is_file():
                continue
            if not (directory / "metadata" / "resolved_config.json").is_file():
                continue

            cached = [split for split in self.SPLITS if (directory / "analysis" / split / self.FILENAME).is_file()]
            runs.append({"run": directory.name, "model": self._model_name(directory), "cached": cached})

        return runs

    def start_load(self, kind: str, name: str, split: str) -> dict:
        if kind == "dataset":
            if name not in [dataset["name"] for dataset in self.DATASETS]:
                return {"ok": False, "error": f"unknown dataset: {name}"}
            split = "all"
        elif kind == "run":
            if split not in self.SPLITS:
                return {"ok": False, "error": f"unknown split: {split}"}
            if self._run_directory(name) is None:
                return {"ok": False, "error": f"unknown run: {name}"}
        else:
            return {"ok": False, "error": f"unknown source kind: {kind}"}

        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": f"already loading {self.status['name']} / {self.status['split']}"}
            if self.status["state"] == "ready" and self.loaded is not None and self.loaded["kind"] == kind and self.loaded["name"] == name and self.loaded["split"] == split:
                return {"ok": True}

            self.loaded = None
            self.status = {"state": "loading", "kind": kind, "name": name, "split": split, "stage": "preparing", "error": ""}

        threading.Thread(target=self._load_worker, args=(kind, name, split), daemon=True).start()
        return {"ok": True}

    def load_status(self) -> dict:
        with self.lock:
            payload = dict(self.status)
            if payload["state"] == "ready" and self.loaded is not None:
                payload["meta"] = self.loaded["meta"]
        return payload

    def events(self, kind: str, name: str, split: str) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["kind"] != kind or self.loaded["name"] != name or self.loaded["split"] != split:
                return {"ok": False, "error": "source not loaded"}
            data = self.loaded["data"]
            meta = self.loaded["meta"]

        payload = {
            "ok"             : True,
            "meta"           : meta,
            "has_prediction" : meta["has_prediction"],
            "gt"             : data["gt"].tolist(),
            "n_active"       : data["n_active"].tolist(),
        }
        if meta["has_prediction"]:
            payload["pred"]  = data["pred"].tolist()
            payload["error"] = data["error"].tolist()
        return payload

    def detail(self, kind: str, name: str, split: str, index: int) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["kind"] != kind or self.loaded["name"] != name or self.loaded["split"] != split:
                return {"ok": False, "error": "source not loaded"}
            data = self.loaded["data"]
            meta = self.loaded["meta"]

        count = int(meta["count"])
        if index < 0 or index >= count:
            return {"ok": False, "error": f"index {index} out of range [0, {count})"}

        if meta["has_prediction"]:
            return self._run_detail(data, meta, index)
        return self._dataset_detail(data, meta, index)

    def _run_detail(self, data, meta, index) -> dict:
        start = int(data["offsets"][index])
        end   = int(data["offsets"][index + 1])

        error = float(data["error"][index])
        rank  = float((data["error"] < error).mean())

        return {
            "ok"             : True,
            "index"          : index,
            "has_prediction" : True,
            "gt"             : data["gt"][index].tolist(),
            "pred"           : data["pred"][index].tolist(),
            "error"          : error,
            "error_xyz"      : data["error_xyz"][index].tolist(),
            "base_event_id"  : int(data["base_event_id"][index]),
            "signs"          : data["signs"][index].tolist(),
            "n_active"       : int(data["n_active"][index]),
            "total_light"    : float(data["total_light"][index]),
            "error_rank"     : rank,
            "sensors"        : {
                "positions" : data["sensor_positions"][start:end].tolist(),
                "light"     : data["sensor_light"][start:end].tolist(),
            },
        }

    def _dataset_detail(self, data, meta, index) -> dict:
        base_event_id = int(data["base_event_id"][index])
        start         = int(data["base_offsets"][base_event_id])
        end           = int(data["base_offsets"][base_event_id + 1])

        signs     = data["signs"][index].astype(np.float32)
        positions = data["base_positions"][start:end] * signs[None, :]
        light     = data["base_light"][start:end]

        return {
            "ok"             : True,
            "index"          : index,
            "has_prediction" : False,
            "gt"             : data["gt"][index].tolist(),
            "base_event_id"  : base_event_id,
            "signs"          : data["signs"][index].tolist(),
            "octant"         : str(data["octant"][index]),
            "n_active"       : int(data["n_active"][index]),
            "total_light"    : float(data["total_light"][index]),
            "sensors"        : {
                "positions" : positions.tolist(),
                "light"     : light.tolist(),
            },
        }

    def _model_name(self, run_directory: Path) -> str:
        config_path = run_directory / "metadata" / "resolved_config.json"
        try:
            return str(json.loads(config_path.read_text(encoding="utf-8")).get("model_name", "unknown"))
        except (OSError, ValueError):
            return "unknown"

    def _run_directory(self, run: str) -> Path | None:
        if not run:
            return None

        run_directory = (self.paths.runs_dir / run).resolve()
        if not run_directory.is_relative_to(self.paths.runs_dir.resolve()):
            return None
        if not (run_directory / "checkpoints" / "best_model.pt").is_file():
            return None
        return run_directory

    def _load_worker(self, kind: str, name: str, split: str) -> None:
        try:
            if kind == "dataset":
                cache = self.paths.explorer_cache_dir / f"{name}.npz"
                if not cache.is_file():
                    self._set_stage("building the dataset cache")
                    self._run_dataset_export(name)
                if not cache.is_file():
                    raise FileNotFoundError(f"export finished but {cache} was not produced; check the export log")

                self._set_stage("loading event cache")
                data, meta = self._read_dataset_cache(cache, name)
            else:
                run_directory = self._run_directory(name)
                cache         = run_directory / "analysis" / split / self.FILENAME
                if not cache.is_file():
                    self._set_stage("running the model over every split")
                    self._run_export(run_directory)
                if not cache.is_file():
                    raise FileNotFoundError(f"export finished but {cache} was not produced; check the export log")

                self._set_stage("loading event cache")
                data, meta = self._read_run_cache(cache, name, split)

            with self.lock:
                self.loaded = {"kind": kind, "name": name, "split": split, "data": data, "meta": meta}
                self.status = {"state": "ready", "kind": kind, "name": name, "split": split, "stage": "ready", "error": ""}

            self.logger.ok(f"event explorer ready: {kind}:{name} / {split} ({meta['count']} events)")
        except Exception as error:
            with self.lock:
                self.loaded = None
                self.status = {"state": "error", "kind": kind, "name": name, "split": split, "stage": "", "error": str(error)}

            self.logger.error(f"event explorer load failed: {kind}:{name} / {split}: {error}")

    def _set_stage(self, stage: str) -> None:
        with self.lock:
            self.status["stage"] = stage

    def _run_export(self, run_directory: Path) -> None:
        script = self.paths.main_dir / "export_events.py"
        argv_tail = ["--run_directory", str(run_directory)]
        self._spawn(script, argv_tail, f"event_export_{run_directory.name}")

    def _run_dataset_export(self, name: str) -> None:
        script = self.paths.main_dir / "export_dataset_events.py"
        argv_tail = ["--kind", name]
        self._spawn(script, argv_tail, f"dataset_export_{name}")

    def _spawn(self, script: Path, argv_tail: list, log_stem: str) -> None:
        interpreter = self.paths.preferred_interpreter()
        argv        = [interpreter, "-u", str(script), *argv_tail]

        self.paths.webui_logs_dir.mkdir(parents=True, exist_ok=True)
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.paths.webui_logs_dir / f"{log_stem}_{stamp}.out"

        with open(log_path, "ab") as sink:
            process = subprocess.run(
                argv,
                cwd    = str(self.paths.repo_root),
                stdout = sink,
                stderr = subprocess.STDOUT,
                stdin  = subprocess.DEVNULL,
                env    = self._environment(interpreter),
            )

        if process.returncode != 0:
            raise RuntimeError(f"{script.name} exited with code {process.returncode}; see {log_path}")

    def _environment(self, interpreter: str) -> dict:
        environment                     = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"

        library_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(interpreter))), "lib")
        library_path      = environment.get("LD_LIBRARY_PATH", "")
        if library_directory not in library_path.split(":"):
            environment["LD_LIBRARY_PATH"] = library_directory + (":" + library_path if library_path else "")

        return environment

    def _read_run_cache(self, cache: Path, name: str, split: str) -> tuple[dict, dict]:
        archive = np.load(cache)
        data    = {key: archive[key] for key in archive.files}

        gt    = data["gt"]
        error = data["error"]

        meta = {
            "kind"           : "run",
            "name"           : name,
            "split"          : split,
            "has_prediction" : True,
            "count"          : int(gt.shape[0]),
            "generated"      : datetime.fromtimestamp(cache.stat().st_mtime).isoformat(timespec="seconds"),
            "bounds_min"     : gt.min(axis=0).tolist() if gt.shape[0] else [0.0, 0.0, 0.0],
            "bounds_max"     : gt.max(axis=0).tolist() if gt.shape[0] else [0.0, 0.0, 0.0],
            "error_mean"     : float(error.mean()) if error.size else 0.0,
            "error_median"   : float(np.median(error)) if error.size else 0.0,
            "error_p90"      : float(np.percentile(error, 90.0)) if error.size else 0.0,
        }
        return data, meta

    def _read_dataset_cache(self, cache: Path, name: str) -> tuple[dict, dict]:
        archive = np.load(cache, allow_pickle=False)
        data    = {key: archive[key] for key in archive.files}

        gt = data["gt"]

        meta = {
            "kind"           : "dataset",
            "name"           : name,
            "split"          : "all",
            "has_prediction" : False,
            "count"          : int(gt.shape[0]),
            "generated"      : datetime.fromtimestamp(cache.stat().st_mtime).isoformat(timespec="seconds"),
            "bounds_min"     : gt.min(axis=0).tolist() if gt.shape[0] else [0.0, 0.0, 0.0],
            "bounds_max"     : gt.max(axis=0).tolist() if gt.shape[0] else [0.0, 0.0, 0.0],
        }
        return data, meta
