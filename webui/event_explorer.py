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

    SPLITS   = ("test", "val", "train")
    FILENAME = "event_explorer.npz"

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths  = paths
        self.logger = logger
        self.lock   = threading.Lock()
        self.loaded = None
        self.status = {"state": "idle", "run": None, "split": None, "stage": "", "error": ""}

    def list_runs(self) -> dict:
        if not self.paths.runs_dir.is_dir():
            return {"ok": True, "runs": []}

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

        return {"ok": True, "runs": runs}

    def start_load(self, run: str, split: str) -> dict:
        if split not in self.SPLITS:
            return {"ok": False, "error": f"unknown split: {split}"}

        run_directory = self._run_directory(run)
        if run_directory is None:
            return {"ok": False, "error": f"unknown run: {run}"}

        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": f"already loading {self.status['run']} / {self.status['split']}"}
            if self.status["state"] == "ready" and self.loaded is not None and self.loaded["run"] == run and self.loaded["split"] == split:
                return {"ok": True}

            self.loaded = None
            self.status = {"state": "loading", "run": run, "split": split, "stage": "preparing", "error": ""}

        threading.Thread(target=self._load_worker, args=(run, split, run_directory), daemon=True).start()
        return {"ok": True}

    def load_status(self) -> dict:
        with self.lock:
            payload = dict(self.status)
            if payload["state"] == "ready" and self.loaded is not None:
                payload["meta"] = self.loaded["meta"]
        return payload

    def events(self, run: str, split: str) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["run"] != run or self.loaded["split"] != split:
                return {"ok": False, "error": "split not loaded"}
            data = self.loaded["data"]
            meta = self.loaded["meta"]

        return {
            "ok"       : True,
            "meta"     : meta,
            "gt"       : data["gt"].tolist(),
            "pred"     : data["pred"].tolist(),
            "error"    : data["error"].tolist(),
            "n_active" : data["n_active"].tolist(),
        }

    def detail(self, run: str, split: str, index: int) -> dict:
        with self.lock:
            if self.loaded is None or self.loaded["run"] != run or self.loaded["split"] != split:
                return {"ok": False, "error": "split not loaded"}
            data = self.loaded["data"]
            meta = self.loaded["meta"]

        count = int(meta["count"])
        if index < 0 or index >= count:
            return {"ok": False, "error": f"index {index} out of range [0, {count})"}

        start = int(data["offsets"][index])
        end   = int(data["offsets"][index + 1])

        gt    = data["gt"][index]
        pred  = data["pred"][index]
        error = float(data["error"][index])
        rank  = float((data["error"] < error).mean())

        return {
            "ok"            : True,
            "index"         : index,
            "gt"            : gt.tolist(),
            "pred"          : pred.tolist(),
            "error"         : error,
            "error_xyz"     : data["error_xyz"][index].tolist(),
            "base_event_id" : int(data["base_event_id"][index]),
            "signs"         : data["signs"][index].tolist(),
            "n_active"      : int(data["n_active"][index]),
            "total_light"   : float(data["total_light"][index]),
            "error_rank"    : rank,
            "sensors"       : {
                "positions" : data["sensor_positions"][start:end].tolist(),
                "light"     : data["sensor_light"][start:end].tolist(),
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

    def _load_worker(self, run: str, split: str, run_directory: Path) -> None:
        try:
            cache = run_directory / "analysis" / split / self.FILENAME

            if not cache.is_file():
                self._set_stage("running the model over every split")
                self._run_export(run_directory)

            if not cache.is_file():
                raise FileNotFoundError(f"export finished but {cache} was not produced; check the export log")

            self._set_stage("loading event cache")
            data, meta = self._read_cache(cache, run, split)

            with self.lock:
                self.loaded = {"run": run, "split": split, "data": data, "meta": meta}
                self.status = {"state": "ready", "run": run, "split": split, "stage": "ready", "error": ""}

            self.logger.ok(f"event explorer ready: {run} / {split} ({meta['count']} events)")
        except Exception as error:
            with self.lock:
                self.loaded = None
                self.status = {"state": "error", "run": run, "split": split, "stage": "", "error": str(error)}

            self.logger.error(f"event explorer load failed: {run} / {split}: {error}")

    def _set_stage(self, stage: str) -> None:
        with self.lock:
            self.status["stage"] = stage

    def _run_export(self, run_directory: Path) -> None:
        interpreter = self.paths.preferred_interpreter()
        script      = self.paths.main_dir / "export_events.py"
        argv        = [interpreter, "-u", str(script), "--run_directory", str(run_directory)]

        self.paths.webui_logs_dir.mkdir(parents=True, exist_ok=True)
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.paths.webui_logs_dir / f"event_export_{run_directory.name}_{stamp}.out"

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
            raise RuntimeError(f"export_events.py exited with code {process.returncode}; see {log_path}")

    def _environment(self, interpreter: str) -> dict:
        environment                     = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"

        library_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(interpreter))), "lib")
        library_path      = environment.get("LD_LIBRARY_PATH", "")
        if library_directory not in library_path.split(":"):
            environment["LD_LIBRARY_PATH"] = library_directory + (":" + library_path if library_path else "")

        return environment

    def _read_cache(self, cache: Path, run: str, split: str) -> tuple[dict, dict]:
        archive = np.load(cache)
        data    = {key: archive[key] for key in archive.files}

        gt    = data["gt"]
        error = data["error"]

        meta = {
            "run"            : run,
            "split"          : split,
            "count"          : int(gt.shape[0]),
            "generated"      : datetime.fromtimestamp(cache.stat().st_mtime).isoformat(timespec="seconds"),
            "bounds_min"     : gt.min(axis=0).tolist() if gt.shape[0] else [0.0, 0.0, 0.0],
            "bounds_max"     : gt.max(axis=0).tolist() if gt.shape[0] else [0.0, 0.0, 0.0],
            "error_mean"     : float(error.mean()) if error.size else 0.0,
            "error_median"   : float(np.median(error)) if error.size else 0.0,
            "error_p90"      : float(np.percentile(error, 90.0)) if error.size else 0.0,
        }
        return data, meta
