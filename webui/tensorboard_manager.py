from __future__ import annotations

import shutil
import socket
import subprocess
import threading
from datetime import datetime
from pathlib  import Path

from project_paths import ProjectPaths
from server_logger import ServerLogger


class TensorboardManager:

    HOST = "127.0.0.1"

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.instances = {}
        self.lock      = threading.Lock()

    def _resolve_logdir(self, run_directory: str) -> Path | None:
        candidate = Path(run_directory).expanduser()
        if not candidate.is_absolute():
            candidate = self.paths.runs_dir / run_directory

        candidate = candidate.resolve()
        if not candidate.is_relative_to(self.paths.runs_dir.resolve()):
            return None
        if not candidate.is_dir():
            return None

        tensorboard_directory = candidate / "tensorboard"
        return tensorboard_directory if tensorboard_directory.is_dir() else candidate

    def _existing(self, logdir: str) -> dict | None:
        for record in self.instances.values():
            if record["logdir"] == logdir and record["process"].poll() is None:
                return record
        return None

    def launch(self, run_directory: str) -> dict:
        logdir = self._resolve_logdir(run_directory)
        if logdir is None:
            return {"ok": False, "error": "run directory not found under runs/"}

        logdir_text = str(logdir)

        with self.lock:
            existing = self._existing(logdir_text)
            if existing is not None:
                return {"ok": True, **self._view(existing)}

        port    = self._free_port()
        binary  = shutil.which("tensorboard")
        command = [binary] if binary else ["tensorboard"]
        argv    = command + ["--logdir", logdir_text, "--host", self.HOST, "--port", str(port), "--reload_interval", "30"]

        log_path = self.paths.webui_logs_dir / f"tensorboard_{port}.log"

        try:
            log_handle = open(log_path, "wb")
            process    = subprocess.Popen(argv, cwd=str(self.paths.repo_root), stdout=log_handle, stderr=subprocess.STDOUT)
        except OSError as error:
            return {"ok": False, "error": str(error)}

        record = {
            "logdir"  : logdir_text,
            "port"    : port,
            "pid"     : process.pid,
            "url"     : f"http://{self.HOST}:{port}/",
            "started" : datetime.now().isoformat(timespec="seconds"),
            "process" : process,
        }

        with self.lock:
            self.instances[process.pid] = record

        self.logger.ok(f"tensorboard started on port {port} for {logdir_text}")
        return {"ok": True, **self._view(record)}

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((self.HOST, 0))
            return probe.getsockname()[1]

    def _view(self, record: dict) -> dict:
        running = record["process"].poll() is None
        return {
            "pid"     : record["pid"],
            "logdir"  : record["logdir"],
            "port"    : record["port"],
            "url"     : record["url"],
            "started" : record["started"],
            "status"  : "running" if running else "stopped",
        }

    def list(self) -> list[dict]:
        with self.lock:
            records = list(self.instances.values())
        return [self._view(record) for record in sorted(records, key=lambda item: item["started"], reverse=True)]

    def stop(self, pid: int) -> dict:
        with self.lock:
            record = self.instances.get(pid)

        if record is None:
            return {"ok": False, "error": "unknown tensorboard instance"}

        if record["process"].poll() is None:
            record["process"].terminate()

        self.logger.warning(f"tensorboard {pid} stopped")
        return {"ok": True}

    def stop_all(self) -> None:
        with self.lock:
            records = list(self.instances.values())

        for record in records:
            if record["process"].poll() is None:
                record["process"].terminate()
