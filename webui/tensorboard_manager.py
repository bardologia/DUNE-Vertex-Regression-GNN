from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib  import Path

from project_paths import ProjectPaths
from server_logger import ServerLogger


class TensorboardManager:

    HOST              = "127.0.0.1"
    STARTUP_TIMEOUT_S = 90.0
    PROBE_INTERVAL_S  = 0.5

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.instances = {}
        self.lock      = threading.Lock()

    def default_logdir(self) -> str:
        return str(self.paths.runs_dir.resolve())

    def ensure_latest_run(self, settle_timeout_s: float = 90.0) -> None:
        runs_root = self.paths.runs_dir.resolve()
        before    = self._run_directories(runs_root)
        deadline  = time.monotonic() + settle_timeout_s

        while time.monotonic() < deadline:
            appeared = [path for path in self._run_directories(runs_root) - before if (path / "tensorboard").is_dir()]
            if appeared:
                newest = max(appeared, key=lambda path: path.stat().st_mtime)
                self.logger.ok(f"auto-starting tensorboard over new run {newest.name}")
                self.ensure(str(newest / "tensorboard"))
                return
            time.sleep(self.PROBE_INTERVAL_S)

        self.logger.warning("no new run directory appeared after launch; falling back to all runs")
        self.ensure(self.default_logdir())

    def _run_directories(self, runs_root: Path) -> set:
        try:
            return {child for child in runs_root.iterdir() if child.is_dir()}
        except OSError:
            return set()

    def resolve_run_logdir(self, run_directory: str) -> str | None:
        candidate = Path(run_directory).expanduser()
        if not candidate.is_absolute():
            candidate = self.paths.runs_dir / run_directory

        candidate = candidate.resolve()
        if not candidate.is_relative_to(self.paths.runs_dir.resolve()):
            return None
        if not candidate.is_dir():
            return None

        tensorboard_directory = candidate / "tensorboard"
        return str(tensorboard_directory if tensorboard_directory.is_dir() else candidate)

    def ensure(self, logdir: str) -> dict:
        logdir = str(logdir)

        with self.lock:
            for record in self.instances.values():
                if record["logdir"] == logdir and record["status"] in ("starting", "running") and record["process"].poll() is None:
                    return {"ok": True, **self._view(record)}

        port  = self._free_port()
        tb_id = uuid.uuid4().hex[:8]

        binary  = shutil.which("tensorboard")
        command = [binary] if binary else [self.paths.preferred_interpreter(), "-m", "tensorboard.main"]
        argv    = command + [
            "--logdir",          logdir,
            "--host",            self.HOST,
            "--port",            str(port),
            "--path_prefix",     f"/tb/{tb_id}",
            "--reload_interval", "30",
        ]

        log_path = self.paths.webui_logs_dir / f"tensorboard_{tb_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            log_handle = open(log_path, "wb")
            process    = subprocess.Popen(argv, cwd=str(self.paths.repo_root), stdout=log_handle, stderr=subprocess.STDOUT)
        except OSError as error:
            return {"ok": False, "error": str(error)}

        record = {
            "id"       : tb_id,
            "logdir"   : logdir,
            "port"     : port,
            "pid"      : process.pid,
            "status"   : "starting",
            "started"  : datetime.now().isoformat(timespec="seconds"),
            "process"  : process,
            "log_path" : log_path,
        }

        with self.lock:
            self.instances[tb_id] = record

        self.logger.ok(f"tensorboard {tb_id} starting on port {port} for {logdir}")

        threading.Thread(target=self._probe, args=(record,), daemon=True).start()

        return {"ok": True, **self._view(record)}

    def _free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((self.HOST, 0))
            return probe.getsockname()[1]

    def _probe(self, record: dict) -> None:
        url      = f"http://{self.HOST}:{record['port']}/tb/{record['id']}/"
        deadline = time.monotonic() + self.STARTUP_TIMEOUT_S

        while time.monotonic() < deadline:
            if record["process"].poll() is not None:
                record["status"] = "failed"
                self.logger.error(f"tensorboard {record['id']} exited during startup: {self._log_tail(record)}")
                return

            try:
                with urllib.request.urlopen(url, timeout=2):
                    record["status"] = "running"
                    self.logger.ok(f"tensorboard {record['id']} ready at /tb/{record['id']}/")
                    return
            except OSError:
                time.sleep(self.PROBE_INTERVAL_S)

        record["status"] = "failed"
        if record["process"].poll() is None:
            record["process"].terminate()
        self.logger.error(f"tensorboard {record['id']} failed to start within {self.STARTUP_TIMEOUT_S:.0f}s: {self._log_tail(record)}")

    def _log_tail(self, record: dict, max_chars: int = 500) -> str:
        try:
            text = record["log_path"].read_text(errors="replace").strip()
        except OSError:
            return "no output captured"
        return text[-max_chars:] if text else "no output captured"

    def get(self, tb_id: str) -> dict | None:
        with self.lock:
            return self.instances.get(tb_id)

    def _view(self, record: dict) -> dict:
        if record["status"] == "running" and record["process"].poll() is not None:
            record["status"] = "stopped"
        return {
            "id"      : record["id"],
            "logdir"  : record["logdir"],
            "port"    : record["port"],
            "pid"     : record["pid"],
            "status"  : record["status"],
            "started" : record["started"],
            "url"     : f"/tb/{record['id']}/",
        }

    def list_instances(self) -> list[dict]:
        with self.lock:
            records = list(self.instances.values())
        return [self._view(record) for record in sorted(records, key=lambda item: item["started"], reverse=True)]

    def stop(self, tb_id: str) -> dict:
        with self.lock:
            record = self.instances.get(tb_id)

        if record is None:
            return {"ok": False, "error": "unknown tensorboard instance"}

        if record["process"].poll() is None:
            record["process"].terminate()

        record["status"] = "stopped"
        self.logger.warning(f"tensorboard {tb_id} stopped")
        return {"ok": True}

    def stop_all(self) -> None:
        with self.lock:
            records = list(self.instances.values())

        for record in records:
            if record["process"].poll() is None:
                record["process"].terminate()
            record["status"] = "stopped"
