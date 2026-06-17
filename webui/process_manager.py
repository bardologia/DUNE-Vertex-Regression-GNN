from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import threading
from collections import deque
from datetime    import datetime

from project_paths import ProjectPaths
from server_logger import ServerLogger


class ProcessManager:

    OVERRIDE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
    TAIL_LIMIT    = 2000

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.processes = {}
        self.lock      = threading.Lock()

        self.paths.webui_logs_dir.mkdir(parents=True, exist_ok=True)

    def launch(self, script_key: str, overrides: dict | None, interpreter: str) -> dict:
        entry = self.paths.script_entry(script_key)
        if entry is None or not entry["path"].exists():
            return {"ok": False, "error": "script not found"}

        cleaned_overrides = self._clean_overrides(overrides) if entry["has_config"] else {}
        argv              = self._build_argv(interpreter, entry, cleaned_overrides)
        environment       = self._runtime_environment(interpreter)

        stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path  = self.paths.webui_logs_dir / f"{script_key}_{stamp}.log"

        try:
            log_handle = open(log_path, "wb")
            process    = subprocess.Popen(
                argv,
                cwd               = str(self.paths.repo_root),
                stdout            = log_handle,
                stderr            = subprocess.STDOUT,
                stdin             = subprocess.DEVNULL,
                env               = environment,
                start_new_session = True,
            )
        except OSError as error:
            return {"ok": False, "error": str(error)}

        record = {
            "pid"         : process.pid,
            "script"      : script_key,
            "command"     : self._render_command(interpreter, entry, cleaned_overrides),
            "interpreter" : interpreter,
            "overrides"   : cleaned_overrides,
            "status"      : "running",
            "started"     : datetime.now().isoformat(timespec="seconds"),
            "exit_code"   : None,
            "log_path"    : str(log_path),
            "process"     : process,
            "log_handle"  : log_handle,
        }

        with self.lock:
            self.processes[process.pid] = record

        self.logger.ok(f"launched {script_key} as pid {process.pid}")

        watcher = threading.Thread(target=self._watch, args=(process.pid,), daemon=True)
        watcher.start()

        return {"ok": True, "pid": process.pid, "log_path": str(log_path)}

    def _clean_overrides(self, overrides: dict | None) -> dict:
        cleaned = {}
        for path, value in (overrides or {}).items():
            if not isinstance(path, str) or not self.OVERRIDE_NAME.match(path):
                continue
            cleaned[path] = str(value)
        return cleaned

    def _build_argv(self, interpreter: str, entry: dict, overrides: dict) -> list[str]:
        argv = [interpreter, "-u", str(entry["path"])]
        for path, value in overrides.items():
            argv += [f"--{path}", value]
        return argv

    def _render_command(self, interpreter: str, entry: dict, overrides: dict) -> str:
        parts = [interpreter, "-u", entry["relative"]]
        for path, value in overrides.items():
            parts += [f"--{path}", shlex.quote(value)]
        return " ".join(parts)

    def _runtime_environment(self, interpreter: str) -> dict:
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["FORCE_COLOR"]      = "1"
        environment["COLUMNS"]          = "160"

        library_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(interpreter))), "lib")
        library_path      = environment.get("LD_LIBRARY_PATH", "")
        if library_directory not in library_path.split(":"):
            environment["LD_LIBRARY_PATH"] = library_directory + (":" + library_path if library_path else "")

        return environment

    def _watch(self, pid: int) -> None:
        with self.lock:
            record = self.processes.get(pid)
        if record is None:
            return

        process = record["process"]
        process.wait()
        code    = process.returncode

        with self.lock:
            record["status"]    = "finished" if code == 0 else "failed"
            record["exit_code"] = code
            handle = record.get("log_handle")
            if handle is not None and not handle.closed:
                handle.close()

        self.logger.info(f"process {pid} ({record['script']}) exited with code {code}")

    def _view(self, record: dict) -> dict:
        return {
            "pid"         : record["pid"],
            "script"      : record["script"],
            "command"     : record["command"],
            "interpreter" : record["interpreter"],
            "status"      : record["status"],
            "started"     : record["started"],
            "exit_code"   : record["exit_code"],
            "log_path"    : record["log_path"],
        }

    def list(self) -> list[dict]:
        with self.lock:
            records = list(self.processes.values())
        return [self._view(record) for record in sorted(records, key=lambda item: item["started"], reverse=True)]

    def kill(self, pid: int) -> dict:
        with self.lock:
            record = self.processes.get(pid)

        if record is None:
            return {"ok": False, "error": "unknown process"}
        if record["status"] != "running":
            return {"ok": False, "error": "process is not running"}

        self._signal_group(pid, signal.SIGTERM)
        self.logger.warning(f"stop requested for pid {pid}")
        return {"ok": True}

    @staticmethod
    def _signal_group(pid: int, requested_signal: signal.Signals) -> None:
        try:
            os.killpg(pid, requested_signal)
            return
        except (ProcessLookupError, PermissionError):
            pass

        try:
            os.kill(pid, requested_signal)
        except (ProcessLookupError, PermissionError):
            pass

    def tail(self, pid: int, line_count: int) -> dict:
        with self.lock:
            record = self.processes.get(pid)

        if record is None:
            return {"ok": False, "error": "unknown process"}

        requested = min(max(line_count, 1), self.TAIL_LIMIT)

        try:
            with open(record["log_path"], "r", encoding="utf-8", errors="replace") as handle:
                lines = deque(handle, maxlen=requested)
        except OSError as error:
            return {"ok": False, "error": str(error)}

        return {
            "ok"        : True,
            "pid"       : pid,
            "status"    : record["status"],
            "exit_code" : record["exit_code"],
            "lines"     : [line.rstrip("\n") for line in lines],
        }

    def stop_all(self, grace: float = 6.0) -> None:
        with self.lock:
            running = [record["pid"] for record in self.processes.values() if record["status"] == "running"]

        for pid in running:
            self._signal_group(pid, signal.SIGTERM)
            self.logger.warning(f"shutdown stop for pid {pid}")
