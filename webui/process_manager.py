from __future__ import annotations

import codecs
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
from datetime    import datetime
from collections import deque

from project_paths import ProjectPaths
from server_logger import ServerLogger


class JobStream:

    def __init__(self) -> None:
        self.buffer      = deque(maxlen=4000)
        self.subscribers = []
        self.lock        = threading.Lock()

    def publish(self, event: dict) -> None:
        with self.lock:
            self.buffer.append(event)
            for subscriber in list(self.subscribers):
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self) -> queue.Queue:
        subscriber = queue.Queue(maxsize=8000)
        with self.lock:
            for event in self.buffer:
                subscriber.put_nowait(event)
            self.subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)


class ProcessManager:

    OVERRIDE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
    TERMINAL_COLUMNS = 120

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths     = paths
        self.logger    = logger
        self.processes = {}
        self.streams   = {}
        self.lock      = threading.Lock()

    def launch(self, script_key: str, overrides: dict | None, interpreter: str) -> dict:
        if not self._is_permitted_interpreter(interpreter):
            return {"ok": False, "error": "interpreter not permitted"}

        entry = self.paths.script_entry(script_key)
        if entry is None or not entry["path"].exists():
            return {"ok": False, "error": "script not found"}

        cleaned_overrides = self._clean_overrides(overrides) if entry["has_config"] else {}
        argv              = self._build_argv(interpreter, entry, cleaned_overrides)
        environment       = self._runtime_environment(interpreter)

        try:
            process = subprocess.Popen(
                argv,
                cwd               = str(self.paths.repo_root),
                stdout            = subprocess.PIPE,
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
            "process"     : process,
        }
        stream = JobStream()

        with self.lock:
            self.processes[process.pid] = record
            self.streams[process.pid]   = stream

        self.logger.ok(f"launched {script_key} as pid {process.pid}")
        stream.publish({"type": "status", "status": "running", "pid": process.pid})

        pump = threading.Thread(target=self._pump, args=(process.pid, process, stream), daemon=True)
        pump.start()

        return {"ok": True, "pid": process.pid}

    def _is_permitted_interpreter(self, interpreter: str) -> bool:
        permitted = {candidate["path"] for candidate in self.paths.discover_interpreters()}
        return interpreter in permitted

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
        environment["COLUMNS"]          = str(self.TERMINAL_COLUMNS)
        environment["LINES"]            = "32"

        library_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(interpreter))), "lib")
        library_path      = environment.get("LD_LIBRARY_PATH", "")
        if library_directory not in library_path.split(":"):
            environment["LD_LIBRARY_PATH"] = library_directory + (":" + library_path if library_path else "")

        return environment

    def _pump(self, pid: int, process: subprocess.Popen, stream: JobStream) -> None:
        descriptor = process.stdout.fileno()
        decoder    = codecs.getincrementaldecoder("utf-8")("replace")

        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                stream.publish({"type": "chunk", "data": text})

        tail = decoder.decode(b"", final=True)
        if tail:
            stream.publish({"type": "chunk", "data": tail})

        process.wait()
        code = process.returncode

        with self.lock:
            record = self.processes.get(pid)
            if record is not None:
                record["status"]    = "finished" if code == 0 else "failed"
                record["exit_code"] = code
            status = record["status"] if record is not None else "finished"

        self.logger.info(f"process {pid} exited with code {code}")
        stream.publish({"type": "status", "status": status, "code": code, "verdict": "ok" if code == 0 else "error"})
        stream.publish({"type": "end"})

    def _view(self, record: dict) -> dict:
        return {
            "pid"         : record["pid"],
            "script"      : record["script"],
            "command"     : record["command"],
            "interpreter" : record["interpreter"],
            "status"      : record["status"],
            "started"     : record["started"],
            "exit_code"   : record["exit_code"],
        }

    def list(self) -> list[dict]:
        with self.lock:
            records = list(self.processes.values())
        return [self._view(record) for record in sorted(records, key=lambda item: item["started"], reverse=True)]

    def get_stream(self, pid: int) -> JobStream | None:
        with self.lock:
            return self.streams.get(pid)

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

    def stop_all(self, grace: float = 6.0) -> None:
        with self.lock:
            running = [record for record in self.processes.values() if record["status"] == "running"]

        for record in running:
            self._signal_group(record["pid"], signal.SIGTERM)
            self.logger.warning(f"shutdown stop for pid {record['pid']}")

        deadline = time.monotonic() + grace
        for record in running:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                record["process"].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                self._signal_group(record["pid"], signal.SIGKILL)
                self.logger.warning(f"escalated to SIGKILL for pid {record['pid']}")
