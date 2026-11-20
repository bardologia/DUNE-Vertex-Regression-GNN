from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib  import Path

from configuration.shared        import IoNames
from configuration.webui.general import LaunchSettings

from project_paths import ProjectPaths
from server_logger import ServerLogger


class SubprocessRunner:

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths  = paths
        self.logger = logger

    def spawn(self, script: Path, argv_tail: list, log_stem: str, output: Path) -> None:
        interpreter = self.paths.preferred_interpreter()
        argv        = [interpreter, "-u", str(script), *argv_tail]

        self.paths.webui_logs_dir.mkdir(parents=True, exist_ok=True)
        stamp    = datetime.now().strftime(IoNames.run_tag_format)
        log_path = self.paths.webui_logs_dir / f"{log_stem}_{stamp}{IoNames.output_log_suffix}"

        with open(log_path, "ab") as sink:
            process = subprocess.run(
                argv,
                cwd    = str(self.paths.repo_root),
                stdout = sink,
                stderr = subprocess.STDOUT,
                stdin  = subprocess.DEVNULL,
                env    = self.environment(interpreter),
            )

        if process.returncode != 0:
            raise RuntimeError(f"{script.name} exited with code {process.returncode}; see {log_path}")
        if not output.is_file():
            raise FileNotFoundError(f"export finished but {output} was not produced; check the export log")

    @staticmethod
    def environment(interpreter: str) -> dict:
        environment                     = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"

        library_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(interpreter))), IoNames.interpreter_lib_dir)
        library_path      = environment.get("LD_LIBRARY_PATH", "")
        if library_directory not in library_path.split(":"):
            environment["LD_LIBRARY_PATH"] = library_directory + (":" + library_path if library_path else "")

        return environment


class SourceIntrospection:

    def __init__(self, paths: ProjectPaths, label: str) -> None:
        self.paths = paths
        self.label = label

    def signature(self, watched: list[Path]) -> tuple:
        stamps = []
        for path in watched:
            try:
                stamps.append((path.name, path.stat().st_mtime_ns))
            except OSError:
                continue
        return tuple(stamps)

    def run(self, argv: list) -> dict:
        try:
            completed = subprocess.run(argv, cwd=str(self.paths.repo_root), capture_output=True, text=True, timeout=LaunchSettings.introspection_timeout_s)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"ok": False, "error": f"{self.label} introspection failed: {error}"}

        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.strip().splitlines()[-LaunchSettings.introspection_error_lines:])
            return {"ok": False, "error": f"{self.label} introspection failed:\n{tail}"}

        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": f"{self.label} introspection produced no output"}

        return {"ok": True, "payload": payload}
