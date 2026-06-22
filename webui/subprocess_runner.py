from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib  import Path

from project_paths import ProjectPaths
from server_logger import ServerLogger


class SubprocessRunner:

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths  = paths
        self.logger = logger

    def _environment(self, interpreter: str) -> dict:
        environment                     = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"

        library_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(interpreter))), "lib")
        library_path      = environment.get("LD_LIBRARY_PATH", "")
        if library_directory not in library_path.split(":"):
            environment["LD_LIBRARY_PATH"] = library_directory + (":" + library_path if library_path else "")

        return environment

    def spawn(self, script: Path, argv_tail: list, log_stem: str) -> None:
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
