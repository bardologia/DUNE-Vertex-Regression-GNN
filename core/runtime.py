from __future__ import annotations

import shutil
import subprocess
from typing import Iterable

import ray


class RuntimeGuard:
    RAY_PROCESS_MARKERS = (
        "raylet",
        "gcs_server",
        "ray::",
        "plasma_store",
        "monitor.py",
        "dashboard.py",
    )

    @classmethod
    def _list_processes(cls) -> list[str]:
        completed_process = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed_process.returncode != 0:
            return []

        return [line.strip() for line in completed_process.stdout.splitlines() if line.strip()]

    @classmethod
    def _filter_processes(cls, process_lines: Iterable[str], markers: tuple[str, ...]) -> list[str]:
        return [
            process_line
            for process_line in process_lines
            if any(marker in process_line for marker in markers)
        ]

    @classmethod
    def cleanup_stale_ray_processes(cls, logger=None) -> list[str]:
        process_lines = cls._list_processes()
        stale_ray_processes = cls._filter_processes(process_lines, cls.RAY_PROCESS_MARKERS)

        if logger is not None:
            logger.section("[Startup Process Check]")
            if stale_ray_processes:
                logger.warning(f"Detected {len(stale_ray_processes)} stale Ray process(es) from a previous run.")
                for stale_process in stale_ray_processes:
                    logger.subsection(stale_process)
            else:
                logger.subsection("No stale Ray processes detected.")

        if ray.is_initialized():
            ray.shutdown()
            if logger is not None:
                logger.subsection("Shutdown active in-process Ray runtime.")

        ray_cli_path = shutil.which("ray")
        if ray_cli_path is not None:
            completed_process = subprocess.run(
                [ray_cli_path, "stop", "--force"],
                capture_output=True,
                text=True,
                check=False,
            )
            if logger is not None:
                command_output = completed_process.stdout.strip() or completed_process.stderr.strip()
                if command_output:
                    logger.subsection(command_output)
        elif logger is not None and stale_ray_processes:
            logger.warning("Ray CLI not found; stale Ray processes could not be force-stopped.")

        return stale_ray_processes


__all__ = ["RuntimeGuard"]