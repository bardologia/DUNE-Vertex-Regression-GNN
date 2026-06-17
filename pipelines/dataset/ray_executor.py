from __future__ import annotations

import os
from collections import deque
from pathlib     import Path

import numpy as np
import ray


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "0")


class RayExecutor:
    def __init__(self, logger, ray_config):
        self.logger     = logger
        self.ray_config = ray_config

    def ensure_initialized(self) -> None:
        if ray.is_initialized():
            return

        ray.init(
            ignore_reinit_error = True,
            include_dashboard   = False,
            log_to_driver       = False,
            logging_level       = "ERROR",
            num_cpus            = self.ray_config.num_cpus,
            runtime_env         = {
                "working_dir" : str(PROJECT_ROOT),
                "excludes"    : ["data", "datasets", "runs", "tuning", "logs", ".git", "test_data", "**/__pycache__", "*.pt", "*.csv", "*.npz"],
            },
        )

    def shutdown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()

    def cpu_count(self) -> int:
        configured_cpus = self.ray_config.num_cpus

        if configured_cpus is None:
            if ray.is_initialized():
                configured_cpus = max(1, int(ray.available_resources().get("CPU", 1)))
            else:
                configured_cpus = max(1, os.cpu_count() or 1)

        return max(1, int(configured_cpus))

    def batch_size(self, total_items: int, stage: str | None = None) -> int:
        target_tasks_per_cpu = self.ray_config.batch_tasks_per_cpu
        max_batch_size       = self.ray_config.max_batch_size

        if stage == "scaling":
            target_tasks_per_cpu = self.ray_config.scaling_batch_tasks_per_cpu
            max_batch_size       = self.ray_config.scaling_max_batch_size

        target_tasks = max(1, self.cpu_count() * int(target_tasks_per_cpu))
        batch_size   = max(1, int(np.ceil(total_items / target_tasks)))

        return min(batch_size, max(1, int(max_batch_size)))

    def max_inflight(self, total_batches: int, stage: str | None = None) -> int:
        inflight_per_cpu = self.ray_config.max_inflight_tasks_per_cpu

        if stage == "scaling":
            inflight_per_cpu = self.ray_config.scaling_max_inflight_tasks_per_cpu

        return max(1, min(total_batches, self.cpu_count() * int(inflight_per_cpu)))

    def make_batches(self, items: list, stage: str | None = None) -> list:
        size = self.batch_size(len(items), stage=stage)
        return [items[index:index + size] for index in range(0, len(items), size)]

    def run_batches(self, remote_function, batch_arguments: list, description: str, stage: str | None = None) -> list:
        total_batches = len(batch_arguments)
        if total_batches == 0:
            return []

        pending_arguments = deque(enumerate(batch_arguments))
        in_flight         = {}
        completed_results = {}
        maximum_inflight  = self.max_inflight(total_batches, stage=stage)

        while pending_arguments and len(in_flight) < maximum_inflight:
            batch_index, batch_args = pending_arguments.popleft()
            in_flight[remote_function.remote(*batch_args)] = batch_index

        with self.logger.track() as progress:
            task_id = progress.add_task(description, total=total_batches)

            while in_flight:
                ready_references, _ = ray.wait(list(in_flight.keys()), num_returns=1)
                ready_reference     = ready_references[0]
                batch_index         = in_flight.pop(ready_reference)
                completed_results[batch_index] = ray.get(ready_reference)
                progress.advance(task_id)

                if pending_arguments:
                    next_index, next_args = pending_arguments.popleft()
                    in_flight[remote_function.remote(*next_args)] = next_index

        return [completed_results[index] for index in range(total_batches)]
