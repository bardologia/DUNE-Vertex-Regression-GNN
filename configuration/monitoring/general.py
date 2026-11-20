from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceMonitorSettings:
    tensorboard_scalars     : tuple = ("ram_pct", "proc_rss_gb", "swap_pct", "shm_pct", "cpu_pct", "proc_cpu_pct", "disk_read_mb_s", "disk_write_mb_s")
    tensorboard_gpu_scalars : tuple = ("util_pct", "vram_pct", "temp_c", "power_w")

    milliwatts_per_watt  : float = 1000.0
    min_disk_interval_s  : float = 1e-6
    join_interval_factor : int   = 2
    min_join_timeout_s   : float = 5.0
