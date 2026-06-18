from __future__ import annotations

import getpass
import os
import socket
import time

import psutil

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class SystemMonitor:

    PROCESS_LIMIT = 12

    def __init__(self, paths) -> None:
        self.paths      = paths
        self.host       = socket.gethostname()
        self.user       = self._resolve_user()
        self.boot_time  = psutil.boot_time()
        self.nvml_ready = False

        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        for process in psutil.process_iter(["cpu_percent"]):
            pass

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_ready = True
            except pynvml.NVMLError:
                self.nvml_ready = False

    def _resolve_user(self) -> str:
        try:
            return getpass.getuser()
        except Exception:
            return os.environ.get("USER", "user")

    def _uptime(self) -> float:
        return max(0.0, time.time() - self.boot_time)

    def _cpu(self) -> dict:
        try:
            load = list(psutil.getloadavg())
        except (AttributeError, OSError):
            load = [0.0, 0.0, 0.0]

        cores = psutil.cpu_percent(interval=None, percpu=True)
        return {
            "total"    : psutil.cpu_percent(interval=None),
            "count"    : psutil.cpu_count(logical=True),
            "load"     : load,
            "cores"    : cores,
            "per_core" : cores,
        }

    def _memory(self) -> dict:
        virtual = psutil.virtual_memory()
        swap    = psutil.swap_memory()
        return {
            "total"      : virtual.total,
            "available"  : virtual.available,
            "used"       : virtual.used,
            "percent"    : virtual.percent,
            "swap_total" : swap.total,
            "swap_free"  : swap.free,
        }

    def _disk(self) -> dict:
        usage = psutil.disk_usage(str(self.paths.repo_root))
        return {
            "path"    : str(self.paths.repo_root),
            "used"    : usage.used,
            "total"   : usage.total,
            "free"    : usage.free,
            "percent" : usage.percent,
        }

    def _gpus(self) -> list[dict]:
        if not self.nvml_ready:
            return []

        devices = []
        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            return []

        for index in range(device_count):
            try:
                handle      = pynvml.nvmlDeviceGetHandleByIndex(index)
                name        = pynvml.nvmlDeviceGetName(handle)
                memory      = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                continue

            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")

            devices.append({
                "index"     : index,
                "name"      : name,
                "mem_used"  : memory.used,
                "mem_total" : memory.total,
                "util"      : utilization.gpu,
                "temp"      : temperature,
            })

        return devices

    def _processes(self) -> list[dict]:
        rows = []
        for process in psutil.process_iter(["pid", "name", "username", "cpu_percent", "memory_info", "status", "cmdline"]):
            try:
                info = process.info
                if info.get("username") not in (self.user, None) and not str(info.get("username", "")).endswith(self.user):
                    continue

                memory  = info.get("memory_info")
                cmdline = info.get("cmdline") or []
                command = " ".join(cmdline) if cmdline else (info.get("name") or "?")

                rows.append({
                    "pid"   : info.get("pid"),
                    "cpu"   : float(info.get("cpu_percent") or 0.0),
                    "rss"   : int(memory.rss) if memory else 0,
                    "state" : (info.get("status") or "?")[:1].upper(),
                    "cmd"   : command,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        rows.sort(key=lambda row: (row["cpu"], row["rss"]), reverse=True)
        return rows[: self.PROCESS_LIMIT]

    def status(self) -> dict:
        memory = self._memory()
        return {
            "host"    : self.host,
            "user"    : self.user,
            "uptime"  : self._uptime(),
            "cpu"     : self._cpu(),
            "mem"     : memory,
            "ram"     : memory,
            "disk"    : self._disk(),
            "gpus"    : self._gpus(),
            "procs"   : self._processes(),
        }
