from __future__ import annotations

import getpass
import socket
import threading
import time
from datetime import datetime

import psutil
import pynvml

from configuration.webui.general import SystemMonitorSettings

from server_logger import ServerLogger


class SystemMonitor:

    def __init__(self, paths) -> None:
        self.paths      = paths
        self.host       = socket.gethostname()
        self.user       = getpass.getuser()
        self.boot_time  = psutil.boot_time()
        self.nvml_ready = False

        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        for process in psutil.process_iter(["cpu_percent"]):
            pass

        try:
            pynvml.nvmlInit()
            self.nvml_ready = True
        except pynvml.NVMLError:
            self.nvml_ready = False

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

    def _uptime(self) -> float:
        return max(0.0, time.time() - self.boot_time)

    def _cpu(self) -> dict:
        load  = list(psutil.getloadavg())
        cores = psutil.cpu_percent(interval=None, percpu=True)
        return {
            "total"    : psutil.cpu_percent(interval=None),
            "count"    : psutil.cpu_count(logical=True),
            "load"     : load,
            "cores"    : cores,
            "per_core" : cores,
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

    def gpus(self) -> list[dict]:
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
            info = process.info
            if info["username"] not in (self.user, None) and not str(info["username"]).endswith(self.user):
                continue

            memory  = info["memory_info"]
            cmdline = info["cmdline"] or []
            command = " ".join(cmdline) if cmdline else (info["name"] or "?")

            rows.append({
                "pid"   : info["pid"],
                "cpu"   : float(info["cpu_percent"] or 0.0),
                "rss"   : int(memory.rss) if memory else 0,
                "state" : (info["status"] or "?")[:1].upper(),
                "cmd"   : command,
            })

        rows.sort(key=lambda row: (row["cpu"], row["rss"]), reverse=True)
        return rows[:SystemMonitorSettings.process_limit]

    def status(self) -> dict:
        memory = self._memory()
        return {
            "host"   : self.host,
            "user"   : self.user,
            "uptime" : self._uptime(),
            "cpu"    : self._cpu(),
            "mem"    : memory,
            "ram"    : memory,
            "disk"   : self._disk(),
            "gpus"   : self.gpus(),
            "procs"  : self._processes(),
        }


class GpuWatchdog:

    def __init__(self, system: SystemMonitor, logger: ServerLogger) -> None:
        self.system   = system
        self.logger   = logger
        self.lock     = threading.Lock()
        self.armed    = False
        self.snapshot = {"updated": None, "gpus": []}

    def start(self) -> None:
        if not self.system.nvml_ready:
            self.logger.info("gpu watchdog idle, no nvml devices available")
            return

        worker = threading.Thread(target=self._poll, daemon=True)
        worker.start()
        self.armed = True
        self.logger.info(f"gpu watchdog armed at {SystemMonitorSettings.gpu_poll_interval_s:.0f}s interval")

    def _poll(self) -> None:
        while True:
            try:
                gpus = self.system.gpus()
                with self.lock:
                    self.snapshot = {"updated": datetime.now().isoformat(timespec="seconds"), "gpus": gpus}
            except Exception as error:
                self.logger.error(f"gpu watchdog error: {error}")
            time.sleep(SystemMonitorSettings.gpu_poll_interval_s)

    def latest(self) -> dict:
        with self.lock:
            return {"armed": self.armed, **self.snapshot}
