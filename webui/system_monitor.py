from __future__ import annotations

import socket

import psutil

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class SystemMonitor:

    def __init__(self, paths) -> None:
        self.paths              = paths
        self.host               = socket.gethostname()
        self.nvml_ready         = False

        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_ready = True
            except pynvml.NVMLError:
                self.nvml_ready = False

    def _cpu(self) -> dict:
        return {
            "percent"   : psutil.cpu_percent(interval=None),
            "per_core"  : psutil.cpu_percent(interval=None, percpu=True),
            "count"     : psutil.cpu_count(logical=True),
        }

    def _memory(self) -> dict:
        virtual = psutil.virtual_memory()
        return {
            "used"    : virtual.used,
            "total"   : virtual.total,
            "percent" : virtual.percent,
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

    def status(self) -> dict:
        return {
            "host" : self.host,
            "cpu"  : self._cpu(),
            "ram"  : self._memory(),
            "disk" : self._disk(),
            "gpus" : self._gpus(),
        }
