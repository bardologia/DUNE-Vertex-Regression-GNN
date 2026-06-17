from __future__ import annotations

import threading
import time
from datetime import datetime

from server_logger  import ServerLogger
from system_monitor import SystemMonitor


class GpuWatchdog:

    INTERVAL_SECONDS = 4.0

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
        self.logger.info(f"gpu watchdog armed at {self.INTERVAL_SECONDS:.0f}s interval")

    def _poll(self) -> None:
        while True:
            try:
                gpus = self.system._gpus()
                with self.lock:
                    self.snapshot = {"updated": datetime.now().isoformat(timespec="seconds"), "gpus": gpus}
            except Exception as error:
                self.logger.error(f"gpu watchdog error: {error}")
            time.sleep(self.INTERVAL_SECONDS)

    def latest(self) -> dict:
        with self.lock:
            return {"armed": self.armed, **self.snapshot}
