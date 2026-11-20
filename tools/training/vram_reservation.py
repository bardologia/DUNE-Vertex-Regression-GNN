from __future__ import annotations

import torch

from configuration.shared import MeasurementUnits
from configuration.tools  import VramReservationSettings


class VramReservation:

    def __init__(self, enabled: bool, keep_free_gb: float, device, logger) -> None:
        self.enabled         = bool(enabled) and torch.device(device).type == "cuda"
        self.keep_free_bytes = int(float(keep_free_gb) * MeasurementUnits.bytes_per_gib)
        self.device          = torch.device(device)
        self.logger          = logger
        self.filled          = False

    def _free_bytes(self) -> int:
        free, _total = torch.cuda.mem_get_info(self.device)
        return free

    def _claim(self) -> int:
        chunks  = []
        claimed = 0
        shrink  = 1

        while True:
            spare = self._free_bytes() - self.keep_free_bytes
            if spare < VramReservationSettings.chunk_floor_bytes:
                break

            chunk = (spare // shrink // VramReservationSettings.granularity_bytes) * VramReservationSettings.granularity_bytes
            if chunk < VramReservationSettings.chunk_floor_bytes:
                break

            try:
                chunks.append(torch.empty(chunk, dtype=torch.uint8, device=self.device))
                claimed += chunk
                shrink   = 1
            except torch.cuda.OutOfMemoryError:
                shrink *= 2

        del chunks
        return claimed

    def fill(self) -> None:
        if not self.enabled:
            return

        torch.cuda.synchronize(self.device)

        claimed     = self._claim()
        self.filled = True

        free, total = torch.cuda.mem_get_info(self.device)
        self.logger.subsection(f"VRAM reservation: parked {claimed / MeasurementUnits.bytes_per_gib:.2f} GB in the allocator cache, {free / MeasurementUnits.bytes_per_gib:.2f} GB of {total / MeasurementUnits.bytes_per_gib:.2f} GB left free (keep-free target {self.keep_free_bytes / MeasurementUnits.bytes_per_gib:.2f} GB)")

    def refill(self) -> None:
        if not (self.enabled and self.filled):
            return

        self._claim()
