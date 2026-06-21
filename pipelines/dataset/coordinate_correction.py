from __future__ import annotations

import numpy as np


class CoordinateTransform:

    def apply_to_target(self, target_x: float, target_y: float, target_z: float):
        sensor_x = target_y
        sensor_y = -target_x
        sensor_z = target_z
        return sensor_x, sensor_y, sensor_z


class EndcapFaceCorrector:

    def __init__(self, logger=None, tolerance: float = 0.05):
        self.logger    = logger
        self.tolerance = tolerance

    def _face_mask(self, z_values):
        extreme   = float(np.max(np.abs(z_values)))
        face_mask = np.abs(np.abs(z_values) - extreme) <= self.tolerance
        return face_mask, extreme

    def apply(self, geometry_frame):
        z_values           = geometry_frame["z"].to_numpy(dtype=np.float64)
        face_mask, extreme = self._face_mask(z_values)

        corrected_frame                    = geometry_frame.copy()
        corrected_frame.loc[face_mask, "z"] = -z_values[face_mask]

        if self.logger is not None:
            self.logger.subsection(f"Endcap-face z correction: flipped {int(face_mask.sum())} channels at |z|={extreme:.3f}")
        return corrected_frame


__all__ = [
    "CoordinateTransform",
    "EndcapFaceCorrector",
]
