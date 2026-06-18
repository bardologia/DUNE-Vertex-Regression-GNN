from __future__ import annotations


class CoordinateTransform:

    def apply_to_target(self, target_x: float, target_y: float, target_z: float):
        sensor_x = target_y
        sensor_y = -target_x
        sensor_z = target_z
        return sensor_x, sensor_y, sensor_z


__all__ = [
    "CoordinateTransform",
]
