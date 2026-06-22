from __future__ import annotations

import numpy as np


class SweepAxis:
    def __init__(self, name, axis_config):
        self.name    = name
        self.minimum = float(axis_config.minimum)
        self.maximum = float(axis_config.maximum)
        self.step    = float(axis_config.step)

    def _validate(self):
        if self.step <= 0.0:
            raise ValueError(f"Sweep axis '{self.name}' requires a positive step, received {self.step}.")
        if self.maximum < self.minimum:
            raise ValueError(f"Sweep axis '{self.name}' has maximum {self.maximum} below minimum {self.minimum}.")

    def values(self):
        self._validate()

        count  = int(round((self.maximum - self.minimum) / self.step)) + 1
        points = self.minimum + self.step * np.arange(count, dtype=np.float64)
        return np.round(points, 10)


class SweepGrid:
    def __init__(self, scale_axis_config, efficiency_axis_config):
        self.scale_axis      = SweepAxis("scale_factor", scale_axis_config)
        self.efficiency_axis = SweepAxis("detection_efficiency", efficiency_axis_config)

    def scale_values(self):
        return self.scale_axis.values()

    def efficiency_values(self):
        scale       = self.scale_values()
        efficiency  = self.efficiency_axis.values()

        out_of_range = efficiency[(efficiency <= 0.0) | (efficiency > 1.0)]
        if out_of_range.size > 0:
            raise ValueError(f"Detection efficiency must lie in (0, 1]; offending values {out_of_range.tolist()}.")
        if np.any(scale < 0.0):
            raise ValueError(f"Light scale factor must be non-negative; offending values {scale[scale < 0.0].tolist()}.")
        return efficiency

    def cells(self):
        cells = []
        for efficiency in self.efficiency_values():
            for scale in self.scale_values():
                cells.append((float(scale), float(efficiency)))
        return cells

    def shape(self):
        return len(self.efficiency_values()), len(self.scale_values())
