from __future__ import annotations

import torch

from configuration.data.general import CoordinateAxes


class CoordinateErrors:
    def __init__(self):
        self.absolute_sum = torch.zeros(len(CoordinateAxes.names), dtype=torch.float64)
        self.squared_sum  = torch.zeros(len(CoordinateAxes.names), dtype=torch.float64)
        self.count        = torch.zeros(len(CoordinateAxes.names), dtype=torch.float64)

    def update(self, difference, scored):
        residual = difference.detach().to(device="cpu", dtype=torch.float64)
        weights  = scored.detach().to(device="cpu", dtype=torch.float64)

        self.absolute_sum += (residual.abs() * weights).sum(dim=0)
        self.squared_sum  += (residual ** 2 * weights).sum(dim=0)
        self.count        += weights.sum(dim=0)

        return self

    def mean_absolute(self):
        return self.absolute_sum / self.count

    def root_mean_squared(self):
        return torch.sqrt(self.squared_sum / self.count)

    def results(self):
        mean_absolute     = self.mean_absolute()
        root_mean_squared = self.root_mean_squared()

        results = {}
        for index, name in enumerate(CoordinateAxes.names):
            results[f"mae_{name}"]  = float(mean_absolute[index].item())
            results[f"rmse_{name}"] = float(root_mean_squared[index].item())

        results["rms_3d"] = float(torch.sqrt((self.squared_sum / self.count).sum()).item())

        return results
