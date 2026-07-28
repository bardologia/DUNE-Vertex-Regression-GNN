from __future__ import annotations

import torch


class CoordinateErrors:
    COORDINATE_NAMES = ("x", "y", "z")
    METRIC_NAMES     = ("mae", "rmse")

    def __init__(self):
        self.absolute_sum = torch.zeros(len(self.COORDINATE_NAMES), dtype=torch.float64)
        self.squared_sum  = torch.zeros(len(self.COORDINATE_NAMES), dtype=torch.float64)
        self.count        = 0

    def update(self, difference):
        residual = difference.detach().to(device="cpu", dtype=torch.float64)

        self.absolute_sum += residual.abs().sum(dim=0)
        self.squared_sum  += (residual ** 2).sum(dim=0)
        self.count        += int(residual.shape[0])

        return self

    def mean_absolute(self):
        return self.absolute_sum / max(1, self.count)

    def root_mean_squared(self):
        return torch.sqrt(self.squared_sum / max(1, self.count))

    def results(self):
        mean_absolute     = self.mean_absolute()
        root_mean_squared = self.root_mean_squared()

        results = {}
        for index, name in enumerate(self.COORDINATE_NAMES):
            results[f"mae_{name}"]  = float(mean_absolute[index].item())
            results[f"rmse_{name}"] = float(root_mean_squared[index].item())

        return results
