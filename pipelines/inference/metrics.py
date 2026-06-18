from __future__ import annotations

import numpy as np
import torch


class InferenceMetrics:
    COORDINATE_NAMES = ("x", "y", "z")

    def __init__(self, logger, tracker=None):
        self.logger  = logger
        self.tracker = tracker

    def _per_coordinate(self, difference, absolute_difference, targets):
        results = {}
        for index, name in enumerate(self.COORDINATE_NAMES):
            residual_sum_of_squares = torch.sum(difference[:, index] ** 2).item()
            total_sum_of_squares    = torch.sum((targets[:, index] - targets[:, index].mean()) ** 2).item()
            r_squared               = 1 - (residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else float("nan")

            results[f"mae_{name}"]    = float(absolute_difference[:, index].mean().item())
            results[f"rmse_{name}"]   = float(torch.sqrt((difference[:, index] ** 2).mean()).item())
            results[f"median_{name}"] = float(absolute_difference[:, index].median().item())
            results[f"std_{name}"]    = float(absolute_difference[:, index].std(correction=0).item())
            results[f"bias_{name}"]   = float(difference[:, index].mean().item())
            results[f"p90_{name}"]    = float(torch.quantile(absolute_difference[:, index], 0.90).item())
            results[f"r2_{name}"]     = float(r_squared)

        return results

    def _euclidean(self, difference):
        distances = torch.sqrt((difference ** 2).sum(dim=1))
        return {
            "euclidean_mean"   : float(distances.mean().item()),
            "euclidean_std"    : float(distances.std(correction=0).item()),
            "euclidean_median" : float(distances.median().item()),
            "euclidean_p90"    : float(torch.quantile(distances, 0.90).item()),
            "euclidean_p95"    : float(torch.quantile(distances, 0.95).item()),
            "euclidean_max"    : float(distances.max().item()),
            "euclidean_min"    : float(distances.min().item()),
        }

    def display(self, stage, results):
        rows = []
        for name in self.COORDINATE_NAMES:
            rows.append({
                "Coord"  : name,
                "MAE"    : results[f"mae_{name}"],
                "RMSE"   : results[f"rmse_{name}"],
                "Median" : results[f"median_{name}"],
                "Bias"   : results[f"bias_{name}"],
                "R2"     : results[f"r2_{name}"],
            })
        rows.append({"Coord": "overall", "MAE": results["mae"], "RMSE": results["rmse"], "Median": results["median_error"], "Bias": results["bias"], "R2": results["r2"]})
        self.logger.metrics_table(rows, ["Coord", "MAE", "RMSE", "Median", "Bias", "R2"], title=f"{stage} metrics")

    def compute(self, predictions, targets, stage="test"):
        difference          = predictions - targets
        absolute_difference = torch.abs(difference)

        residual_sum_of_squares = torch.sum(difference ** 2).item()
        total_sum_of_squares    = torch.sum((targets - targets.mean(dim=0, keepdim=True)) ** 2).item()
        r_squared               = 1 - (residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else float("nan")

        results = {
            "count"        : int(predictions.shape[0]),
            "mae"          : float(absolute_difference.mean().item()),
            "rmse"         : float(np.sqrt((difference ** 2).mean().item())),
            "r2"           : float(r_squared),
            "median_error" : float(absolute_difference.median().item()),
            "bias"         : float(difference.mean().item()),
        }
        results.update(self._per_coordinate(difference, absolute_difference, targets))
        results.update(self._euclidean(difference))

        if self.tracker is not None:
            self.tracker.log_metrics(f"inference_{stage}", results, step=0)

        self.display(stage, results)
        return results
