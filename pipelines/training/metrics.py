from __future__ import annotations

import numpy as np
import torch


class Metrics:
    COORDINATE_NAMES = ("x", "y", "z")

    def __init__(self, verbose, logger, tracker):
        self.verbose = verbose
        self.logger  = logger
        self.tracker = tracker

    def _per_coordinate(self, predictions, targets, difference, absolute_difference):
        results = {}
        for index, name in enumerate(self.COORDINATE_NAMES):
            residual_sum_of_squares = torch.sum(difference[:, index] ** 2).item()
            total_sum_of_squares    = torch.sum((targets[:, index] - targets[:, index].mean()) ** 2).item()
            r_squared               = 1 - (residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else 0.0

            results[f"mae_{name}"]    = float(absolute_difference[:, index].mean().item())
            results[f"rmse_{name}"]   = float(torch.sqrt((difference[:, index] ** 2).mean()).item())
            results[f"median_{name}"] = float(absolute_difference[:, index].median().item())
            results[f"r2_{name}"]     = float(r_squared)

        return results

    def _euclidean(self, difference):
        distances = torch.sqrt((difference ** 2).sum(dim=1))
        return {
            "euclidean_mean"   : float(distances.mean().item()),
            "euclidean_std"    : float(distances.std(correction=0).item()),
            "euclidean_median" : float(distances.median().item()),
            "euclidean_max"    : float(distances.max().item()),
            "euclidean_min"    : float(distances.min().item()),
        }

    def _display(self, epoch, stage, results):
        rows = []
        for name in self.COORDINATE_NAMES:
            rows.append({
                "Coord"  : name,
                "MAE"    : results[f"mae_{name}"],
                "RMSE"   : results[f"rmse_{name}"],
                "Median" : results[f"median_{name}"],
                "R2"     : results[f"r2_{name}"],
            })
        rows.append({"Coord": "overall", "MAE": results["mae"], "RMSE": results["rmse"], "Median": results["median_error"], "R2": results["r2"]})
        self.logger.metrics_table(rows, ["Coord", "MAE", "RMSE", "Median", "R2"], title=f"{stage} metrics (epoch {epoch + 1})")

    def calculate(self, epoch, predictions, targets, stage="validation"):
        difference          = predictions - targets
        absolute_difference = torch.abs(difference)

        mean_absolute_error      = absolute_difference.mean().item()
        root_mean_squared_error  = float(np.sqrt((difference ** 2).mean().item()))

        residual_sum_of_squares = torch.sum(difference ** 2).item()
        total_sum_of_squares    = torch.sum((targets - targets.mean()) ** 2).item()
        r_squared               = 1 - (residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else 0.0

        results = {
            "mae"          : float(mean_absolute_error),
            "rmse"         : float(root_mean_squared_error),
            "r2"           : float(r_squared),
            "median_error" : float(absolute_difference.median().item()),
            "std_error"    : float(absolute_difference.std().item()),
        }
        results.update(self._per_coordinate(predictions, targets, difference, absolute_difference))
        results.update(self._euclidean(difference))

        self.tracker.log_metrics(stage, results, step=epoch)
        if self.verbose:
            self._display(epoch, stage, results)

        return results
