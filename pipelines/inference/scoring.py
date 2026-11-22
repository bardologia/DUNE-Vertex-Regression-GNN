from __future__ import annotations

import numpy as np
import torch

from configuration.data.general         import CoordinateAxes
from configuration.inference.general    import ErrorQuantiles
from pipelines.shared.coordinate_errors import CoordinateErrors


class Predictor:
    def __init__(self, model, checkpoint_path, stats, device, logger):
        self.model           = model.to(device)
        self.checkpoint_path = checkpoint_path
        self.stats           = stats
        self.device          = device
        self.logger          = logger

    def prepare(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["params"])
        self.model.eval()
        self.model.requires_grad_(False)
        self.logger.section("[Checkpoint]")
        self.logger.kv_table({
            "Path"   : str(self.checkpoint_path),
            "Epoch"  : checkpoint["epoch"],
            "Device" : self.device,
        })
        return self

    def _denormalize(self, target):
        return self.stats.target.inverse_torch(target, self.device)

    def predict(self, loader):
        prediction_segments = []
        target_segments     = []
        scored_segments     = []

        with torch.no_grad():
            for data in loader:
                data        = data.to(self.device, non_blocking=True)
                predictions = self.model(data)
                prediction_segments.append(self._denormalize(predictions).cpu())
                target_segments.append(self._denormalize(data.y).cpu())
                scored_segments.append(data.scored.bool().cpu())

        predictions = torch.cat(prediction_segments)
        targets     = torch.cat(target_segments)
        scored      = torch.cat(scored_segments)
        self.logger.subsection(f"Predicted {len(predictions)} events")
        return predictions, targets, scored


class InferenceMetrics:
    def __init__(self, logger, tracker=None):
        self.logger  = logger
        self.tracker = tracker

    def _per_coordinate(self, difference, targets, scored):
        results = CoordinateErrors().update(difference, scored).results()

        for index, name in enumerate(CoordinateAxes.names):
            rows     = scored[:, index]
            residual = difference[rows, index]
            absolute = residual.abs()
            truth    = targets[rows, index]

            residual_sum_of_squares = torch.sum(residual ** 2).item()
            total_sum_of_squares    = torch.sum((truth - truth.mean()) ** 2).item()
            r_squared               = 1 - (residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else float("nan")

            results[f"count_{name}"]  = int(rows.sum().item())
            results[f"median_{name}"] = float(absolute.median().item())
            results[f"std_{name}"]    = float(absolute.std(correction=0).item())
            results[f"bias_{name}"]   = float(residual.mean().item())
            results[f"p90_{name}"]    = float(torch.quantile(absolute, ErrorQuantiles.p90).item())
            results[f"r2_{name}"]     = float(r_squared)

        return results

    def display(self, stage, results):
        rows = []
        for name in CoordinateAxes.names:
            rows.append({
                "Coord"  : name,
                "MAE"    : results[f"mae_{name}"],
                "RMSE"   : results[f"rmse_{name}"],
                "Median" : results[f"median_{name}"],
                "Bias"   : results[f"bias_{name}"],
                "R2"     : results[f"r2_{name}"],
            })
        rows.append({"Coord": "overall", "MAE": results["mae"], "RMSE": results["rmse"], "Median": results["median_error"], "Bias": results["bias"], "R2": results["r2"]})
        self.logger.metrics_table(rows, ["Coord", "MAE", "RMSE", "Median", "Bias", "R2"], title=f"{stage} metrics (each axis scored on its held-out coordinates)")
        self.logger.subsection(f"{stage} 3D RMS error: {results['rms_3d']:.4f} m")

    def compute(self, predictions, targets, scored, stage="test"):
        difference = predictions - targets
        pooled     = difference[scored]
        centred    = (targets - (targets * scored).sum(dim=0, keepdim=True) / scored.sum(dim=0, keepdim=True))[scored]

        residual_sum_of_squares = torch.sum(pooled ** 2).item()
        total_sum_of_squares    = torch.sum(centred ** 2).item()
        r_squared               = 1 - (residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares > 0 else float("nan")

        results = {
            "count"        : int(predictions.shape[0]),
            "mae"          : float(pooled.abs().mean().item()),
            "rmse"         : float(np.sqrt((pooled ** 2).mean().item())),
            "r2"           : float(r_squared),
            "median_error" : float(pooled.abs().median().item()),
            "bias"         : float(pooled.mean().item()),
        }
        results.update(self._per_coordinate(difference, targets, scored))

        if self.tracker is not None:
            self.tracker.log_metrics(f"inference_{stage}", results, step=0)

        self.display(stage, results)
        return results
