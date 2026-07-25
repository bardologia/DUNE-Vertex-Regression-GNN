from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from configuration.data.general import DataConfig
from tools.monitoring.logger    import Logger


class LightCentroidCheck:
    AXES         = ("x", "y", "z")
    FACE_MARGIN  = 0.5
    TABLE_FIELDS = ["Axis", "Pearson r", "Fit slope", "Fit intercept", "MAE raw", "MAE fitted", "Target std"]

    def __init__(self, store_directory, logger):
        self.store_directory = Path(store_directory)
        self.logger          = logger

        self.positions = None
        self.light     = None
        self.targets   = None

    def _load(self):
        geometry       = pq.read_table(self.store_directory / "geometry.parquet")
        self.positions = np.column_stack([geometry.column(axis).to_numpy() for axis in self.AXES]).astype(np.float64)

        events        = pq.read_table(self.store_directory / "events.parquet")
        channel_count = self.positions.shape[0]
        event_count   = events.num_rows

        self.light   = events.column("light").combine_chunks().values.to_numpy().reshape(event_count, channel_count).astype(np.float64)
        self.targets = np.column_stack([events.column(f"target_{axis}").to_numpy() for axis in self.AXES]).astype(np.float64)

    def _report_geometry(self):
        ranges = {f"Range {axis}": f"[{self.positions[:, index].min():.2f}, {self.positions[:, index].max():.2f}]" for index, axis in enumerate(self.AXES)}

        self.logger.section("[Light Centroid Check]")
        self.logger.kv_table({"Channels": self.positions.shape[0], "Events": self.light.shape[0], **ranges}, title="Store")

    def _centroid(self, channel_mask):
        light     = self.light[:, channel_mask]
        positions = self.positions[channel_mask]
        weight    = light.sum(axis=1, keepdims=True)
        return (light @ positions) / np.maximum(weight, 1e-12), weight[:, 0]

    def _axis_row(self, axis, centroid_values, target_values):
        pearson_r        = float(np.corrcoef(centroid_values, target_values)[0, 1])
        slope, intercept = np.polyfit(centroid_values, target_values, 1)

        return {
            "Axis"          : axis,
            "Pearson r"     : f"{pearson_r:.4f}",
            "Fit slope"     : f"{float(slope):.4f}",
            "Fit intercept" : f"{float(intercept):.4f}",
            "MAE raw"       : f"{float(np.mean(np.abs(centroid_values - target_values))):.4f}",
            "MAE fitted"    : f"{float(np.mean(np.abs((slope * centroid_values + intercept) - target_values))):.4f}",
            "Target std"    : f"{float(target_values.std()):.4f}",
        }

    def _report_centroid(self, title, centroid, valid):
        lit_centroid = centroid[valid]
        lit_targets  = self.targets[valid]

        rows = [self._axis_row(axis, lit_centroid[:, index], lit_targets[:, index]) for index, axis in enumerate(self.AXES)]

        self.logger.subsection(f"{title}: {int(valid.sum())}/{len(valid)} events carry light")
        self.logger.metrics_table(rows, self.TABLE_FIELDS, title=title)

    def _report_full_detector(self):
        full_mask         = np.ones(self.positions.shape[0], dtype=bool)
        centroid, weight  = self._centroid(full_mask)
        self._report_centroid("Full detector", centroid, weight > 0)

    def _report_barrel_only(self):
        extreme     = float(np.max(np.abs(self.positions[:, 2])))
        barrel_mask = np.abs(self.positions[:, 2]) < (extreme - self.FACE_MARGIN)

        centroid, weight = self._centroid(barrel_mask)
        self._report_centroid(f"Endcap faces excluded (|z| < {extreme - self.FACE_MARGIN:.2f}, {int(barrel_mask.sum())} channels)", centroid, weight > 0)

    def run(self):
        self._load()
        self._report_geometry()
        self._report_full_detector()
        self._report_barrel_only()


if __name__ == "__main__":
    logger = Logger(log_dir="logs", name="light_centroid_check")
    LightCentroidCheck(DataConfig().parquet_store_dir, logger).run()
    logger.close()
