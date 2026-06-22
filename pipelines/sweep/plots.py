from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pipelines.inference.plots import PublicationStyle


class SweepPlots:
    HEATMAP_METRICS = (
        ("euclidean_mean",   "Mean euclidean error [cm]",   "viridis_r"),
        ("euclidean_median", "Median euclidean error [cm]", "viridis_r"),
        ("mae",              "Mean absolute error [cm]",     "viridis_r"),
        ("rmse",             "Root-mean-square error [cm]",  "viridis_r"),
        ("r2",               "Coefficient of determination", "viridis"),
    )

    SCALE_LABEL      = "Light scale factor"
    EFFICIENCY_LABEL = "Detection efficiency"

    def __init__(self, output_directory, logger):
        self.output_directory = Path(output_directory)
        self.logger           = logger
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.generated        = []

        PublicationStyle.apply()

    def _save(self, figure, filename):
        path = self.output_directory / filename
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        self.generated.append(filename)

    def _baseline_index(self, axis_values, baseline_value):
        if baseline_value < axis_values.min() or baseline_value > axis_values.max():
            return None
        return float(np.interp(baseline_value, axis_values, np.arange(len(axis_values))))

    def _heatmap(self, grid, metric_label, filename, cmap):
        figure, axis = plt.subplots(figsize=(6, 5))
        image        = axis.imshow(grid, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest")

        axis.set_xticks(np.arange(len(self.scale_values)))
        axis.set_yticks(np.arange(len(self.efficiency_values)))
        axis.set_xticklabels([f"{value:g}" for value in self.scale_values], rotation=45, ha="right")
        axis.set_yticklabels([f"{value:g}" for value in self.efficiency_values])
        axis.set_xlabel(self.SCALE_LABEL)
        axis.set_ylabel(self.EFFICIENCY_LABEL)
        axis.set_title(metric_label)
        axis.grid(False)

        if self.annotate:
            finite = grid[np.isfinite(grid)]
            midpoint = 0.5 * (finite.min() + finite.max()) if finite.size > 0 else 0.0
            for row in range(grid.shape[0]):
                for column in range(grid.shape[1]):
                    value = grid[row, column]
                    if not np.isfinite(value):
                        continue
                    colour = "white" if value < midpoint else "black"
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6, color=colour)

        scale_index      = self._baseline_index(self.scale_values, self.baseline_scale)
        efficiency_index = self._baseline_index(self.efficiency_values, self.baseline_efficiency)
        if scale_index is not None and efficiency_index is not None:
            axis.scatter([scale_index], [efficiency_index], marker="*", s=160, facecolors="none", edgecolors="#d95f0e", linewidths=1.4, label="training baseline")
            axis.legend(loc="upper left", frameon=True, fontsize=7)

        figure.colorbar(image, ax=axis, shrink=0.85, label=metric_label)
        self._save(figure, filename)

    def _lines_vs_scale(self, grid):
        figure, axis = plt.subplots(figsize=(6, 4.5))
        colours      = plt.cm.viridis(np.linspace(0.15, 0.9, len(self.efficiency_values)))
        for row, efficiency in enumerate(self.efficiency_values):
            axis.plot(self.scale_values, grid[row, :], marker="o", markersize=3, linewidth=1.2, color=colours[row], label=f"{efficiency:g}")
        axis.set_xlabel(self.SCALE_LABEL)
        axis.set_ylabel("Mean euclidean error [cm]")
        axis.set_title("Mean euclidean error versus light scale factor")
        axis.legend(title=self.EFFICIENCY_LABEL, fontsize=7, title_fontsize=7, ncol=2, loc="best")
        self._save(figure, "euclidean_mean_vs_scale.png")

    def _lines_vs_efficiency(self, grid):
        figure, axis = plt.subplots(figsize=(6, 4.5))
        colours      = plt.cm.plasma(np.linspace(0.15, 0.9, len(self.scale_values)))
        for column, scale in enumerate(self.scale_values):
            axis.plot(self.efficiency_values, grid[:, column], marker="o", markersize=3, linewidth=1.2, color=colours[column], label=f"{scale:g}")
        axis.set_xlabel(self.EFFICIENCY_LABEL)
        axis.set_ylabel("Mean euclidean error [cm]")
        axis.set_title("Mean euclidean error versus detection efficiency")
        axis.legend(title=self.SCALE_LABEL, fontsize=7, title_fontsize=7, ncol=2, loc="best")
        self._save(figure, "euclidean_mean_vs_efficiency.png")

    def run(self, scale_values, efficiency_values, metric_grids, baseline, annotate):
        self.scale_values        = np.asarray(scale_values, dtype=np.float64)
        self.efficiency_values   = np.asarray(efficiency_values, dtype=np.float64)
        self.baseline_scale      = float(baseline[0])
        self.baseline_efficiency = float(baseline[1])
        self.annotate            = annotate

        for metric_key, metric_label, cmap in self.HEATMAP_METRICS:
            if metric_key not in metric_grids:
                continue
            self._heatmap(metric_grids[metric_key], metric_label, f"heatmap_{metric_key}.png", cmap)

        self._lines_vs_scale(metric_grids["euclidean_mean"])
        self._lines_vs_efficiency(metric_grids["euclidean_mean"])

        self.logger.subsection(f"Generated {len(self.generated)} sweep plots in {self.output_directory}")
        return self.generated
