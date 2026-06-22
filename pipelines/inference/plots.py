from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


class PublicationStyle:
    SETTINGS = {
        "figure.dpi"       : 300,
        "savefig.dpi"      : 300,
        "font.size"        : 10,
        "font.family"      : "sans-serif",
        "axes.grid"        : True,
        "grid.alpha"       : 0.3,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
    }

    @staticmethod
    def apply():
        plt.rcParams.update(PublicationStyle.SETTINGS)


class AnalysisPlots:
    COORDINATE_NAMES = ("x", "y", "z")
    UNIT             = "m"

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

    def _predicted_vs_true(self, predictions, targets):
        for index, name in enumerate(self.COORDINATE_NAMES):
            true_values      = targets[:, index]
            predicted_values = predictions[:, index]
            limits           = [min(true_values.min(), predicted_values.min()), max(true_values.max(), predicted_values.max())]

            figure, axis = plt.subplots(figsize=(5, 5))
            axis.scatter(true_values, predicted_values, s=6, alpha=0.4, color="#2c7fb8", edgecolors="none")
            axis.plot(limits, limits, color="#d95f0e", linewidth=1.2, linestyle="--")
            axis.set_xlabel(f"True {name} [{self.UNIT}]")
            axis.set_ylabel(f"Predicted {name} [{self.UNIT}]")
            axis.set_title(f"Predicted vs true {name}")
            axis.set_aspect("equal", adjustable="box")
            self._save(figure, f"predicted_vs_true_{name}.png")

    def _residual_histograms(self, predictions, targets):
        for index, name in enumerate(self.COORDINATE_NAMES):
            residuals = (predictions[:, index] - targets[:, index])

            figure, axis = plt.subplots(figsize=(5, 4))
            axis.hist(residuals, bins=60, color="#2c7fb8", alpha=0.85)
            axis.axvline(0.0, color="#d95f0e", linewidth=1.2, linestyle="--")
            axis.set_xlabel(f"Residual {name} (predicted - true) [{self.UNIT}]")
            axis.set_ylabel("Count")
            axis.set_title(f"Residual distribution {name}")
            self._save(figure, f"residual_histogram_{name}.png")

    def _error_vs_true(self, predictions, targets):
        for index, name in enumerate(self.COORDINATE_NAMES):
            true_values     = targets[:, index]
            absolute_errors = np.abs(predictions[:, index] - targets[:, index])

            figure, axis = plt.subplots(figsize=(5, 4))
            axis.scatter(true_values, absolute_errors, s=6, alpha=0.4, color="#2c7fb8", edgecolors="none")
            axis.set_xlabel(f"True {name} [{self.UNIT}]")
            axis.set_ylabel(f"Absolute error [{self.UNIT}]")
            axis.set_title(f"Absolute error versus true {name}")
            self._save(figure, f"error_vs_true_{name}.png")

    def _euclidean_histogram(self, distances):
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.hist(distances, bins=60, color="#31a354", alpha=0.85)
        axis.axvline(float(np.median(distances)), color="#d95f0e", linewidth=1.2, linestyle="--")
        axis.set_xlabel(f"Euclidean error [{self.UNIT}]")
        axis.set_ylabel("Count")
        axis.set_title("Euclidean error distribution")
        self._save(figure, "euclidean_error_histogram.png")

    def _euclidean_cumulative(self, distances):
        sorted_distances    = np.sort(distances)
        cumulative_fraction = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)

        figure, axis = plt.subplots(figsize=(5, 4))
        axis.plot(sorted_distances, cumulative_fraction, color="#2c7fb8", linewidth=1.5)
        axis.set_xlabel(f"Euclidean error [{self.UNIT}]")
        axis.set_ylabel("Cumulative fraction")
        axis.set_title("Cumulative euclidean error")
        axis.set_ylim(0, 1)
        self._save(figure, "euclidean_error_cumulative.png")

    def _error_three_dimensional(self, targets, distances):
        figure = plt.figure(figsize=(6, 5))
        axis   = figure.add_subplot(111, projection="3d")
        scatter = axis.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c=distances, cmap="viridis", s=8, alpha=0.7)
        axis.set_xlabel(f"x [{self.UNIT}]")
        axis.set_ylabel(f"y [{self.UNIT}]")
        axis.set_zlabel(f"z [{self.UNIT}]")
        axis.set_title("Spatial distribution of euclidean error")
        figure.colorbar(scatter, ax=axis, shrink=0.6, label=f"Euclidean error [{self.UNIT}]")
        self._save(figure, "error_three_dimensional.png")

    def run(self, predictions, targets):
        predictions = predictions.detach().cpu().numpy()
        targets     = targets.detach().cpu().numpy()
        distances   = np.sqrt(((predictions - targets) ** 2).sum(axis=1))

        self._predicted_vs_true(predictions, targets)
        self._residual_histograms(predictions, targets)
        self._error_vs_true(predictions, targets)
        self._euclidean_histogram(distances)
        self._euclidean_cumulative(distances)
        self._error_three_dimensional(targets, distances)

        self.logger.subsection(f"Generated {len(self.generated)} analysis plots in {self.output_directory}")
        return self.generated
