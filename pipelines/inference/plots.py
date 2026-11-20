from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from configuration.inference.general import AnalysisPlotStyle, PlotStyle, VertexCoordinates
from configuration.shared            import IoNames


class PublicationStyle:
    @staticmethod
    def apply():
        plt.rcParams.update(dict(PlotStyle.rc))


class FigureWriter:
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


class AnalysisPlots(FigureWriter):
    def _predicted_vs_true(self, predictions, targets):
        for index, name in enumerate(VertexCoordinates.names):
            true_values      = targets[:, index]
            predicted_values = predictions[:, index]
            limits           = [min(true_values.min(), predicted_values.min()), max(true_values.max(), predicted_values.max())]

            figure, axis = plt.subplots(figsize=AnalysisPlotStyle.square_figure)
            axis.scatter(true_values, predicted_values, s=AnalysisPlotStyle.point_size, alpha=AnalysisPlotStyle.point_alpha, color=PlotStyle.primary_color, edgecolors="none")
            axis.plot(limits, limits, color=PlotStyle.accent_color, linewidth=AnalysisPlotStyle.line_width, linestyle="--")
            axis.set_xlabel(f"True {name} [{VertexCoordinates.unit}]")
            axis.set_ylabel(f"Predicted {name} [{VertexCoordinates.unit}]")
            axis.set_title(f"Predicted vs true {name}")
            axis.set_aspect("equal", adjustable="box")
            self._save(figure, IoNames.predicted_vs_true_figure.format(coordinate=name))

    def _residual_histograms(self, predictions, targets):
        for index, name in enumerate(VertexCoordinates.names):
            residuals = predictions[:, index] - targets[:, index]

            figure, axis = plt.subplots(figsize=AnalysisPlotStyle.panel_figure)
            axis.hist(residuals, bins=AnalysisPlotStyle.bins, color=PlotStyle.primary_color, alpha=AnalysisPlotStyle.bar_alpha)
            axis.axvline(0.0, color=PlotStyle.accent_color, linewidth=AnalysisPlotStyle.line_width, linestyle="--")
            axis.set_xlabel(f"Residual {name} (predicted - true) [{VertexCoordinates.unit}]")
            axis.set_ylabel("Count")
            axis.set_title(f"Residual distribution {name}")
            self._save(figure, IoNames.residual_histogram_figure.format(coordinate=name))

    def _error_vs_true(self, predictions, targets):
        for index, name in enumerate(VertexCoordinates.names):
            true_values     = targets[:, index]
            absolute_errors = np.abs(predictions[:, index] - targets[:, index])

            figure, axis = plt.subplots(figsize=AnalysisPlotStyle.panel_figure)
            axis.scatter(true_values, absolute_errors, s=AnalysisPlotStyle.point_size, alpha=AnalysisPlotStyle.point_alpha, color=PlotStyle.primary_color, edgecolors="none")
            axis.set_xlabel(f"True {name} [{VertexCoordinates.unit}]")
            axis.set_ylabel(f"Absolute error [{VertexCoordinates.unit}]")
            axis.set_title(f"Absolute error versus true {name}")
            self._save(figure, IoNames.error_vs_true_figure.format(coordinate=name))

    def _euclidean_histogram(self, distances):
        figure, axis = plt.subplots(figsize=AnalysisPlotStyle.panel_figure)
        axis.hist(distances, bins=AnalysisPlotStyle.bins, color=PlotStyle.secondary_color, alpha=AnalysisPlotStyle.bar_alpha)
        axis.axvline(float(np.median(distances)), color=PlotStyle.accent_color, linewidth=AnalysisPlotStyle.line_width, linestyle="--")
        axis.set_xlabel(f"Euclidean error [{VertexCoordinates.unit}]")
        axis.set_ylabel("Count")
        axis.set_title("Euclidean error distribution")
        self._save(figure, IoNames.euclidean_histogram_figure)

    def _euclidean_cumulative(self, distances):
        sorted_distances    = np.sort(distances)
        cumulative_fraction = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)

        figure, axis = plt.subplots(figsize=AnalysisPlotStyle.panel_figure)
        axis.plot(sorted_distances, cumulative_fraction, color=PlotStyle.primary_color, linewidth=AnalysisPlotStyle.cumulative_width)
        axis.set_xlabel(f"Euclidean error [{VertexCoordinates.unit}]")
        axis.set_ylabel("Cumulative fraction")
        axis.set_title("Cumulative euclidean error")
        axis.set_ylim(0, 1)
        self._save(figure, IoNames.euclidean_cumulative_figure)

    def _error_three_dimensional(self, targets, distances):
        figure  = plt.figure(figsize=AnalysisPlotStyle.volume_figure)
        axis    = figure.add_subplot(projection="3d")
        scatter = axis.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c=distances, cmap=AnalysisPlotStyle.colormap, s=AnalysisPlotStyle.volume_point_size, alpha=AnalysisPlotStyle.volume_alpha)
        axis.set_xlabel(f"x [{VertexCoordinates.unit}]")
        axis.set_ylabel(f"y [{VertexCoordinates.unit}]")
        axis.set_zlabel(f"z [{VertexCoordinates.unit}]")
        axis.set_title("Spatial distribution of euclidean error")
        figure.colorbar(scatter, ax=axis, shrink=AnalysisPlotStyle.colorbar_shrink, label=f"Euclidean error [{VertexCoordinates.unit}]")
        self._save(figure, IoNames.error_volume_figure)

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
