from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from configuration.inference.general import AnalysisAnimationStyle, AnalysisPlotStyle, PlotStyle, VertexCoordinates
from configuration.shared            import IoNames
from pipelines.inference.plots       import FigureWriter


class AnalysisAnimations(FigureWriter):
    def _subsample(self, predictions, targets):
        if predictions.shape[0] <= AnalysisAnimationStyle.maximum_points:
            return predictions, targets

        generator = np.random.default_rng(0)
        selection = generator.choice(predictions.shape[0], size=AnalysisAnimationStyle.maximum_points, replace=False)
        return predictions[selection], targets[selection]

    def _rotating_predictions(self, predictions, targets):
        figure = plt.figure(figsize=AnalysisPlotStyle.volume_figure)
        axis   = figure.add_subplot(projection="3d")
        axis.scatter(targets[:, 0], targets[:, 1], targets[:, 2], s=AnalysisAnimationStyle.point_size, alpha=AnalysisAnimationStyle.point_alpha, color=PlotStyle.primary_color, label="True")
        axis.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], s=AnalysisAnimationStyle.point_size, alpha=AnalysisAnimationStyle.point_alpha, color=PlotStyle.accent_color, label="Predicted")
        self._label_volume(axis, "Predicted versus true vertices")
        axis.legend(loc="upper left", bbox_to_anchor=AnalysisAnimationStyle.legend_anchor, frameon=False)

        self._spin(figure, axis, IoNames.rotating_prediction_animation)

    def _rotating_error(self, targets, distances):
        figure  = plt.figure(figsize=AnalysisPlotStyle.volume_figure)
        axis    = figure.add_subplot(projection="3d")
        scatter = axis.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c=distances, cmap=AnalysisPlotStyle.colormap, s=AnalysisAnimationStyle.error_point_size, alpha=AnalysisPlotStyle.volume_alpha)
        self._label_volume(axis, "Euclidean error over the detector volume")
        figure.colorbar(scatter, ax=axis, shrink=AnalysisPlotStyle.colorbar_shrink, label=f"Euclidean error [{VertexCoordinates.unit}]")

        self._spin(figure, axis, IoNames.rotating_error_animation)

    def _label_volume(self, axis, title):
        axis.set_xlabel(f"x [{VertexCoordinates.unit}]")
        axis.set_ylabel(f"y [{VertexCoordinates.unit}]")
        axis.set_zlabel(f"z [{VertexCoordinates.unit}]")
        axis.set_title(title)

    def _turn(self, axis, frame_index):
        axis.view_init(elev=AnalysisAnimationStyle.elevation, azim=frame_index * (AnalysisAnimationStyle.full_turn / AnalysisAnimationStyle.frames))
        return []

    def _spin(self, figure, axis, filename):
        animation = FuncAnimation(figure, lambda frame_index: self._turn(axis, frame_index), frames=AnalysisAnimationStyle.frames, blit=False)
        animation.save(self.output_directory / filename, writer=PillowWriter(fps=AnalysisAnimationStyle.frames_per_second), dpi=AnalysisAnimationStyle.frame_dpi)
        self.generated.append(filename)
        plt.close(figure)

    def run(self, predictions, targets):
        predictions          = predictions.detach().cpu().numpy()
        targets              = targets.detach().cpu().numpy()
        predictions, targets = self._subsample(predictions, targets)
        distances            = np.sqrt(((predictions - targets) ** 2).sum(axis=1))

        self._rotating_predictions(predictions, targets)
        self._rotating_error(targets, distances)

        self.logger.subsection(f"Generated {len(self.generated)} analysis animations in {self.output_directory}")
        return self.generated
