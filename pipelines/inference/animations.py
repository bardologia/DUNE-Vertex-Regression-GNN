from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from pipelines.inference.plots import PublicationStyle


class AnalysisAnimations:
    UNIT             = "cm"
    MAXIMUM_POINTS   = 2000
    NUMBER_OF_FRAMES = 48
    FRAME_DPI        = 100

    def __init__(self, output_directory, logger):
        self.output_directory = Path(output_directory)
        self.logger           = logger
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.generated        = []

        PublicationStyle.apply()

    def _subsample(self, predictions, targets):
        if predictions.shape[0] <= self.MAXIMUM_POINTS:
            return predictions, targets
        generator = np.random.default_rng(0)
        selection = generator.choice(predictions.shape[0], size=self.MAXIMUM_POINTS, replace=False)
        return predictions[selection], targets[selection]

    def _save(self, animation, filename):
        path = self.output_directory / filename
        animation.save(path, writer=PillowWriter(fps=15), dpi=self.FRAME_DPI)
        self.generated.append(filename)

    def _rotating_predictions(self, predictions, targets):
        figure = plt.figure(figsize=(6, 5))
        axis   = figure.add_subplot(111, projection="3d")
        axis.scatter(targets[:, 0], targets[:, 1], targets[:, 2], s=8, alpha=0.5, color="#2c7fb8", label="True")
        axis.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], s=8, alpha=0.5, color="#d95f0e", label="Predicted")
        axis.set_xlabel(f"x [{self.UNIT}]")
        axis.set_ylabel(f"y [{self.UNIT}]")
        axis.set_zlabel(f"z [{self.UNIT}]")
        axis.set_title("Predicted versus true vertices")
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        def update(frame_index):
            axis.view_init(elev=20, azim=frame_index * (360 / self.NUMBER_OF_FRAMES))
            return []

        animation = FuncAnimation(figure, update, frames=self.NUMBER_OF_FRAMES, blit=False)
        self._save(animation, "rotating_predicted_vs_true.gif")
        plt.close(figure)

    def _rotating_error(self, targets, distances):
        figure  = plt.figure(figsize=(6, 5))
        axis    = figure.add_subplot(111, projection="3d")
        scatter = axis.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c=distances, cmap="viridis", s=10, alpha=0.7)
        axis.set_xlabel(f"x [{self.UNIT}]")
        axis.set_ylabel(f"y [{self.UNIT}]")
        axis.set_zlabel(f"z [{self.UNIT}]")
        axis.set_title("Euclidean error over the detector volume")
        figure.colorbar(scatter, ax=axis, shrink=0.6, label=f"Euclidean error [{self.UNIT}]")

        def update(frame_index):
            axis.view_init(elev=20, azim=frame_index * (360 / self.NUMBER_OF_FRAMES))
            return []

        animation = FuncAnimation(figure, update, frames=self.NUMBER_OF_FRAMES, blit=False)
        self._save(animation, "rotating_error_volume.gif")
        plt.close(figure)

    def run(self, predictions, targets):
        predictions = predictions.detach().cpu().numpy()
        targets     = targets.detach().cpu().numpy()
        predictions, targets = self._subsample(predictions, targets)
        distances   = np.sqrt(((predictions - targets) ** 2).sum(axis=1))

        self._rotating_predictions(predictions, targets)
        self._rotating_error(targets, distances)

        self.logger.subsection(f"Generated {len(self.generated)} analysis animations in {self.output_directory}")
        return self.generated
