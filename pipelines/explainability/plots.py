from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pipelines.inference.plots import PublicationStyle


class FeatureImportancePlots:
    NODE_COLOR = "#2c7fb8"
    EDGE_COLOR = "#31a354"

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

    def _bar(self, names, values, errors, xlabel, title, filename, color):
        if len(names) == 0:
            return

        values = np.asarray(values, dtype=np.float64)
        order  = np.argsort(values)

        ordered_names  = [names[index] for index in order]
        ordered_values = values[order]
        ordered_errors = np.asarray(errors, dtype=np.float64)[order] if errors is not None else None

        height       = max(2.6, 0.34 * len(ordered_names) + 1.0)
        figure, axis = plt.subplots(figsize=(7.0, height))
        positions    = np.arange(len(ordered_names))

        axis.barh(positions, ordered_values, xerr=ordered_errors, color=color, edgecolor="none", height=0.74, error_kw={"elinewidth": 0.8, "ecolor": "#525252"})
        axis.axvline(0.0, color="#525252", linewidth=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels(ordered_names)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="y", visible=False)

        self._save(figure, filename)

    def run(self, figures):
        for specification in figures:
            self._bar(**specification)

        self.logger.subsection(f"Generated {len(self.generated)} importance plots in {self.output_directory}")
        return self.generated
