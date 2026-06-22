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


class ShapNativePlots:
    SUBDIRECTORY = "shap"

    def __init__(self, output_directory, logger):
        self.output_directory = Path(output_directory) / self.SUBDIRECTORY
        self.logger           = logger
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.generated        = []

        PublicationStyle.apply()

    def _save(self, title, filename):
        figure = plt.gcf()
        figure.axes[0].set_title(title)
        figure.tight_layout()
        figure.savefig(self.output_directory / filename, bbox_inches="tight")
        plt.close(figure)
        self.generated.append(f"{self.SUBDIRECTORY}/{filename}")

    def _beeswarm(self, shap_module, shap_values, feature_values, names, title, filename):
        shap_module.summary_plot(shap_values, features=feature_values, feature_names=names, plot_type="dot", max_display=len(names), show=False)
        self._save(title, filename)

    def _bar(self, shap_module, shap_values, names, title, filename):
        shap_module.summary_plot(shap_values, feature_names=names, plot_type="bar", max_display=len(names), show=False)
        self._save(title, filename)

    def _overall_bar(self, shap_module, shap_values_per_axis, names, coordinates, title, filename):
        shap_module.summary_plot(list(shap_values_per_axis), feature_names=names, plot_type="bar", class_names=list(coordinates), max_display=len(names), show=False)
        self._save(title, filename)

    def _domain_plots(self, shap_module, domain, shap_matrix, feature_values, names, coordinates):
        self._overall_bar(shap_module, [shap_matrix[:, :, index] for index in range(len(coordinates))], names, coordinates, f"SHAP {domain} features (all axes)", f"shap_bar_{domain}_overall.png")

        for index, axis in enumerate(coordinates):
            self._beeswarm(shap_module, shap_matrix[:, :, index], feature_values, names, f"SHAP {domain} features ({axis})", f"shap_beeswarm_{domain}_{axis}.png")
            self._bar(shap_module, shap_matrix[:, :, index], names, f"SHAP {domain} features ({axis})", f"shap_bar_{domain}_{axis}.png")

    def run(self, samples, node_names, edge_names, coordinates):
        import shap

        self._domain_plots(shap, "node", samples["node_shap"], samples["node_feature_values"], node_names, coordinates)
        self._domain_plots(shap, "edge", samples["edge_shap"], samples["edge_feature_values"], edge_names, coordinates)

        self.logger.subsection(f"Generated {len(self.generated)} native SHAP plots in {self.output_directory}")
        return self.generated
