from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from configuration.inference.general import PlotStyle
from configuration.shared            import IoNames
from configuration.sweep.general     import SweepPlotStyle
from pipelines.inference.plots       import FigureWriter


class SweepPlots(FigureWriter):
    def _baseline_index(self, axis_values, baseline_value):
        if baseline_value < axis_values.min() or baseline_value > axis_values.max():
            return None
        return float(np.interp(baseline_value, axis_values, np.arange(len(axis_values))))

    def _heatmap(self, grid, metric_label, filename, cmap):
        figure, axis = plt.subplots(figsize=SweepPlotStyle.heatmap_figure)
        image        = axis.imshow(grid, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest")

        axis.set_xticks(np.arange(len(self.scale_values)))
        axis.set_yticks(np.arange(len(self.efficiency_values)))
        axis.set_xticklabels([f"{value:g}" for value in self.scale_values], rotation=SweepPlotStyle.tick_rotation, ha="right")
        axis.set_yticklabels([f"{value:g}" for value in self.efficiency_values])
        axis.set_xlabel(SweepPlotStyle.scale_label)
        axis.set_ylabel(SweepPlotStyle.efficiency_label)
        axis.set_title(metric_label)
        axis.grid(False)

        if self.annotate:
            finite   = grid[np.isfinite(grid)]
            midpoint = (finite.min() + finite.max()) / 2 if finite.size > 0 else 0.0
            for row in range(grid.shape[0]):
                for column in range(grid.shape[1]):
                    value = grid[row, column]
                    if not np.isfinite(value):
                        continue
                    colour = SweepPlotStyle.light_text_color if value < midpoint else SweepPlotStyle.dark_text_color
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=SweepPlotStyle.annotation_size, color=colour)

        scale_index      = self._baseline_index(self.scale_values, self.baseline_scale)
        efficiency_index = self._baseline_index(self.efficiency_values, self.baseline_efficiency)
        if scale_index is not None and efficiency_index is not None:
            axis.scatter([scale_index], [efficiency_index], marker="*", s=SweepPlotStyle.baseline_marker_size, facecolors="none", edgecolors=PlotStyle.accent_color, linewidths=SweepPlotStyle.baseline_line_width, label="training baseline")
            axis.legend(loc="upper left", frameon=True, fontsize=SweepPlotStyle.legend_font_size)

        figure.colorbar(image, ax=axis, shrink=SweepPlotStyle.colorbar_shrink, label=metric_label)
        self._save(figure, filename)

    def _lines_vs_scale(self, grid):
        figure, axis = plt.subplots(figsize=SweepPlotStyle.line_figure)
        colours      = matplotlib.colormaps[SweepPlotStyle.lines_vs_scale_colormap](np.linspace(*SweepPlotStyle.colormap_range, len(self.efficiency_values)))
        for row, efficiency in enumerate(self.efficiency_values):
            axis.plot(self.scale_values, grid[row, :], marker="o", markersize=SweepPlotStyle.marker_size, linewidth=SweepPlotStyle.line_width, color=colours[row], label=f"{efficiency:g}")
        axis.set_xlabel(SweepPlotStyle.scale_label)
        axis.set_ylabel("Mean euclidean error [m]")
        axis.set_title("Mean euclidean error versus light scale factor")
        axis.legend(title=SweepPlotStyle.efficiency_label, fontsize=SweepPlotStyle.legend_font_size, title_fontsize=SweepPlotStyle.legend_font_size, ncol=2, loc="best")
        self._save(figure, IoNames.euclidean_vs_scale_figure)

    def _lines_vs_efficiency(self, grid):
        figure, axis = plt.subplots(figsize=SweepPlotStyle.line_figure)
        colours      = matplotlib.colormaps[SweepPlotStyle.lines_vs_efficiency_colormap](np.linspace(*SweepPlotStyle.colormap_range, len(self.scale_values)))
        for column, scale in enumerate(self.scale_values):
            axis.plot(self.efficiency_values, grid[:, column], marker="o", markersize=SweepPlotStyle.marker_size, linewidth=SweepPlotStyle.line_width, color=colours[column], label=f"{scale:g}")
        axis.set_xlabel(SweepPlotStyle.efficiency_label)
        axis.set_ylabel("Mean euclidean error [m]")
        axis.set_title("Mean euclidean error versus detection efficiency")
        axis.legend(title=SweepPlotStyle.scale_label, fontsize=SweepPlotStyle.legend_font_size, title_fontsize=SweepPlotStyle.legend_font_size, ncol=2, loc="best")
        self._save(figure, IoNames.euclidean_vs_efficiency_figure)

    def run(self, scale_values, efficiency_values, metric_grids, baseline, annotate):
        self.scale_values        = np.asarray(scale_values, dtype=np.float64)
        self.efficiency_values   = np.asarray(efficiency_values, dtype=np.float64)
        self.baseline_scale      = float(baseline[0])
        self.baseline_efficiency = float(baseline[1])
        self.annotate            = annotate

        for metric_key, metric_label, cmap in SweepPlotStyle.heatmaps:
            if metric_key not in metric_grids:
                continue
            self._heatmap(metric_grids[metric_key], metric_label, IoNames.heatmap_figure.format(metric=metric_key), cmap)

        self._lines_vs_scale(metric_grids["euclidean_mean"])
        self._lines_vs_efficiency(metric_grids["euclidean_mean"])

        self.logger.subsection(f"Generated {len(self.generated)} sweep plots in {self.output_directory}")
        return self.generated
