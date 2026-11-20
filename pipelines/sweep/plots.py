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

    def _lines(self, x_values, series_values, curves, colormap, x_label, series_label, title, filename):
        figure, axis = plt.subplots(figsize=SweepPlotStyle.line_figure)
        colours      = matplotlib.colormaps[colormap](np.linspace(*SweepPlotStyle.colormap_range, len(series_values)))
        for index, series_value in enumerate(series_values):
            axis.plot(x_values, curves[index], marker="o", markersize=SweepPlotStyle.marker_size, linewidth=SweepPlotStyle.line_width, color=colours[index], label=f"{series_value:g}")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Mean euclidean error [m]")
        axis.set_title(title)
        axis.legend(title=series_label, fontsize=SweepPlotStyle.legend_font_size, title_fontsize=SweepPlotStyle.legend_font_size, ncol=2, loc="best")
        self._save(figure, filename)

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

        euclidean = metric_grids["euclidean_mean"]
        self._lines(self.scale_values, self.efficiency_values, euclidean, SweepPlotStyle.lines_vs_scale_colormap, SweepPlotStyle.scale_label, SweepPlotStyle.efficiency_label, "Mean euclidean error versus light scale factor", IoNames.euclidean_vs_scale_figure)
        self._lines(self.efficiency_values, self.scale_values, euclidean.T, SweepPlotStyle.lines_vs_efficiency_colormap, SweepPlotStyle.efficiency_label, SweepPlotStyle.scale_label, "Mean euclidean error versus detection efficiency", IoNames.euclidean_vs_efficiency_figure)

        self.logger.subsection(f"Generated {len(self.generated)} sweep plots in {self.output_directory}")
        return self.generated
