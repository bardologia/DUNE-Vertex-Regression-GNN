from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from configuration.data.general      import CoordinateAxes
from configuration.inference.general import PlotStyle
from configuration.shared            import IoNames
from configuration.sweep.general     import SweepMetrics, SweepPlotStyle
from pipelines.inference.report      import FigureWriter
from tools.reporting.markdown        import MarkdownDoc, MarkdownTable


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


class SweepReport:
    def __init__(self, output_directory, logger):
        self.output_directory = Path(output_directory)
        self.logger           = logger
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _write_json(self, context, records):
        payload = {
            "unit"    : CoordinateAxes.unit,
            "run"     : context,
            "results" : records,
        }
        path = self.output_directory / IoNames.results_file
        path.write_text(json.dumps(payload, indent=IoNames.json_indent), encoding="utf-8")
        return path

    def _write_csv(self, records):
        path = self.output_directory / IoNames.sweep_results_table
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SweepMetrics.csv_columns)
            for record in records:
                metrics = record["metrics"]
                row     = [record["scale_factor"], record["detection_efficiency"]] + [metrics[column] for column in SweepMetrics.csv_columns[2:]]
                writer.writerow(row)
        return path

    def _configuration_table(self, document, context):
        document.heading("Configuration", level=2)
        document.kv_table({
            "Run directory"        : context["run_directory"],
            "Model"                : context["model_name"],
            "Checkpoint"           : context["checkpoint"],
            "Evaluated split"      : context["split"],
            "Split events"         : context["split_event_count"],
            "Baseline scale"       : context["baseline_scale"],
            "Baseline efficiency"  : context["baseline_efficiency"],
            "Efficiency seed"      : context["efficiency_seed"],
            "Normalization stats"  : "training split (reused)",
            "Grid cells"           : context["grid_cells"],
        })

    def _specification_table(self, document, context):
        document.heading("Sweep grid", level=2)
        table = MarkdownTable(["Axis", "Minimum", "Maximum", "Step", "Points", "Values"], align=["left", "right", "right", "right", "right", "left"])
        for axis in context["axes"]:
            table.add_row(axis["name"], f"{axis['minimum']:g}", f"{axis['maximum']:g}", f"{axis['step']:g}", axis["count"], ", ".join(f"{value:g}" for value in axis["values"]))
        document.table(table)

    def _highlights(self, document, records):
        ranked = sorted(records, key=lambda record: record["metrics"]["euclidean_mean"])
        best   = ranked[0]
        worst  = ranked[-1]

        document.heading("Highlights", level=2)
        document.kv_table({
            "Best mean euclidean error [m]"  : f"{best['metrics']['euclidean_mean']:.4f} at scale={best['scale_factor']:g}, efficiency={best['detection_efficiency']:g}",
            "Worst mean euclidean error [m]" : f"{worst['metrics']['euclidean_mean']:.4f} at scale={worst['scale_factor']:g}, efficiency={worst['detection_efficiency']:g}",
        }, code_keys=False)

    def _results_table(self, document, records):
        document.heading("Per-cell metrics", level=2)
        table = MarkdownTable(
            ["Scale", "Efficiency", "Count", "MAE [m]", "RMSE [m]", "Eucl. mean [m]", "Eucl. median [m]", "R2"],
            align=["right", "right", "right", "right", "right", "right", "right", "right"],
        )
        for record in records:
            metrics = record["metrics"]
            table.add_row(
                f"{record['scale_factor']:g}",
                f"{record['detection_efficiency']:g}",
                metrics["count"],
                f"{metrics['mae']:.4f}",
                f"{metrics['rmse']:.4f}",
                f"{metrics['euclidean_mean']:.4f}",
                f"{metrics['euclidean_median']:.4f}",
                f"{metrics['r2']:.4f}",
            )
        document.table(table)

    def _plots(self, document, plot_filenames):
        document.heading("Figures", level=2)
        for filename in plot_filenames:
            document.image(filename, f"{IoNames.plots_dir}/{filename}")

    def build(self, context, records, plot_filenames):
        json_path = self._write_json(context, records)
        csv_path  = self._write_csv(records)

        document = MarkdownDoc("Energy and Efficiency Sweep")
        document.paragraph("Robustness of a trained DUNE-GNN checkpoint when the held-out split is regenerated under different prompt-light scale factors and binomial detection efficiencies. The split membership and the training-split normalization statistics are held fixed; only the photon-counting physics varies.")

        self._configuration_table(document, context)
        self._specification_table(document, context)
        self._highlights(document, records)
        self._results_table(document, records)
        self._plots(document, plot_filenames)

        report_path = self.output_directory / IoNames.report_file
        document.save(report_path)

        self.logger.subsection(f"Sweep report written to {report_path}")
        self.logger.subsection(f"Results written to {csv_path} and {json_path}")
        return report_path
