from __future__ import annotations

import csv
import json
from pathlib import Path

from configuration.inference.general import VertexCoordinates
from configuration.shared            import IoNames
from configuration.sweep.general     import SweepMetrics
from tools.reporting.markdown        import MarkdownDoc, MarkdownTable


class SweepReport:
    def __init__(self, output_directory, logger):
        self.output_directory = Path(output_directory)
        self.logger           = logger
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _write_json(self, context, records):
        payload = {
            "unit"    : VertexCoordinates.unit,
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
