from __future__ import annotations

from pathlib import Path

from tools.reporting.markdown import MarkdownDoc, MarkdownTable


class AnalysisReport:
    COORDINATE_NAMES = ("x", "y", "z")

    def __init__(self, output_directory, logger):
        self.output_directory = Path(output_directory)
        self.logger           = logger

    def _metric_table(self, document, metrics):
        table = MarkdownTable(["Coord", "MAE", "RMSE", "Median", "Bias", "R2"], align=["left", "right", "right", "right", "right", "right"])
        for name in self.COORDINATE_NAMES:
            table.add_row(name, f"{metrics[f'mae_{name}']:.4f}", f"{metrics[f'rmse_{name}']:.4f}", f"{metrics[f'median_{name}']:.4f}", f"{metrics[f'bias_{name}']:.4f}", f"{metrics[f'r2_{name}']:.4f}")
        table.add_row("overall", f"{metrics['mae']:.4f}", f"{metrics['rmse']:.4f}", f"{metrics['median_error']:.4f}", f"{metrics['bias']:.4f}", f"{metrics['r2']:.4f}")
        document.table(table)

        document.kv_table({
            "euclidean_mean"   : f"{metrics['euclidean_mean']:.4f}",
            "euclidean_median" : f"{metrics['euclidean_median']:.4f}",
            "euclidean_p90"    : f"{metrics['euclidean_p90']:.4f}",
            "euclidean_p95"    : f"{metrics['euclidean_p95']:.4f}",
            "count"            : metrics["count"],
        }, header=("Euclidean metric", "Value"))

    def build(self, per_split_results):
        document = MarkdownDoc("Inference Analysis")

        for split_name, payload in per_split_results.items():
            document.heading(f"Split: {split_name}", level=2)
            self._metric_table(document, payload["metrics"])

            for filename in payload["plots"] + payload["animations"]:
                document.image(filename, f"{split_name}/{filename}")

        report_path = self.output_directory / "analysis_report.md"
        document.save(report_path)
        self.logger.subsection(f"Analysis report written to {report_path}")
        return report_path
