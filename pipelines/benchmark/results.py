from __future__ import annotations

import json
from pathlib import Path

from configuration.benchmark.general import BenchmarkReportSettings
from configuration.shared            import IoNames
from tools.reporting.markdown        import MarkdownDoc, MarkdownTable, ScalarFormatter


class TrialCollector:
    def __init__(self, pipeline_directory, models, logger):
        self.pipeline_directory = Path(pipeline_directory)
        self.models             = list(models)
        self.logger             = logger

    def _load(self, file_name):
        path = self.pipeline_directory / file_name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _record_for(self, model_name, size_match, overfit, max_batch, training):
        size_entry      = size_match.get(model_name, {})
        overfit_entry   = overfit.get(model_name, {})
        max_batch_entry = max_batch.get(model_name, {})
        training_entry  = training.get(model_name, {})
        metrics         = training_entry.get("metrics", {})

        return {
            "model"               : model_name,
            "parameters"          : size_entry.get("parameters"),
            "width"               : size_entry.get("width"),
            "size_deviation_pct"  : size_entry.get("deviation_pct"),
            "size_flags"          : size_entry.get("flags", []),
            "overfit_status"      : overfit_entry.get("status"),
            "overfit_loss"        : overfit_entry.get("final_loss"),
            "max_batch"           : max_batch_entry.get("batch_size"),
            "max_batch_peak_gb"   : max_batch_entry.get("peak_gb"),
            "max_batch_status"    : max_batch_entry.get("status"),
            "train_status"        : training_entry.get("status"),
            "best_val_loss"       : training_entry.get("best_val_loss"),
            "train_batch_size"    : training_entry.get("batch_size"),
            "lr_scale"            : training_entry.get("lr_scale"),
            "duration_s"          : training_entry.get("duration_s"),
            "run_directory"       : training_entry.get("run_directory"),
            "metrics"             : metrics,
        }

    def collect(self):
        size_match = self._load(IoNames.size_match_file)
        overfit    = self._load(IoNames.overfit_results_file)
        max_batch  = self._load(IoNames.max_batch_file)
        training   = self._load(IoNames.training_results_file)

        return [self._record_for(model_name, size_match, overfit, max_batch, training) for model_name in self.models]


class ComparisonReport:
    def __init__(self, records, output_directory, configuration, logger):
        self.records          = records
        self.output_directory = Path(output_directory)
        self.config           = configuration
        self.logger           = logger
        self.rank_metric      = configuration.comparison.rank_metric

    def _ranked_records(self):
        scored    = [record for record in self.records if record["metrics"].get(self.rank_metric) is not None]
        ascending = self.rank_metric != "r2"
        return sorted(scored, key=lambda record: record["metrics"][self.rank_metric], reverse=not ascending)

    def _leaderboard_table(self, ranked):
        table = MarkdownTable(["Rank", "Model", *BenchmarkReportSettings.leaderboard_metrics], align=["right", "left"] + ["right"] * len(BenchmarkReportSettings.leaderboard_metrics))
        for position, record in enumerate(ranked, start=1):
            cells = [ScalarFormatter.format_scalar(record["metrics"].get(key)) for key in BenchmarkReportSettings.leaderboard_metrics]
            table.add_row(position, record["model"], *cells)
        return table

    def _capacity_table(self):
        table = MarkdownTable(["Model", "Parameters", "Width", "Deviation", "Flags"], align=["left", "right", "right", "right", "left"])
        for record in self.records:
            deviation  = f"{record['size_deviation_pct']:+.2f} %" if record["size_deviation_pct"] is not None else None
            flags      = "; ".join(record["size_flags"]) or None
            parameters = f"{record['parameters']:,}" if record["parameters"] is not None else None
            table.add_row(record["model"], parameters, record["width"], deviation, flags)
        return table

    def _diagnostics_table(self):
        table = MarkdownTable(["Model", "Overfit", "Overfit loss", "Max batch", "Peak (GB)"], align=["left", "left", "right", "right", "right"])
        for record in self.records:
            overfit_loss = ScalarFormatter.format_scalar(record["overfit_loss"])
            peak         = ScalarFormatter.format_scalar(record["max_batch_peak_gb"])
            table.add_row(record["model"], record["overfit_status"], overfit_loss, record["max_batch"], peak)
        return table

    def _training_table(self):
        table = MarkdownTable(["Model", "Status", "Best val loss", "Batch", "LR scale", "Duration (s)"], align=["left", "left", "right", "right", "right", "right"])
        for record in self.records:
            lr_scale = f"{record['lr_scale']:.4f}" if record.get("lr_scale") is not None else None
            table.add_row(
                record["model"],
                record["train_status"],
                ScalarFormatter.format_scalar(record["best_val_loss"]),
                record.get("train_batch_size"),
                lr_scale,
                ScalarFormatter.format_scalar(record["duration_s"]),
            )
        return table

    def _write_markdown(self, ranked):
        document = MarkdownDoc(title="Model Zoo Benchmark Report")
        document.bold_kv("Models", len(self.records))
        document.bold_kv("Reference model", self.config.size_match.reference_model)
        document.bold_kv("Ranking metric", self.rank_metric)
        document.blank()

        if ranked:
            document.heading("Leaderboard (held-out test)", level=2)
            document.table(self._leaderboard_table(ranked))

        document.heading("Capacity matching", level=2)
        document.table(self._capacity_table())

        document.heading("Overfit gate and maximum batch", level=2)
        document.table(self._diagnostics_table())

        document.heading("Training", level=2)
        document.table(self._training_table())

        report_path = self.output_directory / IoNames.benchmark_report_file
        return document.save(report_path)

    def _write_json(self):
        summary_path = self.output_directory / IoNames.benchmark_summary_file
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"records": self.records}, indent=IoNames.json_indent), encoding="utf-8")
        return summary_path

    def _log_leaderboard(self, ranked):
        rows = []
        for position, record in enumerate(ranked, start=1):
            rows.append({"Rank": position, "Model": record["model"], self.rank_metric: record["metrics"][self.rank_metric]})
        if rows:
            self.logger.metrics_table(rows, ["Rank", "Model", self.rank_metric], title="Benchmark Leaderboard")

    def write_all(self):
        self.output_directory.mkdir(parents=True, exist_ok=True)
        ranked = self._ranked_records()

        report_path  = self._write_markdown(ranked)
        summary_path = self._write_json()
        self._log_leaderboard(ranked)

        return {"report_path": str(report_path), "summary_path": str(summary_path), "ranked": ranked}
