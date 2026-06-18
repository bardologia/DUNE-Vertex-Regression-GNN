from __future__ import annotations

import json
from pathlib import Path

from tools.reporting.markdown import MarkdownDoc, MarkdownTable, ScalarFormatter


class TrialCollector:
    def __init__(self, pipeline_directory, models, logger):
        self.pipeline_directory = Path(pipeline_directory)
        self.models             = list(models)
        self.logger             = logger

    def _load(self, file_name):
        path = self.pipeline_directory / file_name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _as_model_map(self, payload):
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return payload
        return {record["model"]: record for record in payload}

    def _record_for(self, model_name, size_match, overfit, max_batch, training, evaluation):
        size_entry     = size_match.get(model_name, {})
        overfit_entry  = overfit.get(model_name, {})
        max_batch_entry = max_batch.get(model_name, {})
        training_entry = training.get(model_name, {})
        metrics        = evaluation.get(model_name, {})

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
            "final_test_loss"     : training_entry.get("final_test_loss"),
            "duration_s"          : training_entry.get("duration_s"),
            "run_directory"       : training_entry.get("run_directory"),
            "metrics"             : metrics,
        }

    def collect(self):
        size_match  = self._as_model_map(self._load("size_match.json"))
        overfit     = self._as_model_map(self._load("overfit_results.json"))
        max_batch   = self._as_model_map(self._load("max_batch.json"))
        training    = self._as_model_map(self._load("training_results.json"))
        evaluation  = self._as_model_map(self._load("evaluation_results.json"))

        return [self._record_for(model_name, size_match, overfit, max_batch, training, evaluation) for model_name in self.models]


class ComparisonReport:
    LEADERBOARD_METRICS = ("euclidean_mean", "euclidean_median", "mae", "rmse", "median_error", "r2")

    def __init__(self, records, output_directory, configuration, logger):
        self.records          = records
        self.output_directory = Path(output_directory)
        self.config           = configuration
        self.logger           = logger
        self.rank_metric      = configuration.comparison.rank_metric

    def _capacity_table(self):
        table = MarkdownTable(["Model", "Parameters", "Width", "Deviation", "Flags"], align=["left", "right", "right", "right", "left"])
        for record in self.records:
            deviation = f"{record['size_deviation_pct']:+.2f} %" if record.get("size_deviation_pct") is not None else "—"
            flags     = "; ".join(record.get("size_flags") or []) or "—"
            parameters = f"{record['parameters']:,}" if record.get("parameters") is not None else "—"
            table.add_row(record["model"], parameters, record.get("width") or "—", deviation, flags)
        return table

    def _diagnostics_table(self):
        table = MarkdownTable(["Model", "Overfit", "Overfit loss", "Max batch", "Peak (GB)"], align=["left", "left", "right", "right", "right"])
        for record in self.records:
            overfit_loss = ScalarFormatter.format_scalar(record.get("overfit_loss"))
            peak         = ScalarFormatter.format_scalar(record.get("max_batch_peak_gb"))
            table.add_row(record["model"], record.get("overfit_status") or "—", overfit_loss, record.get("max_batch") or "—", peak)
        return table

    def _training_table(self):
        table = MarkdownTable(["Model", "Status", "Best val loss", "Final test loss", "Duration (s)"], align=["left", "left", "right", "right", "right"])
        for record in self.records:
            table.add_row(
                record["model"],
                record.get("train_status") or "—",
                ScalarFormatter.format_scalar(record.get("best_val_loss")),
                ScalarFormatter.format_scalar(record.get("final_test_loss")),
                ScalarFormatter.format_scalar(record.get("duration_s")),
            )
        return table

    def _ranked_records(self):
        scored = [record for record in self.records if record.get("metrics", {}).get(self.rank_metric) is not None]
        ascending = self.rank_metric != "r2"
        return sorted(scored, key=lambda record: record["metrics"][self.rank_metric], reverse=not ascending)

    def _leaderboard_table(self, ranked):
        table = MarkdownTable(["Rank", "Model", *self.LEADERBOARD_METRICS], align=["right", "left"] + ["right"] * len(self.LEADERBOARD_METRICS))
        for position, record in enumerate(ranked, start=1):
            cells = [ScalarFormatter.format_scalar(record["metrics"].get(key)) for key in self.LEADERBOARD_METRICS]
            table.add_row(position, record["model"], *cells)
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

        report_path = self.output_directory / "benchmark_report.md"
        return document.save(report_path)

    def _write_json(self):
        summary_path = self.output_directory / "benchmark_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"records": self.records}, indent=2), encoding="utf-8")
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
