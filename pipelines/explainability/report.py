from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.reporting.markdown import MarkdownDoc, MarkdownTable


class FeatureImportanceReport:
    UNIT = "cm"

    CSV_COLUMNS = (
        "feature", "group",
        "permutation_delta", "permutation_std", "permutation_pct_increase",
        "permutation_delta_mae", "permutation_delta_rmse", "permutation_delta_r2",
        "occlusion_delta", "occlusion_delta_mae", "occlusion_delta_rmse", "occlusion_delta_r2",
        "gradient_saliency", "gradient_input_times_grad",
        "consensus_mean_rank",
    )

    def __init__(self, output_directory, logger, primary_metric, top_k):
        self.output_directory = Path(output_directory)
        self.logger           = logger
        self.primary_metric   = primary_metric
        self.top_k            = top_k
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _write_json(self, payload):
        path = self.output_directory / "results.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_csv(self, name, rows):
        path = self.output_directory / name
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.CSV_COLUMNS)
            for row in rows:
                writer.writerow([row.get(column) for column in self.CSV_COLUMNS])
        return path

    def _introduction(self, document):
        document.paragraph("Feature importance of the event-graph representation for a trained DUNE-GNN vertex regressor. Three complementary attributions are reported, each evaluated on the held-out split with the training-split normalization statistics held fixed.")
        document.paragraph("Permutation importance shuffles a feature column across every node (or edge) of the split and measures the resulting degradation of the held-out metric; it captures how much the model relies on the association between that feature and the target. Occlusion importance replaces a feature with its dataset mean, isolating the effect of removing its information while preserving the input scale. Gradient saliency differentiates the mean squared prediction error with respect to the normalized input features, reporting the local sensitivity of the objective to each feature. Permutation and occlusion degradations are expressed in physical units; gradient quantities are computed in normalized feature space, where channels are directly comparable.")

    def _configuration_table(self, document, context, baseline):
        document.heading("Configuration", level=2)
        document.kv_table({
            "Run directory"        : context["run_directory"],
            "Model"                : context["model_name"],
            "Checkpoint"           : context["checkpoint"],
            "Evaluated split"      : context["split"],
            "Split events"         : context["event_count"],
            "Node feature columns" : context["node_feature_count"],
            "Edge feature columns" : context["edge_feature_count"],
            "Permutation repeats"  : context["permutation_repeats"],
            "Primary metric"       : self.primary_metric,
            "Normalization stats"  : "training split (reused)",
        })

        document.heading("Baseline performance", level=2)
        document.kv_table({
            "Mean euclidean error [cm]"   : f"{baseline['euclidean_mean']:.4f}",
            "Median euclidean error [cm]" : f"{baseline['euclidean_median']:.4f}",
            "Mean absolute error [cm]"    : f"{baseline['mae']:.4f}",
            "Root-mean-square error [cm]" : f"{baseline['rmse']:.4f}",
            "Coefficient of determination": f"{baseline['r2']:.4f}",
        }, code_keys=False)

    def _inventory_table(self, document, title, inventory):
        document.heading(title, level=3)
        table = MarkdownTable(["Index", "Feature", "Group"], align=["right", "left", "left"])
        for index, (name, group) in enumerate(inventory):
            table.add_row(index, name, group)
        document.table(table)

    def _feature_inventory(self, document, inventory):
        document.heading("Feature inventory", level=2)
        self._inventory_table(document, "Node features", inventory["node"])
        self._inventory_table(document, "Edge features", inventory["edge"])

    def _perturbation_table(self, document, title, records):
        document.heading(title, level=3)
        table = MarkdownTable(
            ["Feature", "Group", f"Delta {self.primary_metric} [cm]", "Std [cm]", "Increase [%]", "Delta MAE [cm]", "Delta RMSE [cm]", "Delta R2"],
            align=["left", "left", "right", "right", "right", "right", "right", "right"],
        )
        ranked = sorted(records, key=lambda record: record["deltas"][self.primary_metric], reverse=True)
        for record in ranked:
            deltas = record["deltas"]
            table.add_row(
                record["label"],
                record["group"],
                f"{deltas[self.primary_metric]:.4f}",
                f"{record['primary_std']:.4f}",
                f"{record['pct_increase']:.2f}",
                f"{deltas['mae']:.4f}",
                f"{deltas['rmse']:.4f}",
                f"{deltas['r2']:.4f}",
            )
        document.table(table)

    def _permutation_section(self, document, results):
        document.heading("Permutation importance", level=2)
        self._perturbation_table(document, "Node features", results["node_features"])
        if results["node_groups"]:
            self._perturbation_table(document, "Node feature groups", results["node_groups"])
        self._perturbation_table(document, "Edge features", results["edge_features"])
        if results["edge_groups"]:
            self._perturbation_table(document, "Edge feature groups", results["edge_groups"])

    def _occlusion_section(self, document, results):
        document.heading("Occlusion (mean-ablation) importance", level=2)
        self._perturbation_table(document, "Node features", results["node_features"])
        if results["node_groups"]:
            self._perturbation_table(document, "Node feature groups", results["node_groups"])
        self._perturbation_table(document, "Edge features", results["edge_features"])
        if results["edge_groups"]:
            self._perturbation_table(document, "Edge feature groups", results["edge_groups"])

    def _gradient_table(self, document, title, records):
        document.heading(title, level=3)
        table  = MarkdownTable(["Feature", "Group", "Mean |grad|", "Mean |input x grad|"], align=["left", "left", "right", "right"])
        ranked = sorted(records, key=lambda record: record["saliency"], reverse=True)
        for record in ranked:
            table.add_row(record["feature"], record["group"], f"{record['saliency']:.6g}", f"{record['input_times_grad']:.6g}")
        document.table(table)

    def _gradient_section(self, document, gradient):
        document.heading("Gradient saliency", level=2)
        self._gradient_table(document, "Node features", gradient["node"])
        self._gradient_table(document, "Edge features", gradient["edge"])

    def _consensus_table(self, document, title, records):
        document.heading(title, level=3)
        table = MarkdownTable(["Rank", "Feature", "Group", "Mean method rank", "Methods"], align=["right", "left", "left", "right", "left"])
        for position, record in enumerate(records[: self.top_k], start=1):
            table.add_row(position, record["feature"], record["group"], f"{record['mean_rank']:.2f}", ", ".join(record["methods"]))
        document.table(table)

    def _consensus_section(self, document, consensus):
        document.heading("Consensus ranking", level=2)
        document.paragraph("Per-feature rank averaged across the available methods (lower is more important). Each method ranks features by its own primary quantity; the mean rank summarizes agreement across methods.")
        self._consensus_table(document, "Node features", consensus["node"])
        self._consensus_table(document, "Edge features", consensus["edge"])

    def _figures(self, document, plot_filenames):
        document.heading("Figures", level=2)
        for filename in plot_filenames:
            document.image(filename, f"plots/{filename}")

    def build(self, context, baseline, inventory, analysis, merged_csv, plot_filenames):
        json_path = self._write_json({"unit": self.UNIT, "primary_metric": self.primary_metric, "run": context, "baseline": baseline, "feature_inventory": inventory, "analysis": analysis})
        node_csv  = self._write_csv("node_importance.csv", merged_csv["node"])
        edge_csv  = self._write_csv("edge_importance.csv", merged_csv["edge"])

        document = MarkdownDoc("Graph Feature Importance")
        self._introduction(document)
        self._configuration_table(document, context, baseline)
        self._feature_inventory(document, inventory)

        if analysis.get("permutation") is not None:
            self._permutation_section(document, analysis["permutation"])
        if analysis.get("occlusion") is not None:
            self._occlusion_section(document, analysis["occlusion"])
        if analysis.get("gradient") is not None:
            self._gradient_section(document, analysis["gradient"])

        self._consensus_section(document, analysis["consensus"])
        self._figures(document, plot_filenames)

        report_path = self.output_directory / "report.md"
        document.save(report_path)

        self.logger.subsection(f"Feature importance report written to {report_path}")
        self.logger.subsection(f"Machine-readable results written to {json_path}")
        self.logger.subsection(f"Per-feature tables written to {node_csv} and {edge_csv}")
        return report_path
