from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

import numpy as np

from configuration.entry             import TrainEntryConfig
from models                          import get_model
from tools.monitoring.logger         import Logger, NullLogger
from tools.runtime.config_cli        import ConfigCli
from pipelines.dataset.graph         import FeatureLayout
from pipelines.dataset.normalization import NormalizationStats
from pipelines.dataset.pipeline      import DatasetPipeline
from pipelines.inference.metrics     import InferenceMetrics
from pipelines.inference.predictor   import Predictor

from pipelines.explainability.evaluation import EvaluationGraphTensors, SplitMetricEvaluator
from pipelines.explainability.importance import ExpectedGradients, GradientSaliency, OcclusionImportance, PermutationImportance
from pipelines.explainability.plots      import FeatureImportancePlots
from pipelines.explainability.report     import FeatureImportanceReport


class FeatureImportancePipeline:
    METRIC_KEYS = ("euclidean_mean", "euclidean_median", "mae", "rmse", "r2")

    def __init__(self, config, logger=None):
        self.config           = config
        self.run_directory    = Path(config.run_directory)
        self.logger           = logger if logger is not None else Logger(log_dir=str(self.run_directory / "logs"), name="explain")
        self.device           = config.device
        self.split            = config.split
        self.batch_size       = config.batch_size
        self.primary_metric   = config.primary_metric

        self.entry            = None
        self.stats            = None
        self.split_base_ids   = None
        self.predictor        = None
        self.engine           = None
        self.evaluator        = None
        self.baseline         = None

        self.node_layout      = None
        self.edge_layout      = None
        self.node_columns     = None
        self.edge_columns     = None
        self.node_groups      = None
        self.edge_groups      = None

        self.permutation         = None
        self.occlusion           = None
        self.gradient            = None
        self.expected_gradients  = None
        self.output_directory    = None

    def _resolve_output_directory(self):
        stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        label     = f"feature_importance_{self.config.run_name}_{stamp}" if self.config.run_name else f"feature_importance_{stamp}"
        directory = self.run_directory / "explainability" / label
        directory.mkdir(parents=True, exist_ok=True)
        self.output_directory = directory

    def _load_run(self):
        resolved_path = self.run_directory / "metadata" / "resolved_config.json"
        if not resolved_path.exists():
            raise FileNotFoundError(f"No training run found at {self.run_directory}: expected {resolved_path}. Pass --run_directory pointing at a completed run (runs/<model>_<stamp>).")

        self.entry          = ConfigCli.load_resolved(TrainEntryConfig(), resolved_path)
        self.stats          = NormalizationStats.load(self.run_directory / "metadata")
        self.split_base_ids = json.loads((self.run_directory / "metadata" / "split_base_ids.json").read_text(encoding="utf-8"))

        layout            = FeatureLayout(self.entry.dataset)
        self.node_layout  = layout.node_features()
        self.edge_layout  = layout.edge_features()
        self.node_columns = [(index, name, group) for index, (name, group) in enumerate(self.node_layout)]
        self.edge_columns = [(index, name, group) for index, (name, group) in enumerate(self.edge_layout)]
        self.node_groups  = self._multi_member_groups(self.node_layout)
        self.edge_groups  = self._multi_member_groups(self.edge_layout)

        self.logger.kv_table({
            "Run directory" : str(self.run_directory),
            "Model"         : self.entry.model_name,
            "Device"        : self.device,
            "Split"         : self.split,
            "Node features" : len(self.node_layout),
            "Edge features" : len(self.edge_layout),
            "Output"        : str(self.output_directory),
        }, title="Graph Feature Importance")

    def _multi_member_groups(self, layout):
        groups = FeatureLayout.group_index(layout)
        return {group: indices for group, indices in groups.items() if len(indices) > 1}

    def _build_predictor(self):
        model, _        = get_model(self.entry.model_name, **self.entry.model_overrides)
        checkpoint_path = self.run_directory / "checkpoints" / self.entry.training.io.checkpoint_name
        self.predictor  = Predictor(model, checkpoint_path, self.stats, self.device, self.logger).prepare()
        self.predictor.logger = NullLogger()

    def _prepare_engine(self):
        dataset_pipeline = DatasetPipeline(self.entry.dataset, self.logger, stats=self.stats, evaluation_mode=True)
        dataset_pipeline.prepare_samples()
        dataset          = dataset_pipeline.build_evaluation_split(self.split_base_ids, self.split)

        self.engine = EvaluationGraphTensors(dataset, self.batch_size, self.config.max_events, self.logger).prepare()
        self._assert_layout_consistency()

        self.evaluator = SplitMetricEvaluator(self.predictor, self.METRIC_KEYS)

        predictions, targets = self.predictor.predict(self.engine.loader())
        self.baseline        = InferenceMetrics(self.logger).compute(predictions, targets, stage="baseline")

    def _assert_layout_consistency(self):
        if self.engine.node_feature_count() != len(self.node_layout):
            raise ValueError(f"Node feature layout ({len(self.node_layout)}) disagrees with the materialized graphs ({self.engine.node_feature_count()}). The run's GraphConfig and the feature layout are out of sync.")
        if self.engine.edge_feature_count() != len(self.edge_layout):
            raise ValueError(f"Edge feature layout ({len(self.edge_layout)}) disagrees with the materialized graphs ({self.engine.edge_feature_count()}). The run's GraphConfig and the feature layout are out of sync.")

    def _run_permutation(self):
        if not self.config.permutation:
            return
        node_groups      = self.node_groups if self.config.grouped else {}
        edge_groups      = self.edge_groups if self.config.grouped else {}
        estimator        = PermutationImportance(self.engine, self.evaluator, self.baseline, self.METRIC_KEYS, self.primary_metric, self.config.permutation_seed, self.config.permutation_repeats, self.logger)
        self.permutation = estimator.run(self.node_columns, node_groups, self.edge_columns, edge_groups)

    def _run_occlusion(self):
        if not self.config.occlusion:
            return
        node_groups    = self.node_groups if self.config.grouped else {}
        edge_groups    = self.edge_groups if self.config.grouped else {}
        estimator      = OcclusionImportance(self.engine, self.evaluator, self.baseline, self.METRIC_KEYS, self.primary_metric, self.config.permutation_seed, self.logger)
        self.occlusion = estimator.run(self.node_columns, node_groups, self.edge_columns, edge_groups)

    def _run_gradient(self):
        if not self.config.gradient:
            return
        means         = GradientSaliency(self.predictor.model, self.device, self.engine, self.logger).run()
        self.gradient = {
            "node" : self._gradient_records(self.node_layout, means["node_saliency"], means["node_input_times_grad"]),
            "edge" : self._gradient_records(self.edge_layout, means["edge_saliency"], means["edge_input_times_grad"]),
        }

    def _gradient_records(self, layout, saliency, input_times_grad):
        records = []
        for index, (name, group) in enumerate(layout):
            records.append({"feature": name, "group": group, "saliency": float(saliency[index]), "input_times_grad": float(input_times_grad[index])})
        return records

    def _run_expected_gradients(self):
        if not self.config.expected_gradients:
            return
        means                   = ExpectedGradients(self.predictor.model, self.device, self.engine, self.config.eg_samples, self.config.eg_events, self.config.eg_seed, self.logger).run()
        self.expected_gradients = {
            "coordinates"  : list(means["coordinates"]),
            "completeness" : means["completeness"],
            "events"       : means["events"],
            "samples"      : means["samples"],
            "node"         : self._expected_gradient_records(self.node_layout, means["coordinates"], means["node_overall"], means["node_per_axis"]),
            "edge"         : self._expected_gradient_records(self.edge_layout, means["coordinates"], means["edge_overall"], means["edge_per_axis"]),
        }

    def _expected_gradient_records(self, layout, coordinates, overall, per_axis):
        records = []
        for index, (name, group) in enumerate(layout):
            record = {"feature": name, "group": group, "overall": float(overall[index])}
            for axis_index, axis in enumerate(coordinates):
                record[axis] = float(per_axis[axis_index][index])
            records.append(record)
        return records

    def _perturbation_lookup(self, results, key):
        if results is None:
            return {}
        return {record["label"]: record for record in results[key]}

    def _gradient_lookup(self, domain):
        if self.gradient is None:
            return {}
        return {record["feature"]: record for record in self.gradient[domain]}

    def _expected_gradient_lookup(self, domain):
        if self.expected_gradients is None:
            return {}
        return {record["feature"]: record for record in self.expected_gradients[domain]}

    def _rank_map(self, scores):
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {feature: position for position, (feature, _) in enumerate(ranked, start=1)}

    def _merged_rows(self, layout, domain, permutation_key):
        permutation_features        = self._perturbation_lookup(self.permutation, permutation_key)
        occlusion_features          = self._perturbation_lookup(self.occlusion, permutation_key)
        gradient_features           = self._gradient_lookup(domain)
        expected_gradient_features  = self._expected_gradient_lookup(domain)

        rows = []
        for name, group in layout:
            row = {"feature": name, "group": group}

            if name in permutation_features:
                record = permutation_features[name]
                row["permutation_delta"]        = record["deltas"][self.primary_metric]
                row["permutation_std"]          = record["primary_std"]
                row["permutation_pct_increase"] = record["pct_increase"]
                row["permutation_delta_mae"]    = record["deltas"]["mae"]
                row["permutation_delta_rmse"]   = record["deltas"]["rmse"]
                row["permutation_delta_r2"]     = record["deltas"]["r2"]

            if name in occlusion_features:
                record = occlusion_features[name]
                row["occlusion_delta"]      = record["deltas"][self.primary_metric]
                row["occlusion_delta_mae"]  = record["deltas"]["mae"]
                row["occlusion_delta_rmse"] = record["deltas"]["rmse"]
                row["occlusion_delta_r2"]   = record["deltas"]["r2"]

            if name in gradient_features:
                row["gradient_saliency"]         = gradient_features[name]["saliency"]
                row["gradient_input_times_grad"] = gradient_features[name]["input_times_grad"]

            if name in expected_gradient_features:
                record                     = expected_gradient_features[name]
                row["expected_gradient"]   = record["overall"]
                row["expected_gradient_x"] = record["x"]
                row["expected_gradient_y"] = record["y"]
                row["expected_gradient_z"] = record["z"]

            rows.append(row)
        return rows

    def _consensus(self, rows):
        permutation_scores = {row["feature"]: row["permutation_delta"] for row in rows if "permutation_delta" in row}
        occlusion_scores   = {row["feature"]: row["occlusion_delta"]   for row in rows if "occlusion_delta"   in row}
        gradient_scores    = {row["feature"]: row["gradient_saliency"] for row in rows if "gradient_saliency" in row}
        expected_scores    = {row["feature"]: row["expected_gradient"] for row in rows if "expected_gradient" in row}

        rank_maps = {"permutation": self._rank_map(permutation_scores), "occlusion": self._rank_map(occlusion_scores), "gradient": self._rank_map(gradient_scores), "expected_gradients": self._rank_map(expected_scores)}

        consensus = []
        for row in rows:
            ranks   = [(method, rank_map[row["feature"]]) for method, rank_map in rank_maps.items() if row["feature"] in rank_map]
            if not ranks:
                continue
            mean_rank = float(np.mean([rank for _, rank in ranks]))
            consensus.append({"feature": row["feature"], "group": row["group"], "mean_rank": mean_rank, "methods": [method for method, _ in ranks]})
            row["consensus_mean_rank"] = mean_rank

        return sorted(consensus, key=lambda record: record["mean_rank"])

    def _delta_figure(self, records, title, filename, color, with_errors):
        return {
            "names"    : [record["label"] for record in records],
            "values"   : [record["deltas"][self.primary_metric] for record in records],
            "errors"   : [record["primary_std"] for record in records] if with_errors else None,
            "xlabel"   : "Delta mean euclidean error [cm]",
            "title"    : title,
            "filename" : filename,
            "color"    : color,
        }

    def _gradient_figure(self, records, value_key, title, filename, color):
        return {
            "names"    : [record["feature"] for record in records],
            "values"   : [record[value_key] for record in records],
            "errors"   : None,
            "xlabel"   : "Mean magnitude (normalized feature space)",
            "title"    : title,
            "filename" : filename,
            "color"    : color,
        }

    def _expected_gradient_figure(self, records, value_key, title, filename, color):
        return {
            "names"    : [record["feature"] for record in records],
            "values"   : [record[value_key] for record in records],
            "errors"   : None,
            "xlabel"   : "Mean |expected gradient| (normalized prediction)",
            "title"    : title,
            "filename" : filename,
            "color"    : color,
        }

    def _figure_specifications(self):
        node_color = FeatureImportancePlots.NODE_COLOR
        edge_color = FeatureImportancePlots.EDGE_COLOR
        figures    = []

        if self.permutation is not None:
            figures.append(self._delta_figure(self.permutation["node_features"], "Permutation importance: node features", "permutation_node_features.png", node_color, True))
            figures.append(self._delta_figure(self.permutation["edge_features"], "Permutation importance: edge features", "permutation_edge_features.png", edge_color, True))
            if self.permutation["node_groups"]:
                figures.append(self._delta_figure(self.permutation["node_groups"], "Permutation importance: node feature groups", "permutation_node_groups.png", node_color, True))
            if self.permutation["edge_groups"]:
                figures.append(self._delta_figure(self.permutation["edge_groups"], "Permutation importance: edge feature groups", "permutation_edge_groups.png", edge_color, True))

        if self.occlusion is not None:
            figures.append(self._delta_figure(self.occlusion["node_features"], "Occlusion importance: node features", "occlusion_node_features.png", node_color, False))
            figures.append(self._delta_figure(self.occlusion["edge_features"], "Occlusion importance: edge features", "occlusion_edge_features.png", edge_color, False))

        if self.gradient is not None:
            figures.append(self._gradient_figure(self.gradient["node"], "saliency",         "Gradient saliency: node features",            "gradient_node_saliency.png",         node_color))
            figures.append(self._gradient_figure(self.gradient["node"], "input_times_grad", "Gradient x input: node features",             "gradient_node_input_times_grad.png", node_color))
            figures.append(self._gradient_figure(self.gradient["edge"], "saliency",         "Gradient saliency: edge features",            "gradient_edge_saliency.png",         edge_color))
            figures.append(self._gradient_figure(self.gradient["edge"], "input_times_grad", "Gradient x input: edge features",             "gradient_edge_input_times_grad.png", edge_color))

        if self.expected_gradients is not None:
            figures.append(self._expected_gradient_figure(self.expected_gradients["node"], "overall", "Expected gradients: node features (all axes)", "expected_gradients_node_overall.png", node_color))
            figures.append(self._expected_gradient_figure(self.expected_gradients["edge"], "overall", "Expected gradients: edge features (all axes)", "expected_gradients_edge_overall.png", edge_color))
            for axis in self.expected_gradients["coordinates"]:
                figures.append(self._expected_gradient_figure(self.expected_gradients["node"], axis, f"Expected gradients: node features ({axis})", f"expected_gradients_node_{axis}.png", node_color))
                figures.append(self._expected_gradient_figure(self.expected_gradients["edge"], axis, f"Expected gradients: edge features ({axis})", f"expected_gradients_edge_{axis}.png", edge_color))

        return figures

    def _build_outputs(self):
        node_rows = self._merged_rows(self.node_layout, "node", "node_features")
        edge_rows = self._merged_rows(self.edge_layout, "edge", "edge_features")

        consensus_node = self._consensus(node_rows)
        consensus_edge = self._consensus(edge_rows)

        plots         = FeatureImportancePlots(self.output_directory / "plots", self.logger).run(self._figure_specifications())

        context = {
            "run_directory"       : str(self.run_directory),
            "model_name"          : self.entry.model_name,
            "checkpoint"          : self.entry.training.io.checkpoint_name,
            "split"               : self.split,
            "event_count"         : len(self.engine.graphs),
            "node_feature_count"  : len(self.node_layout),
            "edge_feature_count"  : len(self.edge_layout),
            "permutation_repeats" : self.config.permutation_repeats if self.config.permutation else 0,
        }

        analysis = {
            "permutation"        : self.permutation,
            "occlusion"          : self.occlusion,
            "gradient"           : self.gradient,
            "expected_gradients" : self.expected_gradients,
            "consensus"          : {"node": consensus_node, "edge": consensus_edge},
        }

        inventory  = {"node": self.node_layout, "edge": self.edge_layout}
        merged_csv = {"node": node_rows, "edge": edge_rows}

        FeatureImportanceReport(self.output_directory, self.logger, self.primary_metric, self.config.top_k).build(context, self.baseline, inventory, analysis, merged_csv, plots)

    def run(self):
        self.logger.section("[Graph Feature Importance]")
        self._resolve_output_directory()
        self._load_run()
        self._build_predictor()
        self._prepare_engine()
        self._run_permutation()
        self._run_occlusion()
        self._run_gradient()
        self._run_expected_gradients()
        self._build_outputs()
        self.logger.ok(f"Feature importance complete -> {self.output_directory}")
        return self.output_directory
