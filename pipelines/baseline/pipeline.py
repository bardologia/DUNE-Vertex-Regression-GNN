from __future__ import annotations

import json

from datetime import datetime
from pathlib  import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

from optuna.samplers import TPESampler

from tools.reporting.markdown      import MarkdownDoc, MarkdownTable
from tools.runtime.config_cli      import ConfigCli
from tools.runtime.reproducibility import Reproducibility

from pipelines.inference.metrics import InferenceMetrics
from pipelines.inference.plots   import PublicationStyle

from pipelines.baseline.boosters import BASELINE_MODEL_FAMILIES
from pipelines.baseline.features import TabularSplitBuilder


class BoosterTuner:
    def __init__(self, family, booster_config, search_config, seed, splits, logger):
        self.family  = family
        self.booster = booster_config
        self.search  = search_config
        self.seed    = seed
        self.logger  = logger

        self.train_features, self.train_targets           = splits["train"]
        self.validation_features, self.validation_targets = splits["val"]

    def _objective(self, trial):
        params = self.family.search_space(trial)
        model  = self.family(self.booster, self.seed).fit(self.train_features, self.train_targets, self.validation_features, self.validation_targets, params)

        predictions = model.predict(self.validation_features)
        return float(np.mean(np.linalg.norm(predictions - self.validation_targets, axis=1)))

    def optimize(self):
        if self.search.startup_trials >= self.search.n_trials:
            raise ValueError(f"startup_trials ({self.search.startup_trials}) must be smaller than n_trials ({self.search.n_trials}) for the TPE sampler to ever model the objective; otherwise every trial is random search.")

        sampler = TPESampler(seed=self.search.random_state, n_startup_trials=self.search.startup_trials)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self._objective, n_trials=self.search.n_trials)

        self.logger.subsection(f"{self.family.LABEL} best val euclidean mean: {study.best_value:.4f} m (trial {study.best_trial.number})")
        self.logger.kv_table(study.best_params, title=f"{self.family.LABEL} Best Hyperparameters")
        return study


class BaselineRunWriter:
    TOP_IMPORTANCES = 25

    def __init__(self, run_directory, logger):
        self.run_directory      = Path(run_directory)
        self.metadata_directory = self.run_directory / "metadata"
        self.checkpoint_dir     = self.run_directory / "checkpoints"
        self.analysis_directory = self.run_directory / "analysis"
        self.logger             = logger

        if self.run_directory.exists() and any(self.run_directory.iterdir()):
            raise FileExistsError(f"Run directory {self.run_directory} already exists and is not empty; remove it or choose a different run_name so metrics, models, and reports cannot be mixed across separate runs.")

        for directory in (self.metadata_directory, self.checkpoint_dir, self.analysis_directory):
            directory.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path, payload):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_resolved_config(self, model_name, entry_config):
        return self._write_json(self.metadata_directory / "resolved_config.json", {"model_name": model_name, **ConfigCli.to_mapping(entry_config)})

    def write_split(self, split_base_ids):
        return self._write_json(self.metadata_directory / "split_base_ids.json", split_base_ids)

    def write_best_trial(self, record):
        return self._write_json(self.metadata_directory / "best_trial.json", record)

    def write_metrics(self, per_split_metrics):
        return self._write_json(self.metadata_directory / "metrics.json", {"unit": "m", "splits": per_split_metrics})

    def _plot_importances(self, label, ranked):
        PublicationStyle.apply()

        names  = [name for name, _ in ranked][::-1]
        shares = [share for _, share in ranked][::-1]

        figure, axes = plt.subplots(figsize=(7, 8))
        axes.barh(names, shares, color="#4C72B0")
        axes.set_xlabel("Mean gain share across coordinate models")
        axes.set_title(f"{label} feature importance")

        figure.tight_layout()
        figure.savefig(self.analysis_directory / "feature_importance.png", bbox_inches="tight")
        plt.close(figure)

    def write_importances(self, label, importances):
        ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        self._write_json(self.analysis_directory / "feature_importance.json", dict(ranked))
        self._plot_importances(label, ranked[: self.TOP_IMPORTANCES])

    def write_report(self, family, entry_config, study, model, per_split_metrics):
        document = MarkdownDoc(f"{family.LABEL} baseline")

        document.paragraph(f"Tuned {family.LABEL} baseline for vertex regression: one booster per coordinate, trained on summary features of the realized light with early stopping on the validation split. The training split is octant-expanded when data.augment_octants is on; validation and test contain clean base events only.")

        document.kv_table({
            "trials"             : len(study.trials),
            "best_trial"         : study.best_trial.number,
            "best_val_euclidean" : f"{study.best_value:.4f} m",
            "best_iterations"    : ", ".join(str(value) for value in model.best_iterations()),
            "octants"            : entry_config.dataset.data.augment_octants,
            "seed"               : entry_config.seed,
        }, header=("Search", "Value"))

        document.heading("Best hyperparameters", level=2)
        document.kv_table(study.best_params)

        document.heading("Metrics", level=2)
        table = MarkdownTable(["Split", "Euclidean mean [m]", "Euclidean median [m]", "P90 [m]", "MAE x [m]", "MAE y [m]", "MAE z [m]"], align=["left"] + ["right"] * 6)
        for split_name, metrics in per_split_metrics.items():
            table.add_row(split_name, f"{metrics['euclidean_mean']:.4f}", f"{metrics['euclidean_median']:.4f}", f"{metrics['euclidean_p90']:.4f}", f"{metrics['mae_x']:.4f}", f"{metrics['mae_y']:.4f}", f"{metrics['mae_z']:.4f}")
        document.table(table)

        document.heading("Feature importance", level=2)
        document.image("feature importance", "feature_importance.png")

        report_path = document.save(self.analysis_directory / "report.md")
        self.logger.subsection(f"Baseline report written to {report_path}")
        return report_path


class BaselinePipeline:
    SPLIT_ORDER = ("train", "val", "test")

    def __init__(self, entry_config, logger):
        self.entry  = entry_config
        self.logger = logger

        stamp              = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.resolved_name = entry_config.run_name if entry_config.run_name else stamp

        self.builder = None
        self.splits  = None

    def _validate_models(self):
        models  = tuple(self.entry.models)
        unknown = tuple(key for key in models if key not in BASELINE_MODEL_FAMILIES)

        if not models:
            raise ValueError(f"models is empty; select at least one of {tuple(BASELINE_MODEL_FAMILIES)}.")
        if unknown:
            raise KeyError(f"Unknown baseline model(s) {unknown}. Expected any of {tuple(BASELINE_MODEL_FAMILIES)}.")
        return models

    def _prepare_data(self):
        Reproducibility.seed_everything(self.entry.seed)

        self.builder = TabularSplitBuilder(self.entry.dataset, self.entry.baseline.features, self.logger)
        self.splits  = self.builder.run()

    def _tune(self, family):
        self.logger.section(f"[Hyperparameter Search | {family.LABEL}]")
        return BoosterTuner(family, self.entry.baseline.booster, self.entry.baseline.search, self.entry.seed, self.splits, self.logger).optimize()

    def _final_fit(self, family, best_params):
        self.logger.section(f"[Final Fit | {family.LABEL}]")

        train_features, train_targets           = self.splits["train"]
        validation_features, validation_targets = self.splits["val"]
        return family(self.entry.baseline.booster, self.entry.seed).fit(train_features, train_targets, validation_features, validation_targets, best_params)

    def _evaluate(self, family, model):
        per_split_metrics = {}
        for split_name in self.SPLIT_ORDER:
            features, targets            = self.splits[split_name]
            predictions                  = model.predict(features)
            per_split_metrics[split_name] = InferenceMetrics(self.logger).compute(torch.from_numpy(predictions), torch.from_numpy(targets), stage=f"{family.KEY}/{split_name}")
        return per_split_metrics

    def _write_run(self, family, study, model, per_split_metrics):
        run_directory = Path(self.entry.baseline.io.log_base_dir) / f"baseline_{family.KEY}_{self.resolved_name}"
        writer        = BaselineRunWriter(run_directory, self.logger)

        writer.write_resolved_config(f"baseline_{family.KEY}", self.entry)
        writer.write_split(self.builder.split_base_ids)
        writer.write_best_trial({
            "model_name"        : f"baseline_{family.KEY}",
            "best_value"        : study.best_value,
            "best_trial_number" : study.best_trial.number,
            "random_state"      : self.entry.baseline.search.random_state,
            "n_trials"          : self.entry.baseline.search.n_trials,
            "best_params"       : study.best_params,
            "best_iterations"   : model.best_iterations(),
        })
        writer.write_metrics(per_split_metrics)
        writer.write_importances(family.LABEL, model.importances(self.builder.feature_names))
        writer.write_report(family, self.entry, study, model, per_split_metrics)
        model.save(writer.checkpoint_dir)

        self.logger.subsection(f"Baseline run written to {run_directory}")
        return run_directory

    def _log_comparison(self, summary):
        rows = []
        for key, payload in summary.items():
            rows.append({
                "Model"           : BASELINE_MODEL_FAMILIES[key].LABEL,
                "Val euclidean"   : f"{payload['metrics']['val']['euclidean_mean']:.4f}",
                "Test euclidean"  : f"{payload['metrics']['test']['euclidean_mean']:.4f}",
                "Test median"     : f"{payload['metrics']['test']['euclidean_median']:.4f}",
                "Test p90"        : f"{payload['metrics']['test']['euclidean_p90']:.4f}",
            })
        self.logger.metrics_table(rows, ["Model", "Val euclidean", "Test euclidean", "Test median", "Test p90"], title="Baseline Comparison [m]")

        strongest = min(summary, key=lambda key: summary[key]["metrics"]["val"]["euclidean_mean"])
        self.logger.subsection(f"Strongest baseline by val euclidean mean: {BASELINE_MODEL_FAMILIES[strongest].LABEL} (test euclidean mean {summary[strongest]['metrics']['test']['euclidean_mean']:.4f} m)")
        return strongest

    def run(self):
        self.logger.section("[Baseline Pipeline]")

        models = self._validate_models()
        self._prepare_data()

        summary = {}
        for key in models:
            family            = BASELINE_MODEL_FAMILIES[key]
            study             = self._tune(family)
            model             = self._final_fit(family, study.best_params)
            per_split_metrics = self._evaluate(family, model)
            run_directory     = self._write_run(family, study, model, per_split_metrics)

            summary[key] = {"run_directory": str(run_directory), "best_value": study.best_value, "metrics": per_split_metrics}

        summary["strongest"] = self._log_comparison(summary)
        return summary
