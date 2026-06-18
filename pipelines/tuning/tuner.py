from __future__ import annotations

from copy    import deepcopy
from pathlib import Path

import optuna
from optuna.pruners  import MedianPruner
from optuna.samplers import TPESampler

from configuration.architectures      import MODEL_CONFIG_REGISTRY
from configuration.training.general   import OptimizerConfig
from models                           import get_model
from tools.runtime.reproducibility import Reproducibility
from pipelines.dataset.pipeline   import DatasetPipeline
from pipelines.shared.run_metadata import LightweightRunContext
from pipelines.training.trainer   import Trainer


class Tuner:
    def __init__(self, entry_config, logger):
        self.entry            = entry_config
        self.logger           = logger
        self.tuning           = entry_config.tuning
        self.training_config  = entry_config.training
        self.tuner_directory  = Path(entry_config.training.io.tuner_base_dir) / entry_config.model_name
        self.tuner_directory.mkdir(parents=True, exist_ok=True)
        self.study            = None

        self.model_space     = MODEL_CONFIG_REGISTRY[entry_config.model_name].tunable_params()
        self.optimizer_space = OptimizerConfig.tunable_params()

    def _prepare_data(self):
        self.dataset_pipeline     = DatasetPipeline(self.entry.dataset, self.logger)
        self.datasets, self.stats = self.dataset_pipeline.run()
        self.train_loader, self.val_loader, _ = DatasetPipeline.build_loaders(self.datasets, self.tuning.batch_size, num_workers=0)

    def _suggest(self, trial):
        return self._sample(trial, self.model_space), self._sample(trial, self.optimizer_space)

    def _sample(self, trial, space):
        sampled = {}
        for name, specification in space.items():
            kind = specification["type"]
            if kind == "float":
                sampled[name] = trial.suggest_float(name, specification["low"], specification["high"], step=specification.get("step"), log=specification.get("log", False))
            elif kind == "int":
                sampled[name] = trial.suggest_int(name, specification["low"], specification["high"])
            elif kind == "categorical":
                sampled[name] = trial.suggest_categorical(name, specification["choices"])
            else:
                raise ValueError(f"Unknown tunable type '{kind}' for '{name}'")
        return sampled

    def _objective(self, trial):
        Reproducibility.seed_everything(self.tuning.random_state + trial.number)

        model_overrides, optimizer_overrides = self._suggest(trial)

        training_config = deepcopy(self.training_config)
        for field_name, value in optimizer_overrides.items():
            setattr(training_config.optimizer, field_name, value)
        training_config.loop.epochs      = self.tuning.epochs
        training_config.loop.tuning_mode = True
        training_config.loop.verbose     = False
        training_config.loop.batch_size  = self.tuning.batch_size

        degree_histogram = self.dataset_pipeline.pna_degree_histogram(self.entry.model_name, self.datasets["train"])
        if degree_histogram is not None:
            model_overrides = {**model_overrides, "degree_histogram": degree_histogram}

        model, _ = get_model(self.entry.model_name, **model_overrides)
        context  = LightweightRunContext(self.logger, self.tuner_directory / f"trial_{trial.number}.pt")
        trainer  = Trainer(model, self.stats, training_config, context)

        best_validation_loss = float("inf")
        for epoch in range(self.tuning.epochs):
            _, validation_loss   = trainer.run_epoch(self.train_loader, self.val_loader, epoch)
            best_validation_loss = min(best_validation_loss, validation_loss)

            trial.report(validation_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_validation_loss

    def build_best_overrides(self):
        best_parameters     = self.study.best_params
        optimizer_keys      = set(self.optimizer_space)
        model_overrides     = {key: value for key, value in best_parameters.items() if key not in optimizer_keys}
        optimizer_overrides = {key: value for key, value in best_parameters.items() if key in optimizer_keys}
        return model_overrides, optimizer_overrides

    def optimize(self):
        self.logger.section("[Hyperparameter Tuning]")

        if self.tuning.warmup_trials >= self.tuning.n_trials:
            raise ValueError(f"warmup_trials ({self.tuning.warmup_trials}) must be smaller than n_trials ({self.tuning.n_trials}) for the median pruner to ever activate.")

        self._prepare_data()

        sampler = TPESampler(seed=self.tuning.random_state)
        pruner  = MedianPruner(n_startup_trials=self.tuning.warmup_trials, n_warmup_steps=self.tuning.warmup_steps)
        self.study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        self.study.optimize(self._objective, n_trials=self.tuning.n_trials)

        self.logger.subsection(f"Best value: {self.study.best_value:.4f}")
        self.logger.kv_table(self.study.best_params, title="Best Hyperparameters")
        return self.study
