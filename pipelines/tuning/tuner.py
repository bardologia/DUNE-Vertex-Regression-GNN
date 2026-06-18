from __future__ import annotations

from copy    import deepcopy
from pathlib import Path

import optuna
from optuna.pruners  import MedianPruner
from optuna.samplers import TPESampler

from configuration.architectures      import MODEL_CONFIG_REGISTRY
from configuration.training.general   import OptimizerConfig
from models                           import get_model
from tools.monitoring.tracker     import NullTracker
from pipelines.dataset.pipeline   import DatasetPipeline
from pipelines.training.trainer   import Trainer


class TuningTrialContext:
    def __init__(self, logger, checkpoint_path):
        self.logger             = logger
        self.tracker            = NullTracker()
        self.checkpoint_path    = checkpoint_path
        self.metadata_directory = Path(checkpoint_path).parent


class Tuner:
    def __init__(self, entry_config, logger):
        self.entry            = entry_config
        self.logger           = logger
        self.tuning           = entry_config.tuning
        self.training_config  = entry_config.training
        self.tuner_directory  = Path(entry_config.training.io.tuner_base_dir) / entry_config.model_name
        self.tuner_directory.mkdir(parents=True, exist_ok=True)
        self.study            = None

    def _prepare_data(self):
        datasets, self.stats = DatasetPipeline(self.entry.dataset, self.logger).run()
        self.train_loader, self.val_loader, _ = DatasetPipeline.build_loaders(datasets, self.tuning.batch_size, num_workers=0)

    def _suggest(self, trial):
        self.model_space     = MODEL_CONFIG_REGISTRY[self.entry.model_name].tunable_params()
        self.optimizer_space = OptimizerConfig.tunable_params()
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
        model_overrides, optimizer_overrides = self._suggest(trial)

        training_config = deepcopy(self.training_config)
        for field_name, value in optimizer_overrides.items():
            setattr(training_config.optimizer, field_name, value)
        training_config.loop.epochs               = self.tuning.epochs
        training_config.loop.tuning_mode          = True
        training_config.loop.verbose              = False
        training_config.loop.batch_size           = self.tuning.batch_size
        training_config.loop.validation_frequency = 1

        model, _ = get_model(self.entry.model_name, **model_overrides)
        context  = TuningTrialContext(self.logger, self.tuner_directory / f"trial_{trial.number}.pt")
        trainer  = Trainer(model, self.stats, training_config, context)

        best_validation_loss = float("inf")
        for epoch in range(self.tuning.epochs):
            trainer.scheduler.step(epoch)
            trainer._apply_learning_rates()
            trainer.train_epoch(self.train_loader, epoch)

            validation_results   = trainer.evaluate(self.val_loader, epoch, stage="validation")
            best_validation_loss = min(best_validation_loss, validation_results["avg_loss"])

            trial.report(validation_results["avg_loss"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_validation_loss

    def build_best_overrides(self):
        best_parameters     = self.study.best_params
        optimizer_keys      = set(OptimizerConfig.tunable_params())
        model_overrides     = {key: value for key, value in best_parameters.items() if key not in optimizer_keys}
        optimizer_overrides = {key: value for key, value in best_parameters.items() if key in optimizer_keys}
        return model_overrides, optimizer_overrides

    def optimize(self):
        self.logger.section("[Hyperparameter Tuning]")
        self._prepare_data()

        sampler = TPESampler(seed=self.tuning.random_state)
        pruner  = MedianPruner(n_startup_trials=self.tuning.warmup_trials, n_warmup_steps=self.tuning.warmup_steps)
        self.study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        self.study.optimize(self._objective, n_trials=self.tuning.n_trials)

        self.logger.subsection(f"Best value: {self.study.best_value:.4f}")
        self.logger.kv_table(self.study.best_params, title="Best Hyperparameters")
        return self.study
