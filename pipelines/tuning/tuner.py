from __future__ import annotations

from copy    import deepcopy
from pathlib import Path

import optuna
from optuna.pruners  import MedianPruner
from optuna.samplers import TPESampler

from configuration.architectures import MODEL_CONFIG_REGISTRY
from models                       import get_model
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
        default_config = MODEL_CONFIG_REGISTRY[self.entry.model_name]()

        model_overrides = {
            "hidden_dim"               : trial.suggest_categorical("hidden_dim", [64, 128, 192, 256]),
            "num_layers"               : trial.suggest_int("num_layers", 2, 6),
            "dropout"                  : trial.suggest_float("dropout", 0.0, 0.2, step=0.05),
            "drop_path_rate"           : trial.suggest_float("drop_path_rate", 0.0, 0.2, step=0.05),
            "hierarchical_feature_dim" : trial.suggest_categorical("hierarchical_feature_dim", [128, 256, 384]),
            "coord_embed_dim"          : trial.suggest_categorical("coord_embed_dim", [32, 64, 128]),
            "regression_dropout"       : trial.suggest_float("regression_dropout", 0.0, 0.2, step=0.05),
        }

        if hasattr(default_config, "heads"):
            model_overrides["heads"] = trial.suggest_categorical("heads", [2, 4, 8])

        if default_config.pooling == "sagpool_multiscale":
            model_overrides["pool_num_levels"] = trial.suggest_int("pool_num_levels", 2, 4)
            model_overrides["sag_ratio"]       = trial.suggest_float("sag_ratio", 0.3, 0.7, step=0.1)

        optimizer_overrides = {
            "learning_rate_regression_head" : trial.suggest_float("learning_rate_regression_head", 5e-5, 2e-3, log=True),
            "learning_rate_pool"            : trial.suggest_float("learning_rate_pool", 1e-5, 1e-3, log=True),
            "learning_rate_encoder"         : trial.suggest_float("learning_rate_encoder", 1e-5, 1e-3, log=True),
            "weight_decay_regression_head"  : trial.suggest_float("weight_decay_regression_head", 1e-6, 1e-4, log=True),
            "weight_decay_pool"             : trial.suggest_float("weight_decay_pool", 1e-6, 1e-4, log=True),
            "weight_decay_encoder"          : trial.suggest_float("weight_decay_encoder", 1e-6, 5e-4, log=True),
        }

        return model_overrides, optimizer_overrides

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
        optimizer_keys      = {"learning_rate_regression_head", "learning_rate_pool", "learning_rate_encoder", "weight_decay_regression_head", "weight_decay_pool", "weight_decay_encoder"}
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
