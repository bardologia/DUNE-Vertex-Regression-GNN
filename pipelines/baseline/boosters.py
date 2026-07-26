from __future__ import annotations

import lightgbm
import numpy as np
import xgboost


class BoostedAxisEnsemble:
    AXES = ("x", "y", "z")

    def __init__(self, booster_config, seed):
        self.booster = booster_config
        self.seed    = int(seed)
        self.models  = []

    def best_iterations(self):
        return [int(self._best_iteration(model)) for model in self.models]

    def fit(self, features, targets, validation_features, validation_targets, params):
        self.models = []
        for axis_index in range(targets.shape[1]):
            model = self._fit_axis(features, targets[:, axis_index], validation_features, validation_targets[:, axis_index], params, axis_index)
            self.models.append(model)
        return self

    def predict(self, features):
        return np.column_stack([model.predict(features) for model in self.models]).astype(np.float32)

    def importances(self, feature_names):
        gains = np.mean([np.asarray(model.feature_importances_, dtype=np.float64) for model in self.models], axis=0)
        share = gains / (gains.sum() + 1e-12)
        return {name: float(value) for name, value in zip(feature_names, share)}


class LightGbmBaseline(BoostedAxisEnsemble):
    KEY   = "lightgbm"
    LABEL = "LightGBM"

    @staticmethod
    def search_space(trial):
        return {
            "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves"        : trial.suggest_int("num_leaves", 16, 512, log=True),
            "min_child_samples" : trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample"         : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def _fit_axis(self, features, target, validation_features, validation_target, params, axis_index):
        model = lightgbm.LGBMRegressor(
            n_estimators    = self.booster.max_estimators,
            subsample_freq  = 1,
            importance_type = "gain",
            random_state    = self.seed + axis_index,
            n_jobs          = self.booster.thread_count,
            verbose         = -1,
            **params,
        )
        model.fit(
            features, target,
            eval_set    = [(validation_features, validation_target)],
            eval_metric = "l2",
            callbacks   = [lightgbm.early_stopping(self.booster.early_stopping_rounds, verbose=False), lightgbm.log_evaluation(period=0)],
        )
        return model

    def _best_iteration(self, model):
        return model.best_iteration_

    def save(self, directory):
        saved = []
        for axis, model in zip(self.AXES, self.models):
            path = directory / f"lightgbm_{axis}.txt"
            model.booster_.save_model(str(path))
            saved.append(path)
        return saved


class XgBoostBaseline(BoostedAxisEnsemble):
    KEY   = "xgboost"
    LABEL = "XGBoost"

    @staticmethod
    def search_space(trial):
        return {
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 4, 12),
            "min_child_weight" : trial.suggest_float("min_child_weight", 0.5, 64.0, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma"            : trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        }

    def _fit_axis(self, features, target, validation_features, validation_target, params, axis_index):
        model = xgboost.XGBRegressor(
            n_estimators          = self.booster.max_estimators,
            early_stopping_rounds = self.booster.early_stopping_rounds,
            tree_method           = "hist",
            device                = self.booster.device,
            eval_metric           = "rmse",
            importance_type       = "gain",
            random_state          = self.seed + axis_index,
            n_jobs                = self.booster.thread_count,
            verbosity             = 0,
            **params,
        )
        model.fit(features, target, eval_set=[(validation_features, validation_target)], verbose=False)
        return model

    def _best_iteration(self, model):
        return model.best_iteration

    def save(self, directory):
        saved = []
        for axis, model in zip(self.AXES, self.models):
            path = directory / f"xgboost_{axis}.ubj"
            model.save_model(str(path))
            saved.append(path)
        return saved


BASELINE_MODEL_FAMILIES = {family.KEY: family for family in (LightGbmBaseline, XgBoostBaseline)}
