from __future__ import annotations

import lightgbm
import numpy as np
import xgboost

from configuration.baseline.general import BaselineImportance, BaselineLabels, LightGbmSearchSpace, XgBoostSearchSpace
from configuration.data.general     import CoordinateAxes
from configuration.shared           import IoNames


class BoostedAxisEnsemble:
    def __init__(self, booster_config, seed):
        self.booster = booster_config
        self.seed    = int(seed)
        self.models  = []

    def fit(self, features, targets, validation_features, validation_targets, params):
        self.models = []

        features            = np.ascontiguousarray(features)
        validation_features = np.ascontiguousarray(validation_features)

        for axis_index in range(targets.shape[1]):
            target            = np.ascontiguousarray(targets[:, axis_index])
            validation_target = np.ascontiguousarray(validation_targets[:, axis_index])

            self.models.append(self._fit_axis(features, target, validation_features, validation_target, params, axis_index))
        return self

    def predict(self, features):
        return np.column_stack([model.predict(features) for model in self.models]).astype(np.float32)

    def best_iterations(self):
        return [int(self._best_iteration(model)) for model in self.models]

    def importances(self, feature_names):
        gains = np.mean([np.asarray(model.feature_importances_, dtype=np.float64) for model in self.models], axis=0)
        share = gains / (gains.sum() + BaselineImportance.gain_epsilon)
        return {name: float(value) for name, value in zip(feature_names, share)}


class LightGbmBaseline(BoostedAxisEnsemble):
    KEY   = BaselineLabels.lightgbm_key
    LABEL = BaselineLabels.lightgbm_label

    @staticmethod
    def search_space(trial):
        return {
            "learning_rate"     : trial.suggest_float("learning_rate", *LightGbmSearchSpace.learning_rate, log=True),
            "num_leaves"        : trial.suggest_int("num_leaves", *LightGbmSearchSpace.num_leaves, log=True),
            "min_child_samples" : trial.suggest_int("min_child_samples", *LightGbmSearchSpace.min_child_samples, log=True),
            "subsample"         : trial.suggest_float("subsample", *LightGbmSearchSpace.subsample),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", *LightGbmSearchSpace.colsample_bytree),
            "reg_alpha"         : trial.suggest_float("reg_alpha", *LightGbmSearchSpace.reg_alpha, log=True),
            "reg_lambda"        : trial.suggest_float("reg_lambda", *LightGbmSearchSpace.reg_lambda, log=True),
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
            eval_X      = validation_features,
            eval_y      = validation_target,
            eval_metric = "l2",
            callbacks   = [lightgbm.early_stopping(self.booster.early_stopping_rounds, verbose=False), lightgbm.log_evaluation(period=0)],
        )
        return model

    def _best_iteration(self, model):
        return model.best_iteration_

    def save(self, directory):
        saved = []
        for axis, model in zip(CoordinateAxes.names, self.models):
            path = directory / IoNames.lightgbm_model_file.format(axis=axis)
            model.booster_.save_model(str(path))
            saved.append(path)
        return saved


class XgBoostBaseline(BoostedAxisEnsemble):
    KEY   = BaselineLabels.xgboost_key
    LABEL = BaselineLabels.xgboost_label

    @staticmethod
    def search_space(trial):
        return {
            "learning_rate"    : trial.suggest_float("learning_rate", *XgBoostSearchSpace.learning_rate, log=True),
            "max_depth"        : trial.suggest_int("max_depth", *XgBoostSearchSpace.max_depth),
            "min_child_weight" : trial.suggest_float("min_child_weight", *XgBoostSearchSpace.min_child_weight, log=True),
            "subsample"        : trial.suggest_float("subsample", *XgBoostSearchSpace.subsample),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", *XgBoostSearchSpace.colsample_bytree),
            "reg_alpha"        : trial.suggest_float("reg_alpha", *XgBoostSearchSpace.reg_alpha, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", *XgBoostSearchSpace.reg_lambda, log=True),
            "gamma"            : trial.suggest_float("gamma", *XgBoostSearchSpace.gamma, log=True),
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
        for axis, model in zip(CoordinateAxes.names, self.models):
            path = directory / IoNames.xgboost_model_file.format(axis=axis)
            model.save_model(str(path))
            saved.append(path)
        return saved


class BaselineFamilies:
    @staticmethod
    def by_key():
        return {family.KEY: family for family in (LightGbmBaseline, XgBoostBaseline)}
