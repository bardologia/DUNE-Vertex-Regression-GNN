from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit


class StratificationLabeller:
    def __init__(self, maximum_bins_cap=10):
        self.maximum_bins_cap = maximum_bins_cap

    def maximum_bins(self, number_of_targets):
        return max(1, min(self.maximum_bins_cap, number_of_targets // 10))

    def build_labels(self, targets, number_of_bins):
        if number_of_bins <= 1:
            return np.zeros(len(targets), dtype=np.int64)

        binned_coordinates = []
        for dimension in range(targets.shape[1]):
            percentiles = np.linspace(0, 100, number_of_bins + 1)
            bins        = np.unique(np.percentile(targets[:, dimension], percentiles))

            if len(bins) <= 1:
                binned = np.zeros(len(targets), dtype=np.int64)
            else:
                binned = np.digitize(targets[:, dimension], bins[:-1]) - 1
                binned = np.clip(binned, 0, len(bins) - 2)

            binned_coordinates.append(binned)

        binned_coordinates = np.array(binned_coordinates).T
        actual_bins        = max(len(np.unique(column)) for column in binned_coordinates.T)
        labels             = binned_coordinates[:, 0] * actual_bins ** 2 + binned_coordinates[:, 1] * actual_bins + binned_coordinates[:, 2]
        return labels.astype(np.int64)


class TargetBalancer:
    def __init__(self, logger, coordinate_columns, split_config):
        self.logger             = logger
        self.coordinate_columns = list(coordinate_columns)
        self.split_config       = split_config
        self.labeller           = StratificationLabeller()

    @staticmethod
    def _resolve_split_size(total_size, split_size):
        if isinstance(split_size, float):
            return int(np.ceil(total_size * split_size))
        return int(split_size)

    def _find_valid_stratification(self, targets):
        number_of_targets = len(targets)
        maximum_bins      = self.labeller.maximum_bins(number_of_targets)
        test_size_count   = self._resolve_split_size(number_of_targets, self.split_config.test_size)
        val_size_count    = self._resolve_split_size(number_of_targets, self.split_config.val_ratio)

        for number_of_bins in range(maximum_bins, 0, -1):
            stratification_labels       = self.labeller.build_labels(targets, number_of_bins)
            unique_labels, label_counts = np.unique(stratification_labels, return_counts=True)

            if len(unique_labels) > test_size_count or len(unique_labels) > val_size_count:
                continue

            if label_counts.min() < 2:
                continue

            test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.split_config.test_size, random_state=self.split_config.random_state)
            try:
                train_validation_indices, test_indices = next(test_splitter.split(np.arange(number_of_targets), stratification_labels))
            except ValueError:
                continue

            validation_ratio_adjusted = self.split_config.val_ratio / (1 - self.split_config.test_size)
            validation_splitter       = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio_adjusted, random_state=self.split_config.random_state)
            try:
                train_indices, validation_indices = next(validation_splitter.split(train_validation_indices, stratification_labels[train_validation_indices]))
            except ValueError:
                continue

            return number_of_bins, stratification_labels, train_validation_indices, test_indices, train_indices, validation_indices

        raise ValueError("Unable to construct a valid stratified split for the current dataset and split configuration.")

    def _log_distribution(self, title, targets, named_indices):
        rows = []
        for split_name, indices in named_indices:
            for dimension, column in enumerate(self.coordinate_columns):
                rows.append({
                    "Split"  : split_name,
                    "Coord"  : column,
                    "Mean"   : float(targets[indices, dimension].mean()),
                    "Std"    : float(targets[indices, dimension].std()),
                    "Min"    : float(targets[indices, dimension].min()),
                    "Max"    : float(targets[indices, dimension].max()),
                })
        self.logger.metrics_table(rows, ["Split", "Coord", "Mean", "Std", "Min", "Max"], title=title)

    def balance(self, target_dataframe):
        self.logger.section("[Balancing Targets]")
        targets = target_dataframe[self.coordinate_columns].to_numpy(dtype=np.float32)

        number_of_bins, stratification_labels, train_validation_indices, test_indices, train_indices, validation_indices = self._find_valid_stratification(targets)

        train_indices      = train_validation_indices[train_indices]
        validation_indices = train_validation_indices[validation_indices]

        total_events = len(train_indices) + len(validation_indices) + len(test_indices)
        self.logger.kv_table({
            "Bins per dimension"    : number_of_bins,
            "Stratification groups" : len(np.unique(stratification_labels)),
            "Train"                 : f"{len(train_indices)} ({len(train_indices) / total_events:.1%})",
            "Validation"            : f"{len(validation_indices)} ({len(validation_indices) / total_events:.1%})",
            "Test"                  : f"{len(test_indices)} ({len(test_indices) / total_events:.1%})",
        })

        self._log_distribution(
            "Target Distribution After Balancing",
            targets,
            [("train", train_indices), ("val", validation_indices), ("test", test_indices)],
        )

        return train_indices, validation_indices, test_indices, targets


class CrossValidationSplitter:
    def __init__(self, logger, coordinate_columns, cross_validation_config):
        self.logger             = logger
        self.coordinate_columns = list(coordinate_columns)
        self.config             = cross_validation_config
        self.labeller           = StratificationLabeller()

    def _build_labels(self, targets):
        if not self.config.stratified:
            return np.zeros(len(targets), dtype=np.int64), 1

        maximum_bins = self.labeller.maximum_bins(len(targets))
        for number_of_bins in range(maximum_bins, 0, -1):
            labels       = self.labeller.build_labels(targets, number_of_bins)
            label_counts = np.unique(labels, return_counts=True)[1]

            if label_counts.min() >= self.config.n_folds:
                return labels, number_of_bins

        return np.zeros(len(targets), dtype=np.int64), 1

    def _outer_splitter(self, labels):
        is_stratified = labels.max() > 0
        shuffle       = self.config.shuffle
        random_state  = self.config.random_state if shuffle else None

        if is_stratified:
            return StratifiedKFold(n_splits=self.config.n_folds, shuffle=shuffle, random_state=random_state), True

        return KFold(n_splits=self.config.n_folds, shuffle=shuffle, random_state=random_state), False

    def _carve_validation(self, train_validation_indices, labels, fold_index):
        subset_labels        = labels[train_validation_indices]
        unique, label_counts = np.unique(subset_labels, return_counts=True)
        validation_count     = int(np.ceil(len(train_validation_indices) * self.config.validation_fraction))

        stratifiable = subset_labels.max() > 0 and label_counts.min() >= 2 and len(unique) <= validation_count
        random_state = self.config.random_state + fold_index

        if stratifiable:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.config.validation_fraction, random_state=random_state)
            try:
                relative_train, relative_validation = next(splitter.split(train_validation_indices, subset_labels))
                return train_validation_indices[relative_train], train_validation_indices[relative_validation]
            except ValueError:
                self.logger.subsection(f"Fold {fold_index}: stratified validation carve-out infeasible, using a random carve-out")

        generator           = np.random.default_rng(random_state)
        permuted            = generator.permutation(len(train_validation_indices))
        validation_relative = permuted[:max(1, validation_count)]
        train_relative      = permuted[max(1, validation_count):]
        return train_validation_indices[train_relative], train_validation_indices[validation_relative]

    def _log_fold_sizes(self, fold_index, train_indices, validation_indices, test_indices):
        self.logger.subsection(f"Fold {fold_index}: train={len(train_indices)} | val={len(validation_indices)} | test={len(test_indices)}")

    def split(self, target_dataframe):
        self.logger.section("[Cross-Validation Folds]")
        targets = target_dataframe[self.coordinate_columns].to_numpy(dtype=np.float32)

        labels, number_of_bins = self._build_labels(targets)
        splitter, is_stratified = self._outer_splitter(labels)

        self.logger.kv_table({
            "Folds"      : self.config.n_folds,
            "Stratified" : is_stratified,
            "Bins"       : number_of_bins,
            "Groups"     : len(np.unique(labels)),
        })

        folds = []
        for fold_index, (train_validation_indices, test_indices) in enumerate(splitter.split(np.arange(len(targets)), labels)):
            train_indices, validation_indices = self._carve_validation(train_validation_indices, labels, fold_index)

            self._log_fold_sizes(fold_index, train_indices, validation_indices, test_indices)
            folds.append((train_indices, validation_indices, test_indices))

        return folds
