from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class TargetBalancer:
    def __init__(self, logger, coordinate_columns, split_config):
        self.logger             = logger
        self.coordinate_columns = list(coordinate_columns)
        self.split_config       = split_config

    @staticmethod
    def _resolve_split_size(total_size, split_size):
        if isinstance(split_size, float):
            return int(np.ceil(total_size * split_size))
        return int(split_size)

    def _build_stratification_labels(self, targets, number_of_bins):
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

    def _find_valid_stratification(self, targets):
        number_of_targets = len(targets)
        maximum_bins      = max(1, min(10, number_of_targets // 10))
        test_size_count   = self._resolve_split_size(number_of_targets, self.split_config.test_size)
        val_size_count    = self._resolve_split_size(number_of_targets, self.split_config.val_ratio)

        for number_of_bins in range(maximum_bins, 0, -1):
            stratification_labels       = self._build_stratification_labels(targets, number_of_bins)
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

        self.logger.subsection(f"Binning into {number_of_bins} bins per dimension | {len(np.unique(stratification_labels))} stratification groups")
        self.logger.subsection(f"Split sizes: train={len(train_indices)} | val={len(validation_indices)} | test={len(test_indices)}")

        self._log_distribution(
            "Target Distribution After Balancing",
            targets,
            [("train", train_indices), ("val", validation_indices), ("test", test_indices)],
        )

        return train_indices, validation_indices, test_indices, targets
