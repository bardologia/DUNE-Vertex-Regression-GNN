from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class NormalizationStats:
    KEYS = ("node_mean", "node_std", "edge_mean", "edge_std", "target_mean", "target_std")

    def __init__(self, node_mean, node_std, edge_mean, edge_std, target_mean, target_std):
        self.node_mean   = np.asarray(node_mean,   dtype=np.float32)
        self.node_std    = np.asarray(node_std,    dtype=np.float32)
        self.edge_mean   = np.asarray(edge_mean,   dtype=np.float32)
        self.edge_std    = np.asarray(edge_std,    dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std  = np.asarray(target_std,  dtype=np.float32)

    def as_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.KEYS}

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "normalization_stats.npz"
        np.savez(path, **self.as_dict())
        return path

    @classmethod
    def load(cls, directory):
        path    = Path(directory) / "normalization_stats.npz"
        archive = np.load(path)
        return cls(**{key: archive[key] for key in cls.KEYS})


class Normalizer:
    def __init__(self, logger, source_split: str = "train"):
        self.logger       = logger
        self.source_split = source_split
        self.stats        = None

    def _streaming_statistics(self, graphs, attribute):
        running_sum   = None
        running_count = 0

        for graph in graphs:
            values = getattr(graph, attribute)
            if values is None or values.numel() == 0:
                continue
            array = values.detach().cpu().numpy().astype(np.float64)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if running_sum is None:
                running_sum = np.zeros(array.shape[1], dtype=np.float64)
            running_sum   += array.sum(axis=0)
            running_count += array.shape[0]

        mean = running_sum / running_count

        running_square = np.zeros_like(running_sum)
        for graph in graphs:
            values = getattr(graph, attribute)
            if values is None or values.numel() == 0:
                continue
            array = values.detach().cpu().numpy().astype(np.float64)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            running_square += ((array - mean) ** 2).sum(axis=0)

        standard_deviation = np.sqrt(running_square / running_count).astype(np.float32) + 1e-8
        return mean.astype(np.float32), standard_deviation

    def _log_statistics(self, title, labels, mean, standard_deviation):
        rows = [{"Feature": label, "Mean": float(mean[index]), "Std": float(standard_deviation[index])} for index, label in enumerate(labels)]
        self.logger.metrics_table(rows, ["Feature", "Mean", "Std"], title=title)

    def compute_stats(self, splits) -> NormalizationStats:
        self.logger.section("[Normalization Statistics]")
        source_graphs = splits[self.source_split]

        node_mean,   node_std   = self._streaming_statistics(source_graphs, "x")
        edge_mean,   edge_std   = self._streaming_statistics(source_graphs, "edge_attr")
        target_mean, target_std = self._streaming_statistics(source_graphs, "y")

        self._log_statistics("Node Statistics",   [f"F{index}" for index in range(len(node_mean))], node_mean, node_std)
        self._log_statistics("Edge Statistics",   [f"F{index}" for index in range(len(edge_mean))], edge_mean, edge_std)
        self._log_statistics("Target Statistics", ["x", "y", "z"], target_mean, target_std)

        self.stats = NormalizationStats(node_mean, node_std, edge_mean, edge_std, target_mean, target_std)
        return self.stats

    def normalize(self, splits):
        self.logger.section("[Normalizing Splits]")

        node_mean   = torch.tensor(self.stats.node_mean,   dtype=torch.float32)
        node_std    = torch.tensor(self.stats.node_std,    dtype=torch.float32)
        edge_mean   = torch.tensor(self.stats.edge_mean,   dtype=torch.float32)
        edge_std    = torch.tensor(self.stats.edge_std,    dtype=torch.float32)
        target_mean = torch.tensor(self.stats.target_mean, dtype=torch.float32)
        target_std  = torch.tensor(self.stats.target_std,  dtype=torch.float32)

        for split_name, graphs in splits.items():
            for graph in graphs:
                graph.x         = (graph.x - node_mean) / node_std
                graph.edge_attr = (graph.edge_attr - edge_mean) / edge_std
                graph.y         = (graph.y - target_mean) / target_std

        return splits

    def run(self, splits):
        self.compute_stats(splits)
        splits = self.normalize(splits)
        return splits, self.stats
