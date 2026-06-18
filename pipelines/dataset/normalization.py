from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import skew


class ChannelStrategySelector:
    EPSILON           = 1e-8
    SKEWNESS_LOG      = 2.0
    OUTLIER_RATIO     = 3.0
    ROBUST_IQR_FACTOR = 1.349

    def select(self, values):
        finite_values = values[np.isfinite(values)].astype(np.float64)

        standard_deviation = float(finite_values.std())
        if standard_deviation < self.EPSILON:
            return "zscore", float(finite_values.mean()), 1.0, False

        minimum         = float(finite_values.min())
        is_non_negative = minimum >= -self.EPSILON
        skewness        = float(skew(finite_values)) if finite_values.size > 2 else 0.0

        if is_non_negative and skewness > self.SKEWNESS_LOG:
            logged = np.log1p(np.maximum(finite_values, 0.0))
            return "log1p_zscore", float(logged.mean()), float(logged.std()) + self.EPSILON, True

        quantile_75, quantile_25 = np.percentile(finite_values, [75, 25])
        interquartile_range      = float(quantile_75 - quantile_25)
        robust_spread            = interquartile_range / self.ROBUST_IQR_FACTOR

        if robust_spread > self.EPSILON and (standard_deviation / robust_spread) > self.OUTLIER_RATIO:
            return "robust_iqr", float(np.median(finite_values)), interquartile_range + self.EPSILON, False

        return "zscore", float(finite_values.mean()), standard_deviation + self.EPSILON, False


class FeatureGroupNormalizer:
    def __init__(self, methods, center, scale, log_mask):
        self.methods  = list(methods)
        self.center   = np.asarray(center,   dtype=np.float32)
        self.scale    = np.asarray(scale,    dtype=np.float32)
        self.log_mask = np.asarray(log_mask, dtype=bool)

    @classmethod
    def fit(cls, matrix):
        selector       = ChannelStrategySelector()
        matrix         = np.asarray(matrix, dtype=np.float64)
        number_of_channels = matrix.shape[1]

        methods   = []
        centers   = np.zeros(number_of_channels, dtype=np.float32)
        scales    = np.zeros(number_of_channels, dtype=np.float32)
        log_masks = np.zeros(number_of_channels, dtype=bool)

        for channel in range(number_of_channels):
            method, center, scale, log_flag = selector.select(matrix[:, channel])
            methods.append(method)
            centers[channel]   = center
            scales[channel]    = scale
            log_masks[channel] = log_flag

        return cls(methods, centers, scales, log_masks)

    def forward_numpy(self, matrix):
        matrix      = np.asarray(matrix, dtype=np.float32)
        transformed = np.where(self.log_mask, np.log1p(np.maximum(matrix, 0.0)), matrix)
        return (transformed - self.center) / self.scale

    def inverse_torch(self, values, device):
        center   = torch.tensor(self.center,   dtype=torch.float32, device=device)
        scale    = torch.tensor(self.scale,    dtype=torch.float32, device=device)
        log_mask = torch.tensor(self.log_mask, dtype=torch.bool,    device=device)

        unscaled = values * scale + center
        return torch.where(log_mask, torch.expm1(unscaled), unscaled)

    def strategy_counts(self):
        from collections import Counter
        return dict(Counter(self.methods))

    def to_dict(self):
        return {
            "methods"  : self.methods,
            "center"   : self.center.tolist(),
            "scale"    : self.scale.tolist(),
            "log_mask" : self.log_mask.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["methods"], payload["center"], payload["scale"], payload["log_mask"])


class NormalizationStats:
    GROUPS = ("node", "edge", "target")

    def __init__(self, node, edge, target):
        self.node   = node
        self.edge   = edge
        self.target = target

    def as_dict(self):
        return {group: getattr(self, group).to_dict() for group in self.GROUPS}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path    = directory / "normalization_stats.json"
        payload = {group: getattr(self, group).to_dict() for group in self.GROUPS}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory):
        path    = Path(directory) / "normalization_stats.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{group: FeatureGroupNormalizer.from_dict(payload[group]) for group in cls.GROUPS})
