from __future__ import annotations

import json
from collections import Counter
from pathlib     import Path

import numpy as np
import torch
from scipy.stats import skew

from configuration.data.general import NormalizationSettings
from configuration.shared       import IoNames


class ChannelStrategySelector:
    def select(self, values):
        finite_values = values[np.isfinite(values)].astype(np.float64)

        standard_deviation = float(finite_values.std())
        if standard_deviation < NormalizationSettings.epsilon:
            return "zscore", float(finite_values.mean()), 1.0, False

        minimum         = float(finite_values.min())
        is_non_negative = minimum >= -NormalizationSettings.epsilon
        skewness        = float(skew(finite_values)) if finite_values.size > 2 else 0.0

        if is_non_negative and skewness > NormalizationSettings.skewness_log:
            logged = np.log1p(np.maximum(finite_values, 0.0))
            return "log1p_zscore", float(logged.mean()), float(logged.std()) + NormalizationSettings.epsilon, True

        quantile_75, quantile_25 = np.percentile(finite_values, list(NormalizationSettings.quartiles))
        interquartile_range      = float(quantile_75 - quantile_25)
        robust_spread            = interquartile_range / NormalizationSettings.robust_iqr_factor

        if robust_spread > NormalizationSettings.epsilon and (standard_deviation / robust_spread) > NormalizationSettings.outlier_ratio:
            return "robust_iqr", float(np.median(finite_values)), interquartile_range + NormalizationSettings.epsilon, False

        return "zscore", float(finite_values.mean()), standard_deviation + NormalizationSettings.epsilon, False


class FeatureGroupNormalizer:
    def __init__(self, methods, center, scale, log_mask):
        self.methods        = list(methods)
        self.center         = np.asarray(center,   dtype=np.float32)
        self.scale          = np.asarray(scale,    dtype=np.float32)
        self.log_mask       = np.asarray(log_mask, dtype=bool)
        self.device_tensors = {}

    @classmethod
    def fit(cls, matrix):
        selector           = ChannelStrategySelector()
        matrix             = np.asarray(matrix)
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

    @classmethod
    def fit_empty(cls):
        return cls([], np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool))

    @classmethod
    def fit_axiswise(cls, matrix):
        matrix             = np.asarray(matrix, dtype=np.float64)
        number_of_channels = matrix.shape[1]

        centers = matrix.mean(axis=0)
        scales  = matrix.std(axis=0) + NormalizationSettings.epsilon

        methods   = ["axiswise"] * number_of_channels
        log_masks = np.zeros(number_of_channels, dtype=bool)

        return cls(methods, centers.astype(np.float32), scales.astype(np.float32), log_masks)

    @classmethod
    def fit_isotropic(cls, matrix):
        matrix             = np.asarray(matrix, dtype=np.float64)
        number_of_channels = matrix.shape[1]

        centers      = matrix.mean(axis=0)
        shared_scale = float(np.sqrt(((matrix - centers) ** 2).mean())) + NormalizationSettings.epsilon

        methods   = ["isotropic"] * number_of_channels
        scales    = np.full(number_of_channels, shared_scale, dtype=np.float32)
        log_masks = np.zeros(number_of_channels, dtype=bool)

        return cls(methods, centers.astype(np.float32), scales, log_masks)

    def align_channels(self, channels, reference):
        width = len(reference.methods)
        for position, index in enumerate(channels):
            axis                 = position % width
            self.methods[index]  = reference.methods[axis]
            self.center[index]   = reference.center[axis]
            self.scale[index]    = reference.scale[axis]
            self.log_mask[index] = reference.log_mask[axis]

        self.device_tensors = {}
        return self

    def forward_numpy(self, matrix):
        matrix      = np.asarray(matrix, dtype=np.float32)
        transformed = np.where(self.log_mask, np.log1p(np.maximum(matrix, 0.0)), matrix)
        return (transformed - self.center) / self.scale

    def inverse_torch(self, values, device):
        center, scale, log_mask = self._tensors_for(device)

        unscaled = values * scale + center
        return torch.where(log_mask, torch.expm1(unscaled), unscaled)

    def _tensors_for(self, device):
        key = str(device)
        if key not in self.device_tensors:
            center   = torch.tensor(self.center,   dtype=torch.float32, device=device)
            scale    = torch.tensor(self.scale,    dtype=torch.float32, device=device)
            log_mask = torch.tensor(self.log_mask, dtype=torch.bool,    device=device)

            self.device_tensors[key] = (center, scale, log_mask)
        return self.device_tensors[key]

    def strategy_counts(self):
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
    def __init__(self, node, edge, graph, target):
        self.node   = node
        self.edge   = edge
        self.graph  = graph
        self.target = target

    def as_dict(self):
        return {group: getattr(self, group).to_dict() for group in NormalizationSettings.groups}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / IoNames.normalization_stats_file
        path.write_text(json.dumps(self.as_dict(), indent=IoNames.json_indent), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory):
        path    = Path(directory) / IoNames.normalization_stats_file
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{group: FeatureGroupNormalizer.from_dict(payload[group]) for group in NormalizationSettings.groups})
