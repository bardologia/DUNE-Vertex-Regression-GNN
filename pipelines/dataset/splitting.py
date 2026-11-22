from __future__ import annotations

import numpy as np

from configuration.data.general import CoordinateAxes


class LatticePlanes:
    @staticmethod
    def held(targets, stride, offset):
        columns = [np.searchsorted(np.unique(targets[:, axis]), targets[:, axis]) % stride == offset for axis in range(targets.shape[1])]

        return np.stack(columns, axis=1)

    @staticmethod
    def scored(targets, stride, offset):
        held = LatticePlanes.held(targets, stride, offset)

        return np.where(held.any(axis=1, keepdims=True), held, True)


class PlaneSplitter:
    def __init__(self, logger, stride, offset, val_share, random_state):
        self.logger       = logger
        self.stride       = stride
        self.offset       = offset
        self.val_share    = val_share
        self.random_state = random_state

    def _describe(self, targets, held, parts):
        planes = {f"Held {name} planes": ", ".join(f"{value:g}" for value in np.unique(targets[held[:, axis], axis])) for axis, name in enumerate(CoordinateAxes.names)}
        counts = {name: f"{len(indices)} ({len(indices) / len(targets):.1%})" for name, indices in zip(("train", "val", "test"), parts)}

        self.logger.kv_table({"Plane stride / offset": f"{self.stride} / {self.offset}", **planes, **counts}, title="Lattice-plane split")

    def split(self, targets):
        held     = LatticePlanes.held(targets, self.stride, self.offset)
        pooled   = np.flatnonzero(held.any(axis=1))
        shuffled = np.random.default_rng(self.random_state).permutation(pooled)
        cut      = int(round(len(shuffled) * self.val_share))
        parts    = (np.flatnonzero(~held.any(axis=1)), np.sort(shuffled[:cut]), np.sort(shuffled[cut:]))

        self._describe(targets, held, parts)

        return parts
