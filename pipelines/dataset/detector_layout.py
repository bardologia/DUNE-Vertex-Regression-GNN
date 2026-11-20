from __future__ import annotations

import numpy as np
import pandas as pd


class CoordinateTransform:
    def apply_to_target(self, target_x: float, target_y: float, target_z: float) -> tuple[float, float, float]:
        sensor_x = target_y
        sensor_y = -target_x
        sensor_z = target_z
        return sensor_x, sensor_y, sensor_z

    def apply_to_positions(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        return np.column_stack([positions[:, 1], -positions[:, 0], positions[:, 2]])


class ApexModuleLayout:
    BATCH_SIZE       = 48
    HALF_SLOTS       = 12
    DRIFT_PITCH      = 0.505
    DRIFT_EDGE       = 6.0275
    WALL_SIDE        = 6.673
    WALL_FIRST_Z     = -29.75
    WALL_RING_PITCH  = 0.5
    WALL_MODULES     = 5760
    ENDCAP_FACE      = 29.9265
    ENDCAP_FIRST_ROW = -5.8075
    ENDCAP_ROW_PITCH = 0.505
    ENDCAP_MODULES   = 1152
    WALL_FACES       = ("right_wall", "left_wall")
    ENDCAP_FACES     = ("front_endcap", "back_endcap")
    BLOCKS           = ("wall", "endcap")

    def __init__(self) -> None:
        self.transform = CoordinateTransform()

    def wall_block(self) -> pd.DataFrame:
        modules = np.arange(self.WALL_MODULES, dtype=np.int64)
        rings   = modules // self.BATCH_SIZE
        slots   = modules % self.BATCH_SIZE
        second  = slots % 2

        simulation = np.column_stack([
            np.where(second == 0, self.WALL_SIDE, -self.WALL_SIDE),
            self.drift(slots // 2),
            self.WALL_FIRST_Z + self.WALL_RING_PITCH * rings,
        ])
        return self._block_frame(self.BLOCKS[0], modules, np.asarray(self.WALL_FACES, dtype=object)[second], simulation)

    def endcap_block(self) -> pd.DataFrame:
        modules = np.arange(self.ENDCAP_MODULES, dtype=np.int64)
        rows    = modules // self.BATCH_SIZE
        slots   = modules % self.BATCH_SIZE
        second  = slots % 2

        simulation = np.column_stack([
            self.ENDCAP_FIRST_ROW + self.ENDCAP_ROW_PITCH * rows,
            self.drift(slots // 2),
            np.where(second == 0, self.ENDCAP_FACE, -self.ENDCAP_FACE),
        ])
        return self._block_frame(self.BLOCKS[1], modules, np.asarray(self.ENDCAP_FACES, dtype=object)[second], simulation)

    def drift(self, pair_slots: np.ndarray) -> np.ndarray:
        pair_slots = np.asarray(pair_slots, dtype=np.int64)
        lower      = -self.DRIFT_EDGE + self.DRIFT_PITCH * pair_slots
        upper      = self.DRIFT_EDGE - self.DRIFT_PITCH * (pair_slots - self.HALF_SLOTS)
        return np.where(pair_slots < self.HALF_SLOTS, lower, upper)

    def _block_frame(self, block: str, modules: np.ndarray, faces: np.ndarray, simulation: np.ndarray) -> pd.DataFrame:
        sensor = self.transform.apply_to_positions(simulation)
        return pd.DataFrame({
            "block"        : np.full(modules.shape[0], block, dtype=object),
            "module_index" : modules,
            "face"         : faces,
            "x"            : sensor[:, 0],
            "y"            : sensor[:, 1],
            "z"            : sensor[:, 2],
        })

    def module_table(self) -> pd.DataFrame:
        return pd.concat([self.wall_block(), self.endcap_block()], ignore_index=True)


class RawBinMapping:
    LEAD_BIN             = 12
    LEAD_MODULE          = 1
    WALL_BINS            = (15, 5771)
    WALL_OFFSET          = 12
    ENDCAP_BINS          = (5774, 6924)
    ENDCAP_OFFSET        = 5773
    FACE_TOTAL_BINS      = (13, 5772, 6925)
    ABSENT_BINS          = (14, 5773)
    EMPTY_WALL_MODULES   = (0, 2)
    EMPTY_ENDCAP_MODULES = (0,)
    FACE_TOTAL_BLOCK     = "face_total"

    def raw_bins(self) -> np.ndarray:
        first = self.LEAD_BIN
        last  = self.FACE_TOTAL_BINS[-1]
        every = np.arange(first, last + 1, dtype=np.int64)
        return every[~np.isin(every, self.ABSENT_BINS)]

    def assign(self, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bins    = np.asarray(bins, dtype=np.int64)
        blocks  = np.full(bins.shape[0], "", dtype=object)
        modules = np.full(bins.shape[0], -1, dtype=np.int64)

        on_wall   = (bins >= self.WALL_BINS[0]) & (bins <= self.WALL_BINS[1])
        on_endcap = (bins >= self.ENDCAP_BINS[0]) & (bins <= self.ENDCAP_BINS[1])
        lead      = bins == self.LEAD_BIN
        total     = np.isin(bins, self.FACE_TOTAL_BINS)

        blocks[on_wall | lead] = ApexModuleLayout.BLOCKS[0]
        blocks[on_endcap]      = ApexModuleLayout.BLOCKS[1]
        blocks[total]          = self.FACE_TOTAL_BLOCK
        modules[on_wall]       = bins[on_wall] - self.WALL_OFFSET
        modules[on_endcap]     = bins[on_endcap] - self.ENDCAP_OFFSET
        modules[lead]          = self.LEAD_MODULE

        unknown = blocks == ""
        if unknown.any():
            raise ValueError(f"Raw bins {bins[unknown][:10].tolist()} belong to no module block of the documented mapping.")

        return blocks, modules


class ChannelMapping:
    RAW_COLUMNS        = ("bin", "x", "y", "z")
    GEOMETRY_COLUMNS   = ("channel_index", "raw_channel_index", "bin", "module_index", "face", "x", "y", "z")
    CHANNEL_COUNT      = 6909
    POSITION_DECIMALS  = 6
    POSITION_TOLERANCE = 1e-6

    def __init__(self, logger) -> None:
        self.logger      = logger
        self.layout      = ApexModuleLayout()
        self.bin_mapping = RawBinMapping()

    def _raw_bins(self, raw_frame: pd.DataFrame) -> np.ndarray:
        values = raw_frame["bin"].to_numpy(dtype=np.float64)
        if not np.all(np.mod(values, 1.0) == 0.0):
            raise ValueError("The raw bin column holds non-integer values; the mapping reads integer histogram bins.")

        bins     = values.astype(np.int64)
        expected = self.bin_mapping.raw_bins()
        if bins.shape != expected.shape or not np.array_equal(np.sort(bins), expected):
            missing = np.setdiff1d(expected, bins)
            extra   = np.setdiff1d(bins, expected)
            raise ValueError(f"The raw bins are not exactly 12, 13, 15..5772 and 5774..6925: {bins.shape[0]} bins, missing {missing[:10].tolist()}, unexpected {extra[:10].tolist()}.")

        return bins

    def _check_position_sets(self, raw_frame: pd.DataFrame, modules: pd.DataFrame) -> None:
        raw        = self._sorted_positions(raw_frame[["x", "y", "z"]].to_numpy(dtype=np.float64))
        documented = self._sorted_positions(modules[["x", "y", "z"]].to_numpy(dtype=np.float64))

        if raw.shape != documented.shape:
            raise ValueError(f"The raw files hold {raw.shape[0]} positions and the document {documented.shape[0]} modules; the layouts must match one to one.")

        deviation = float(np.abs(raw - documented).max())
        if deviation > self.POSITION_TOLERANCE:
            raise ValueError(f"The set of raw positions differs from the set of documented module positions by up to {deviation:.3e} m; the raw files do not hold the documented layout.")

    def _sorted_positions(self, positions: np.ndarray) -> np.ndarray:
        rounded = np.round(positions, self.POSITION_DECIMALS)
        return positions[np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))]

    def _map_channels(self, bins: np.ndarray, modules: pd.DataFrame) -> pd.DataFrame:
        blocks, module_indices = self.bin_mapping.assign(bins)
        kept_rows              = np.flatnonzero(blocks != self.bin_mapping.FACE_TOTAL_BLOCK)

        lookup  = modules.set_index(["block", "module_index"])
        located = lookup.loc[list(zip(blocks[kept_rows], module_indices[kept_rows]))]

        return pd.DataFrame({
            "channel_index"     : np.arange(kept_rows.shape[0], dtype=np.int32),
            "raw_channel_index" : kept_rows.astype(np.int32),
            "bin"               : bins[kept_rows].astype(np.int32),
            "module_index"      : module_indices[kept_rows].astype(np.int32),
            "face"              : located["face"].to_numpy(dtype=object),
            "x"                 : located["x"].to_numpy(dtype=np.float64),
            "y"                 : located["y"].to_numpy(dtype=np.float64),
            "z"                 : located["z"].to_numpy(dtype=np.float64),
        })

    def _check_channels(self, geometry: pd.DataFrame, modules: pd.DataFrame) -> None:
        if len(geometry) != self.CHANNEL_COUNT:
            raise ValueError(f"The mapping keeps {len(geometry)} channels, expected {self.CHANNEL_COUNT}.")

        rounded  = np.round(geometry[["x", "y", "z"]].to_numpy(dtype=np.float64), self.POSITION_DECIMALS)
        distinct = np.unique(rounded, axis=0).shape[0]
        if distinct != self.CHANNEL_COUNT:
            raise ValueError(f"The {self.CHANNEL_COUNT} kept channels sit at only {distinct} distinct positions.")

        wall_block, endcap_block = ApexModuleLayout.BLOCKS
        on_wall                  = geometry["face"].isin(ApexModuleLayout.WALL_FACES).to_numpy()
        wall_modules             = modules.loc[modules["block"] == wall_block, "module_index"].to_numpy()
        endcap_modules           = modules.loc[modules["block"] == endcap_block, "module_index"].to_numpy()

        empty_wall   = np.setdiff1d(wall_modules, geometry.loc[on_wall, "module_index"].to_numpy()).tolist()
        empty_endcap = np.setdiff1d(endcap_modules, geometry.loc[~on_wall, "module_index"].to_numpy()).tolist()
        if empty_wall != list(self.bin_mapping.EMPTY_WALL_MODULES) or empty_endcap != list(self.bin_mapping.EMPTY_ENDCAP_MODULES):
            raise ValueError(f"The modules without a channel are wall {empty_wall} and endcap {empty_endcap}, expected wall {list(self.bin_mapping.EMPTY_WALL_MODULES)} and endcap {list(self.bin_mapping.EMPTY_ENDCAP_MODULES)}.")

    def build(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        modules = self.layout.module_table()
        bins    = self._raw_bins(raw_frame)

        self._check_position_sets(raw_frame, modules)
        geometry = self._map_channels(bins, modules)
        self._check_channels(geometry, modules)

        self.logger.subsection(f"Documented channel mapping: {len(raw_frame)} raw bins to {len(geometry)} channels; face totals {list(self.bin_mapping.FACE_TOTAL_BINS)} removed; empty modules wall {list(self.bin_mapping.EMPTY_WALL_MODULES)}, endcap {list(self.bin_mapping.EMPTY_ENDCAP_MODULES)}")
        return geometry


__all__ = [
    "CoordinateTransform",
    "ApexModuleLayout",
    "RawBinMapping",
    "ChannelMapping",
]
