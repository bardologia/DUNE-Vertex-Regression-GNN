from __future__ import annotations

import numpy as np
import pandas as pd

from configuration.data.general import ApexModuleGeometry, ChannelBins, ChannelMappingChecks


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
    def __init__(self) -> None:
        self.transform = CoordinateTransform()

    def wall_block(self) -> pd.DataFrame:
        modules = np.arange(ApexModuleGeometry.wall_modules, dtype=np.int64)
        rings   = modules // ApexModuleGeometry.batch_size
        slots   = modules % ApexModuleGeometry.batch_size
        second  = slots % 2

        simulation = np.column_stack([
            np.where(second == 0, ApexModuleGeometry.wall_side, -ApexModuleGeometry.wall_side),
            self.drift(slots // 2),
            ApexModuleGeometry.wall_first_z + ApexModuleGeometry.wall_ring_pitch * rings,
        ])
        return self._block_frame(ApexModuleGeometry.blocks[0], modules, np.asarray(ApexModuleGeometry.wall_faces, dtype=object)[second], simulation)

    def endcap_block(self) -> pd.DataFrame:
        modules = np.arange(ApexModuleGeometry.endcap_modules, dtype=np.int64)
        rows    = modules // ApexModuleGeometry.batch_size
        slots   = modules % ApexModuleGeometry.batch_size
        second  = slots % 2

        simulation = np.column_stack([
            ApexModuleGeometry.endcap_first_row + ApexModuleGeometry.endcap_row_pitch * rows,
            self.drift(slots // 2),
            np.where(second == 0, ApexModuleGeometry.endcap_face, -ApexModuleGeometry.endcap_face),
        ])
        return self._block_frame(ApexModuleGeometry.blocks[1], modules, np.asarray(ApexModuleGeometry.endcap_faces, dtype=object)[second], simulation)

    def drift(self, pair_slots: np.ndarray) -> np.ndarray:
        pair_slots = np.asarray(pair_slots, dtype=np.int64)
        lower      = -ApexModuleGeometry.drift_edge + ApexModuleGeometry.drift_pitch * pair_slots
        upper      = ApexModuleGeometry.drift_edge - ApexModuleGeometry.drift_pitch * (pair_slots - ApexModuleGeometry.half_slots)
        return np.where(pair_slots < ApexModuleGeometry.half_slots, lower, upper)

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
    def raw_bins(self) -> np.ndarray:
        first = ChannelBins.lead_bin
        last  = ChannelBins.face_total_bins[-1]
        every = np.arange(first, last + 1, dtype=np.int64)
        return every[~np.isin(every, ChannelBins.absent_bins)]

    def assign(self, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bins    = np.asarray(bins, dtype=np.int64)
        blocks  = np.full(bins.shape[0], "", dtype=object)
        modules = np.full(bins.shape[0], -1, dtype=np.int64)

        on_wall   = (bins >= ChannelBins.wall_bins[0]) & (bins <= ChannelBins.wall_bins[1])
        on_endcap = (bins >= ChannelBins.endcap_bins[0]) & (bins <= ChannelBins.endcap_bins[1])
        lead      = bins == ChannelBins.lead_bin
        total     = np.isin(bins, ChannelBins.face_total_bins)

        blocks[on_wall | lead] = ApexModuleGeometry.blocks[0]
        blocks[on_endcap]      = ApexModuleGeometry.blocks[1]
        blocks[total]          = ChannelBins.face_total_block
        modules[on_wall]       = bins[on_wall] - ChannelBins.wall_offset
        modules[on_endcap]     = bins[on_endcap] - ChannelBins.endcap_offset
        modules[lead]          = ChannelBins.lead_module

        unknown = blocks == ""
        if unknown.any():
            raise ValueError(f"Raw bins {bins[unknown][:ChannelMappingChecks.listed_bins].tolist()} belong to no module block of the documented mapping.")

        return blocks, modules


class ChannelMapping:
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
            raise ValueError(f"The raw bins are not exactly 12, 13, 15..5772 and 5774..6925: {bins.shape[0]} bins, missing {missing[:ChannelMappingChecks.listed_bins].tolist()}, unexpected {extra[:ChannelMappingChecks.listed_bins].tolist()}.")

        return bins

    def _check_position_sets(self, raw_frame: pd.DataFrame, modules: pd.DataFrame) -> None:
        raw        = self._sorted_positions(raw_frame[["x", "y", "z"]].to_numpy(dtype=np.float64))
        documented = self._sorted_positions(modules[["x", "y", "z"]].to_numpy(dtype=np.float64))

        if raw.shape != documented.shape:
            raise ValueError(f"The raw files hold {raw.shape[0]} positions and the document {documented.shape[0]} modules; the layouts must match one to one.")

        deviation = float(np.abs(raw - documented).max())
        if deviation > ChannelMappingChecks.position_tolerance:
            raise ValueError(f"The set of raw positions differs from the set of documented module positions by up to {deviation:.3e} m; the raw files do not hold the documented layout.")

    def _sorted_positions(self, positions: np.ndarray) -> np.ndarray:
        rounded = np.round(positions, ChannelMappingChecks.position_decimals)
        return positions[np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))]

    def _map_channels(self, bins: np.ndarray, modules: pd.DataFrame) -> pd.DataFrame:
        blocks, module_indices = self.bin_mapping.assign(bins)
        kept_rows              = np.flatnonzero(blocks != ChannelBins.face_total_block)

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
        if len(geometry) != ChannelMappingChecks.channel_count:
            raise ValueError(f"The mapping keeps {len(geometry)} channels, expected {ChannelMappingChecks.channel_count}.")

        rounded  = np.round(geometry[["x", "y", "z"]].to_numpy(dtype=np.float64), ChannelMappingChecks.position_decimals)
        distinct = np.unique(rounded, axis=0).shape[0]
        if distinct != ChannelMappingChecks.channel_count:
            raise ValueError(f"The {ChannelMappingChecks.channel_count} kept channels sit at only {distinct} distinct positions.")

        wall_block, endcap_block = ApexModuleGeometry.blocks
        on_wall                  = geometry["face"].isin(ApexModuleGeometry.wall_faces).to_numpy()
        wall_modules             = modules.loc[modules["block"] == wall_block, "module_index"].to_numpy()
        endcap_modules           = modules.loc[modules["block"] == endcap_block, "module_index"].to_numpy()

        empty_wall   = np.setdiff1d(wall_modules, geometry.loc[on_wall, "module_index"].to_numpy()).tolist()
        empty_endcap = np.setdiff1d(endcap_modules, geometry.loc[~on_wall, "module_index"].to_numpy()).tolist()
        if empty_wall != list(ChannelBins.empty_wall_modules) or empty_endcap != list(ChannelBins.empty_endcap_modules):
            raise ValueError(f"The modules without a channel are wall {empty_wall} and endcap {empty_endcap}, expected wall {list(ChannelBins.empty_wall_modules)} and endcap {list(ChannelBins.empty_endcap_modules)}.")

    def build(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        modules = self.layout.module_table()
        bins    = self._raw_bins(raw_frame)

        self._check_position_sets(raw_frame, modules)
        geometry = self._map_channels(bins, modules)
        self._check_channels(geometry, modules)

        self.logger.subsection(f"Documented channel mapping: {len(raw_frame)} raw bins to {len(geometry)} channels; face totals {list(ChannelBins.face_total_bins)} removed; empty modules wall {list(ChannelBins.empty_wall_modules)}, endcap {list(ChannelBins.empty_endcap_modules)}")
        return geometry


__all__ = [
    "CoordinateTransform",
    "ApexModuleLayout",
    "RawBinMapping",
    "ChannelMapping",
]
