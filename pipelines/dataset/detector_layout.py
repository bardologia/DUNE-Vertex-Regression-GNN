"""Documented module layout of the FD3 photon detectors and the mapping of the raw light bins onto it.

The simulation places 6,912 photon-detector modules in the order that the APEX
tile-positioning document describes (``notes/DUNE-FD3/tiles_positioning_and_indexes_apex.pdf``
in the vault, two pages): 5,760 modules on the two long lateral walls and 1,152
on the two endcaps, numbered in batches of 48. The document gives the module
coordinates in the simulation frame, with the long sides on X at +-6.673 m, the
drift on Y within +-6.0275 m and the beam on Z. The store works in the sensor
frame (x, y, z) = (Y, -X, Z), the transform that also maps the vertices of the
raw file names (``CoordinateTransform``), so the document's right side
(X = +6.673 m) is the wall at y = -6.673 m and its front endcap
(Z = +29.9265 m) the endcap at z = +29.9265 m.

The raw CSV files hold the light of each module in one histogram bin, the
``bin`` column. Their ``x``, ``y`` and ``z`` columns are the 2024 position
table, which is wrong in two compensating ways: it read the document's side
coordinate X as the sensor y without the sign change of the frame conversion,
and it skipped bin 14 without advancing its module counter, so it lags one
module behind the light bins from there on. On the walls the two errors cancel
for even raw rows and add up to a two-row shift for odd ones; on the endcaps
the lag swaps the two faces and the frame error mirrors the rows. The store
therefore builds every position from the bin through the document, never from
the raw x, y and z (channel-defect sweep of 2026-11-19, vault note
``notes/DUNE-FD3/odd-channel-position-defect-2026-11-19.md``; the full report,
sections 2, 3 and 8, sits in the FD3 scintillation study repository under
``logs/channel_defect_sweep/report.md``):

- bins 15 to 5771 hold wall module b - 12 and bins 5774 to 6924 hold endcap module b - 5773;
- bin 12 holds wall module 1, the far-wall module at (x, y, z) = (-6.0275, +6.673, -29.75) m. The document would put module 0 there, but the light of bin 12 behaves as a far-wall sensor and the light decides;
- bins 13, 5772 and 6925 lie outside both module blocks of the bin numbering and carry face totals, light that follows the summed light of whole faces at 100 to 1,000 times a module's level. They are not modules and are removed from the geometry and from the light;
- bins 14 and 5773 are absent from the raw files, so wall modules 0 and 2 and endcap module 0 have no channel and stay empty.

The stored geometry has 6,909 channels at 6,909 distinct positions, and on
every one of them it equals the position the sweep derived from the light
alone (its variant V1-drop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CoordinateTransform:
    """Maps points from the simulation frame into the sensor frame, (x, y, z) = (Y, -X, Z)."""

    def apply_to_target(self, target_x: float, target_y: float, target_z: float) -> tuple[float, float, float]:
        """Returns the vertex in the sensor frame, (y, -x, z) of the simulation frame.

        Args:
            target_x : Vertex x in the simulation frame, in metres.
            target_y : Vertex y in the simulation frame, in metres.
            target_z : Vertex z in the simulation frame, in metres.

        Returns:
            The vertex (x, y, z) in the sensor frame, in metres.
        """
        sensor_x = target_y
        sensor_y = -target_x
        sensor_z = target_z
        return sensor_x, sensor_y, sensor_z

    def apply_to_positions(self, positions: np.ndarray) -> np.ndarray:
        """Returns points of the simulation frame in the sensor frame.

        Args:
            positions : Points (X, Y, Z) in the simulation frame, shape [K, 3], in metres.

        Returns:
            The points (Y, -X, Z) as float64, shape [K, 3], in metres.
        """
        positions = np.asarray(positions, dtype=np.float64)
        return np.column_stack([positions[:, 1], -positions[:, 0], positions[:, 2]])


class ApexModuleLayout:
    """Positions of the 6,912 photon-detector modules in the order of the APEX tile-positioning document.

    Every constant below is a fact of the document. Modules are numbered from 0
    inside each of the two blocks, walls and endcaps, in batches of 48. Inside a
    batch the modules come in pairs that share a drift position: on the walls
    the right side (X = +6.673 m) before the left side (X = -6.673 m), on the
    endcaps the front face (Z = +29.9265 m) before the back face
    (Z = -29.9265 m). The 24 pair slots of a batch climb the lower drift volume
    from -6.0275 m in steps of 0.505 m and then descend the upper one from
    +6.0275 m, so pair slot j sits at the drift d(j) = -6.0275 + 0.505 j for
    j = 0..11 and d(j) = +6.0275 - 0.505 (j - 12) for j = 12..23. Wall batch r
    is the z ring at -29.75 + 0.5 r m, r = 0..119; endcap batch q is the row at
    X = -5.8075 + 0.505 q m, q = 0..23. Module m of a block is slot s = m mod 48
    of batch m // 48, on pair slot s // 2, on the first side of the pair for an
    even s and on the second for an odd s. The document numbers its modules from
    1; the indices here are that number minus one.

    Attributes:
        BATCH_SIZE       : Modules per batch: one z ring on the walls, one X row on the endcaps.
        HALF_SLOTS       : Pair slots per drift volume of a batch, 12 below and 12 above the cathode.
        DRIFT_PITCH      : Drift distance between neighbouring pair slots, in metres.
        DRIFT_EDGE       : Drift of the outermost pair slots, -6.0275 m below and +6.0275 m above the cathode.
        WALL_SIDE        : Distance |X| of the two long walls from the centre in the simulation frame, in metres.
        WALL_FIRST_Z     : z of the first wall ring, in metres.
        WALL_RING_PITCH  : z distance between neighbouring wall rings, in metres.
        WALL_MODULES     : Number of wall modules, 120 rings of 48.
        ENDCAP_FACE      : Distance |Z| of the two endcap faces from the centre, in metres.
        ENDCAP_FIRST_ROW : Simulation-frame X of the first endcap row, in metres.
        ENDCAP_ROW_PITCH : X distance between neighbouring endcap rows, in metres.
        ENDCAP_MODULES   : Number of endcap modules, 24 rows of 48.
        WALL_FACES       : Names of the right (even slot) and left (odd slot) walls.
        ENDCAP_FACES     : Names of the front (even slot) and back (odd slot) endcaps.
        BLOCKS           : Names of the two module blocks, walls first.
        transform        : CoordinateTransform into the sensor frame.
    """

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
        """Holds the transform into the sensor frame."""
        self.transform = CoordinateTransform()

    def wall_block(self) -> pd.DataFrame:
        """Returns the 5,760 wall modules in document order.

        Returns:
            One row per module with the columns ``block``, ``module_index``, ``face`` and the sensor-frame ``x``, ``y`` and ``z`` in metres.
        """
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
        """Returns the 1,152 endcap modules in document order.

        Returns:
            One row per module with the columns ``block``, ``module_index``, ``face`` and the sensor-frame ``x``, ``y`` and ``z`` in metres.
        """
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
        """Returns the drift coordinate d(j) of pair slots j = 0..23 of a batch.

        Args:
            pair_slots : Pair slots inside a batch, shape [K].

        Returns:
            The drift Y of each slot in the simulation frame as float64, shape [K], in metres.
        """
        pair_slots = np.asarray(pair_slots, dtype=np.int64)
        lower      = -self.DRIFT_EDGE + self.DRIFT_PITCH * pair_slots
        upper      = self.DRIFT_EDGE - self.DRIFT_PITCH * (pair_slots - self.HALF_SLOTS)
        return np.where(pair_slots < self.HALF_SLOTS, lower, upper)

    def _block_frame(self, block: str, modules: np.ndarray, faces: np.ndarray, simulation: np.ndarray) -> pd.DataFrame:
        """Returns the module table of one block with its positions converted into the sensor frame."""
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
        """Returns every documented module, the walls then the endcaps, each block in document order.

        Returns:
            One row per module, 6,912 rows, with the columns ``block``, ``module_index``, ``face`` and the sensor-frame ``x``, ``y`` and ``z`` in metres.
        """
        return pd.concat([self.wall_block(), self.endcap_block()], ignore_index=True)


class RawBinMapping:
    """Assignment of the raw light bins to the documented modules.

    A bin of the wall block holds the wall module ``bin - WALL_OFFSET`` and a
    bin of the endcap block the endcap module ``bin - ENDCAP_OFFSET``.
    ``LEAD_BIN`` holds the wall module ``LEAD_MODULE``, which the light places
    on the far wall although the offset rule would give module 0. The face-total
    bins are no modules. Block bounds are inclusive.

    Attributes:
        LEAD_BIN             : First bin of the raw files, a wall module by exception.
        LEAD_MODULE          : Wall module of the lead bin, the far-wall module of pair slot 0 in ring 0.
        WALL_BINS            : First and last bin of the wall block.
        WALL_OFFSET          : Offset from a wall-block bin to its wall module.
        ENDCAP_BINS          : First and last bin of the endcap block.
        ENDCAP_OFFSET        : Offset from an endcap-block bin to its endcap module.
        FACE_TOTAL_BINS      : Bins that carry face totals instead of a module.
        ABSENT_BINS          : Bins of the numbering that the raw files do not contain.
        EMPTY_WALL_MODULES   : Wall modules without a bin.
        EMPTY_ENDCAP_MODULES : Endcap modules without a bin.
        FACE_TOTAL_BLOCK     : Block name given to a face-total bin.
    """

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
        """Returns the bins the raw files must hold, in increasing order: 12, 13, 15 to 5772 and 5774 to 6925, 6,912 bins."""
        first = self.LEAD_BIN
        last  = self.FACE_TOTAL_BINS[-1]
        every = np.arange(first, last + 1, dtype=np.int64)
        return every[~np.isin(every, self.ABSENT_BINS)]

    def assign(self, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the block and the module of every bin.

        Args:
            bins : Raw bins, shape [K].

        Returns:
            The block of every bin as an object array of shape [K], ``wall``, ``endcap`` or ``FACE_TOTAL_BLOCK``, and its module index within the block as int64, shape [K], -1 for a face total.

        Raises:
            ValueError : If a bin belongs to neither block, is not the lead bin and is not a face total.
        """
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
    """Builds the stored channel geometry of the raw layout from the documented module positions, and checks it at build time.

    Every check fails loudly: the raw bins must be exactly ``RawBinMapping.raw_bins``;
    the 6,912 documented module positions must equal, as a set, the 6,912
    positions of the raw files, which differ only in the assignment of
    positions to bins; the kept channels must number ``CHANNEL_COUNT`` at as
    many distinct positions; and the modules without a channel must be exactly
    the empty modules of ``RawBinMapping``.

    Attributes:
        RAW_COLUMNS        : Layout columns of a raw file.
        GEOMETRY_COLUMNS   : Columns of the stored geometry, in order.
        CHANNEL_COUNT      : Number of kept channels.
        POSITION_DECIMALS  : Decimals, in metres, to which positions are rounded before they are compared as sets or for repeats.
        POSITION_TOLERANCE : Largest accepted difference between a documented and a raw position of the matched sets, in metres.
        logger             : Logger receiving the mapping summary.
        layout             : ApexModuleLayout of the documented modules.
        bin_mapping        : RawBinMapping of the raw bins.
    """

    RAW_COLUMNS        = ("bin", "x", "y", "z")
    GEOMETRY_COLUMNS   = ("channel_index", "raw_channel_index", "bin", "module_index", "face", "x", "y", "z")
    CHANNEL_COUNT      = 6909
    POSITION_DECIMALS  = 6
    POSITION_TOLERANCE = 1e-6

    def __init__(self, logger) -> None:
        """Holds the layout, the bin mapping and the logger.

        Args:
            logger : Logger receiving the mapping summary.
        """
        self.logger      = logger
        self.layout      = ApexModuleLayout()
        self.bin_mapping = RawBinMapping()

    def _raw_bins(self, raw_frame: pd.DataFrame) -> np.ndarray:
        """Returns the raw bins as int64 after checking that they are exactly the documented ones.

        Args:
            raw_frame : Raw layout with the columns of ``RAW_COLUMNS``, one row per raw channel.

        Returns:
            The bin of every raw row as int64, shape [6912].

        Raises:
            ValueError : If a bin is not an integer, or the bins are not exactly 12, 13, 15 to 5772 and 5774 to 6925, each once.
        """
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
        """Checks that the documented module positions and the raw positions are one set.

        Args:
            raw_frame : Raw layout, one row per raw channel.
            modules   : Documented module table of ``ApexModuleLayout.module_table``.

        Raises:
            ValueError : If the two sets differ in size or a matched pair differs by more than ``POSITION_TOLERANCE``.
        """
        raw        = self._sorted_positions(raw_frame[["x", "y", "z"]].to_numpy(dtype=np.float64))
        documented = self._sorted_positions(modules[["x", "y", "z"]].to_numpy(dtype=np.float64))

        if raw.shape != documented.shape:
            raise ValueError(f"The raw files hold {raw.shape[0]} positions and the document {documented.shape[0]} modules; the layouts must match one to one.")

        deviation = float(np.abs(raw - documented).max())
        if deviation > self.POSITION_TOLERANCE:
            raise ValueError(f"The set of raw positions differs from the set of documented module positions by up to {deviation:.3e} m; the raw files do not hold the documented layout.")

    def _sorted_positions(self, positions: np.ndarray) -> np.ndarray:
        """Returns positions [K, 3] in lexicographic order of their rounded coordinates."""
        rounded = np.round(positions, self.POSITION_DECIMALS)
        return positions[np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))]

    def _map_channels(self, bins: np.ndarray, modules: pd.DataFrame) -> pd.DataFrame:
        """Returns the stored geometry: the raw rows that hold a module, in raw order, with the position of their module.

        Args:
            bins    : Raw bin of every raw row, shape [6912].
            modules : Documented module table.

        Returns:
            The geometry with the columns of ``GEOMETRY_COLUMNS``.
        """
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
        """Checks the count and the distinct positions of the kept channels and the set of empty modules.

        Args:
            geometry : Stored geometry of ``_map_channels``.
            modules  : Documented module table.

        Raises:
            ValueError : If the channels do not number ``CHANNEL_COUNT``, two channels share a position, or the modules without a channel are not wall modules 0 and 2 and endcap module 0.
        """
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
        """Returns the stored geometry of a raw layout, built from the bins through the document.

        Args:
            raw_frame : Raw layout with the columns of ``RAW_COLUMNS``, one row per raw channel in raw file order.

        Returns:
            The stored geometry, one row per kept channel in raw order, with the columns of ``GEOMETRY_COLUMNS``: the store channel index, the raw row, the bin, the module index within its block, the face and the sensor-frame position in metres.

        Raises:
            ValueError : If any build-time check of the class fails.
        """
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
