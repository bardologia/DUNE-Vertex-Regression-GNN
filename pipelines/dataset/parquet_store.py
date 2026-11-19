"""Parquet store of the simulated events: writer from raw CSV files and reader into arrays.

The store holds three tables. ``geometry.parquet`` keeps the channel layout
once, built from the raw bins through the documented module mapping of
``pipelines.dataset.detector_layout``: one row per kept channel with its store
index, its raw row, its bin, its module and face, and its sensor-frame
position. ``events.parquet`` keeps one row per simulated event with its vertex
and its light vector, one entry per kept channel, and ``octants.parquet``
indexes the distinct mirror images of every event in the eight detector
octants. Positions and vertices are in metres, light is in raw photon counts.
"""

from __future__ import annotations

import os
import hashlib
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pa_parquet

from tools.monitoring.logger           import Logger
from pipelines.dataset.detector_layout import ChannelMapping, CoordinateTransform


class ParquetDatasetWriter:
    """Builds the three-table Parquet store from a directory of raw CSV events.

    Each raw file is named ``<prefix>_<x>_<y>_<z>_<...>.csv``, where x, y and z
    are the simulated vertex in metres in the simulation frame, and holds the
    columns ``bin``, ``x``, ``y``, ``z`` and ``ly_amount`` with one row per raw
    bin. Every file must carry the same raw layout, checked by an MD5 hash of
    its ``bin``, ``x``, ``y`` and ``z`` columns. The stored geometry is built
    from the bins by ``ChannelMapping``, never from the raw positions, and the
    light of every event keeps the entries of the kept channels only, in raw
    row order, so the face totals never reach the store.

    Attributes:
        SIGN_COMBINATIONS : The eight (sign_x, sign_y, sign_z) reflections, identity first.
        GEOMETRY_TYPES    : Arrow type of every column of ``ChannelMapping.GEOMETRY_COLUMNS``, in that order.
        input_directory   : Directory holding the raw CSV files.
        store_directory   : Directory that receives the three Parquet tables, ``<output_directory>/parquet``.
        geometry_path     : Path of geometry.parquet.
        events_path       : Path of events.parquet.
        octants_path      : Path of octants.parquet.
        worker_count      : Worker processes reading raw files.
        logger            : Logger receiving the build report.
        channel_mapping   : ChannelMapping that builds the stored geometry from the raw layout.
    """

    SIGN_COMBINATIONS = list(itertools.product((1, -1), repeat=3))
    GEOMETRY_TYPES    = (pa.int32(), pa.int32(), pa.int32(), pa.int32(), pa.string(), pa.float32(), pa.float32(), pa.float32())

    def __init__(self, input_directory: Path, output_directory: Path, worker_count: int = 10, logger: Logger = None) -> None:
        """Resolves the store paths and the channel mapping.

        Args:
            input_directory  : Directory holding the raw CSV files.
            output_directory : Directory that receives the ``parquet`` store directory.
            worker_count     : Worker processes reading raw files.
            logger           : Logger receiving the build report.
        """
        self.input_directory = Path(input_directory)
        self.store_directory = Path(output_directory) / "parquet"
        self.geometry_path   = self.store_directory / "geometry.parquet"
        self.events_path     = self.store_directory / "events.parquet"
        self.octants_path    = self.store_directory / "octants.parquet"

        self.worker_count    = worker_count
        self.logger          = logger
        self.channel_mapping = ChannelMapping(logger)

    def discover_files(self) -> list[Path]:
        """Returns the raw CSV files of the input directory in sorted order."""
        self.logger.section("[Parquet Store]")
        discovered_files = sorted(self.input_directory.glob("*.csv"))
        self.logger.subsection(f"Discovered {len(discovered_files)} raw CSV files in {self.input_directory}")
        return discovered_files

    def build_canonical_geometry(self, reference_file: Path) -> tuple[pd.DataFrame, str]:
        """Reads the raw layout from one raw file and builds the stored geometry from its bins.

        Args:
            reference_file : Raw CSV file whose layout defines the store.

        Returns:
            The stored geometry of ``ChannelMapping.build``, one row per kept channel, and the MD5 hash of the raw ``bin``, ``x``, ``y`` and ``z`` columns that every other file must match.

        Raises:
            ValueError : If a build-time check of the channel mapping fails.
        """
        reference_frame = pd.read_csv(reference_file)
        raw_frame       = reference_frame[list(ChannelMapping.RAW_COLUMNS)].reset_index(drop=True)

        canonical_bytes = np.ascontiguousarray(raw_frame.values)
        geometry_hash   = hashlib.md5(canonical_bytes.tobytes()).hexdigest()

        geometry_frame  = self.channel_mapping.build(raw_frame)

        self.logger.subsection(f"Canonical geometry: {len(geometry_frame)} channels of {len(raw_frame)} raw bins (stored once)")
        return geometry_frame, geometry_hash

    @staticmethod
    def read_event(source_path_string: str, geometry_hash: str) -> tuple[str, float, float, float, np.ndarray]:
        """Reads one raw event file and checks it against the canonical layout.

        Static so that it pickles cheaply into the worker processes.

        Args:
            source_path_string : Path of the raw CSV file.
            geometry_hash      : MD5 hash of the canonical raw layout.

        Returns:
            The file name, the vertex x, y and z in the sensor frame in metres, and the photon counts of every raw bin as int32 of shape [B], in raw row order.

        Raises:
            ValueError : If the raw layout of the file differs from the canonical one, or a photon count is not an integer.
        """
        source_path = Path(source_path_string)
        transform   = CoordinateTransform()

        tokens   = source_path.stem.split("_")
        target_x = float(tokens[1])
        target_y = float(tokens[2])
        target_z = float(tokens[3])
        sensor_x, sensor_y, sensor_z = transform.apply_to_target(target_x, target_y, target_z)

        event_frame    = pd.read_csv(source_path)
        geometry_bytes = np.ascontiguousarray(event_frame[list(ChannelMapping.RAW_COLUMNS)].values)
        event_hash     = hashlib.md5(geometry_bytes.tobytes()).hexdigest()
        if event_hash != geometry_hash:
            raise ValueError(f"Geometry mismatch in {source_path.name}; the store assumes one shared raw layout.")

        light_values = event_frame["ly_amount"].values
        if not np.all(np.mod(light_values, 1.0) == 0.0):
            raise ValueError(f"Non-integer photon count in {source_path.name}; light is expected to be integer counts.")
        light_vector = light_values.astype(np.int32)

        return source_path.name, sensor_x, sensor_y, sensor_z, light_vector

    def read_all_events(self, discovered_files: list[Path], geometry_hash: str, geometry_frame: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Reads every raw event in a process pool, keeping the sorted file order and the light of the kept channels only.

        The light of every event is cut to the raw rows of the stored
        geometry as soon as it arrives, so the face totals are gone before any
        statistic sees the light.

        Args:
            discovered_files : Raw CSV files in the order that defines the event ids.
            geometry_hash    : MD5 hash of the canonical raw layout.
            geometry_frame   : Stored geometry, whose ``raw_channel_index`` column selects the kept raw rows.

        Returns:
            The file names, the vertices as float32 of shape [N, 3] in metres, and the light matrix as int32 of shape [N, C] in photon counts, C being the number of kept channels.
        """
        kept_rows    = geometry_frame["raw_channel_index"].to_numpy(dtype=np.int64)
        source_names = [None] * len(discovered_files)
        target_array = np.zeros((len(discovered_files), 3), dtype=np.float32)
        light_matrix = np.zeros((len(discovered_files), kept_rows.shape[0]), dtype=np.int32)

        index_by_path = {str(path): position for position, path in enumerate(discovered_files)}

        with self.logger.track() as progress:
            task_id = progress.add_task("Reading events into store", total=len(discovered_files))
            with ProcessPoolExecutor(max_workers=self.worker_count) as executor:
                futures = {
                    executor.submit(self.read_event, str(path), geometry_hash): str(path)
                    for path in discovered_files
                }
                for future in as_completed(futures):
                    position = index_by_path[futures[future]]
                    source_name, sensor_x, sensor_y, sensor_z, light_vector = future.result()
                    source_names[position] = source_name
                    target_array[position] = (sensor_x, sensor_y, sensor_z)
                    light_matrix[position] = light_vector[kept_rows]
                    progress.advance(task_id)

        return source_names, target_array, light_matrix

    def build_octant_index(self, target_array: np.ndarray) -> list[tuple]:
        """Lists the distinct mirror images of every event over the eight octants.

        A reflection that maps a vertex onto an image already listed, as for
        vertices on a symmetry plane, is emitted once.

        Args:
            target_array : Vertices in the sensor frame, shape [N, 3], in metres.

        Returns:
            One tuple per mirror image: base event id, octant tag, the three signs and the reflected vertex in metres.
        """
        octant_rows = []
        symbol      = {1: "p", -1: "n"}
        for event_id in range(len(target_array)):
            base_x, base_y, base_z = target_array[event_id]
            emitted_targets = set()
            for sign_x, sign_y, sign_z in self.SIGN_COMBINATIONS:
                reflected_x = float(sign_x * base_x) + 0.0
                reflected_y = float(sign_y * base_y) + 0.0
                reflected_z = float(sign_z * base_z) + 0.0

                target_key = (round(reflected_x, 6), round(reflected_y, 6), round(reflected_z, 6))
                if target_key in emitted_targets:
                    continue
                emitted_targets.add(target_key)

                octant_tag = symbol[sign_x] + symbol[sign_y] + symbol[sign_z]
                octant_rows.append((event_id, octant_tag, sign_x, sign_y, sign_z, reflected_x, reflected_y, reflected_z))

        self.logger.subsection(f"Octant index: {len(target_array)} events expanded to {len(octant_rows)} full-volume events")
        return octant_rows

    @classmethod
    def geometry_table(cls, geometry_frame: pd.DataFrame) -> pa.Table:
        """Returns a geometry as an Arrow table with the stored column order and types.

        Args:
            geometry_frame : Geometry with the columns of ``ChannelMapping.GEOMETRY_COLUMNS``, positions in metres in any float precision.

        Returns:
            The table as ``write_parquet`` stores it: int32 indices and bins, string faces and float32 positions.
        """
        columns = {}
        for name, arrow_type in zip(ChannelMapping.GEOMETRY_COLUMNS, cls.GEOMETRY_TYPES):
            values        = geometry_frame[name].tolist() if pa.types.is_string(arrow_type) else geometry_frame[name].to_numpy()
            columns[name] = pa.array(values, type=arrow_type)
        return pa.table(columns)

    def write_parquet(self, geometry_frame: pd.DataFrame, source_names: list[str], target_array: np.ndarray, light_matrix: np.ndarray, octant_rows: list[tuple]) -> None:
        """Writes the geometry, events and octants tables with zstd compression.

        Args:
            geometry_frame : Stored geometry of ``ChannelMapping.build``, one row per kept channel.
            source_names   : Raw file name of every event.
            target_array   : Vertices in the sensor frame, shape [N, 3], in metres.
            light_matrix   : Photon counts of the kept channels, shape [N, C].
            octant_rows    : Rows of the octant index.
        """
        os.makedirs(self.store_directory, exist_ok=True)

        pa_parquet.write_table(self.geometry_table(geometry_frame), self.geometry_path, compression="zstd")

        channel_count   = light_matrix.shape[1]
        light_values    = pa.array(light_matrix.reshape(-1), type=pa.int32())
        light_list      = pa.FixedSizeListArray.from_arrays(light_values, channel_count)
        events_table = pa.table({
            "event_id"        : pa.array(np.arange(len(source_names), dtype=np.int32)),
            "source_filename" : pa.array(source_names, type=pa.string()),
            "target_x"        : pa.array(target_array[:, 0], type=pa.float32()),
            "target_y"        : pa.array(target_array[:, 1], type=pa.float32()),
            "target_z"        : pa.array(target_array[:, 2], type=pa.float32()),
            "light"           : light_list,
        })
        pa_parquet.write_table(events_table, self.events_path, compression="zstd")

        octant_frame = pd.DataFrame(
            octant_rows,
            columns=["base_event_id", "octant", "sign_x", "sign_y", "sign_z", "target_x", "target_y", "target_z"],
        )

        octants_table = pa.table({
            "base_event_id" : pa.array(octant_frame["base_event_id"].values, type=pa.int32()),
            "octant"        : pa.array(octant_frame["octant"].values, type=pa.string()),
            "sign_x"        : pa.array(octant_frame["sign_x"].values, type=pa.int8()),
            "sign_y"        : pa.array(octant_frame["sign_y"].values, type=pa.int8()),
            "sign_z"        : pa.array(octant_frame["sign_z"].values, type=pa.int8()),
            "target_x"      : pa.array(octant_frame["target_x"].values, type=pa.float32()),
            "target_y"      : pa.array(octant_frame["target_y"].values, type=pa.float32()),
            "target_z"      : pa.array(octant_frame["target_z"].values, type=pa.float32()),
        })
        pa_parquet.write_table(octants_table, self.octants_path, compression="zstd")

        self.logger.subsection(f"Geometry written to {self.geometry_path}")
        self.logger.subsection(f"Events written to {self.events_path}")
        self.logger.subsection(f"Octants written to {self.octants_path}")

    def report_sizes(self, octant_rows: list[tuple]) -> None:
        """Logs the size of each written table and the number of mirror images.

        Args:
            octant_rows : Rows of the octant index.
        """
        geometry_size = self.geometry_path.stat().st_size
        events_size   = self.events_path.stat().st_size
        octants_size  = self.octants_path.stat().st_size
        total_size    = geometry_size + events_size + octants_size

        self.logger.kv_table({
            "geometry.parquet"   : f"{geometry_size / 1e6:.2f} MB",
            "events.parquet"     : f"{events_size / 1e6:.2f} MB",
            "octants.parquet"    : f"{octants_size / 1e6:.2f} MB",
            "total"              : f"{total_size / 1e6:.2f} MB",
            "full-volume events" : f"{len(octant_rows)}",
        }, title="Parquet Store Summary")

    def run(self) -> Path:
        """Builds the whole store: geometry from the documented mapping, events, octant index and the three tables.

        Returns:
            The store directory.

        Raises:
            ValueError : If a build-time check of the channel mapping fails, or a raw file breaks the shared layout or holds a non-integer count.
        """
        discovered_files = self.discover_files()

        geometry_frame, geometry_hash            = self.build_canonical_geometry(discovered_files[0])
        source_names, target_array, light_matrix = self.read_all_events(discovered_files, geometry_hash, geometry_frame)
        octant_rows                              = self.build_octant_index(target_array)

        self.write_parquet(geometry_frame, source_names, target_array, light_matrix, octant_rows)
        self.report_sizes(octant_rows)
        return self.store_directory


class ParquetEventReader:
    """Loads the Parquet store into arrays and rebuilds per-event tables from it.

    Attributes:
        OCTANT_COLUMNS       : Columns of the octant index.
        store_directory      : Directory holding the three Parquet tables.
        geometry_path        : Path of geometry.parquet.
        events_path          : Path of events.parquet.
        octants_path         : Path of octants.parquet.
        geometry_frame       : Geometry table with the columns of ``ChannelMapping.GEOMETRY_COLUMNS``, one row per channel, positions in metres; None before ``load_geometry``.
        geometry_fingerprint : sha256 of the stored geometry from ``fingerprint``; None before ``load_geometry``.
        light_matrix         : Photon counts as int32, shape [N, C]; None before ``load_store``.
        event_targets        : Vertices in the sensor frame as float32, shape [N, 3], in metres; None before ``load_store``.
        octant_frame         : Octant index, one row per distinct mirror image; None before ``load_store``.
    """

    OCTANT_COLUMNS = ("base_event_id", "octant", "sign_x", "sign_y", "sign_z", "target_x", "target_y", "target_z")

    def __init__(self, store_directory: Path) -> None:
        """Resolves the store paths; nothing is read until ``load_geometry`` or ``load_store``.

        Args:
            store_directory : Directory holding the three Parquet tables.
        """
        self.store_directory = Path(store_directory)
        self.geometry_path   = self.store_directory / "geometry.parquet"
        self.events_path     = self.store_directory / "events.parquet"
        self.octants_path    = self.store_directory / "octants.parquet"

        self.geometry_frame       = None
        self.geometry_fingerprint = None
        self.light_matrix         = None
        self.event_targets        = None
        self.octant_frame         = None

    @classmethod
    def fingerprint(cls, geometry_frame: pd.DataFrame) -> str:
        """Returns the sha256 of a geometry in its stored form: every column of ``ChannelMapping.GEOMETRY_COLUMNS`` in order, numbers by their stored bytes and face names as text.

        The geometry is first cast to the stored column types, so the frame
        that ``ChannelMapping.build`` returns and the frame read back from
        geometry.parquet give the same digest.

        Args:
            geometry_frame : Geometry with the columns of ``ChannelMapping.GEOMETRY_COLUMNS``.

        Returns:
            The hexadecimal digest; two geometries share it exactly when their stored tables are equal cell for cell.
        """
        table  = ParquetDatasetWriter.geometry_table(geometry_frame)
        digest = hashlib.sha256()
        for name in ChannelMapping.GEOMETRY_COLUMNS:
            column = table.column(name)
            digest.update(name.encode("utf-8"))
            if pa.types.is_string(column.type):
                digest.update("\n".join(column.to_pylist()).encode("utf-8"))
            else:
                digest.update(np.ascontiguousarray(column.to_numpy()).tobytes())
        return digest.hexdigest()

    def load_geometry(self) -> "ParquetEventReader":
        """Reads the geometry table alone, checks its columns and fingerprints it, and returns the reader itself.

        Raises:
            ValueError : If the geometry table lacks the columns of the documented channel mapping, as a store written before the mapping lacks ``raw_channel_index``.
        """
        self.geometry_frame = pa_parquet.read_table(self.geometry_path).to_pandas()

        if tuple(self.geometry_frame.columns) != ChannelMapping.GEOMETRY_COLUMNS:
            raise ValueError(f"The store {self.store_directory} has the geometry columns {list(self.geometry_frame.columns)}, expected {list(ChannelMapping.GEOMETRY_COLUMNS)}. A store without raw_channel_index was built from the raw positions before the documented channel mapping, with misplaced sensors and the three face totals as channels; rebuild it with main/build_parquet_store.py.")

        self.geometry_fingerprint = self.fingerprint(self.geometry_frame)
        return self

    def load_store(self) -> "ParquetEventReader":
        """Reads the three tables into memory and returns the reader itself.

        Raises:
            ValueError : If the geometry fails ``load_geometry``, or the octant index still carries the per-octant light column of an older layout.
        """
        self.load_geometry()
        channel_count = len(self.geometry_frame)

        events_table        = pa_parquet.read_table(self.events_path)
        event_count         = events_table.num_rows
        flat_light          = events_table.column("light").combine_chunks().values.to_numpy()
        self.light_matrix   = flat_light.reshape(event_count, channel_count)
        self.event_targets  = np.column_stack([
            events_table.column("target_x").to_numpy(),
            events_table.column("target_y").to_numpy(),
            events_table.column("target_z").to_numpy(),
        ])

        self.octant_frame = self._read_octants()
        return self

    def _read_octants(self) -> pd.DataFrame:
        """Returns the octant index after refusing the per-octant light column of an older layout."""
        schema = pa_parquet.read_schema(self.octants_path)
        if "light" in schema.names:
            raise ValueError(f"{self.octants_path} still carries a per-octant light column from the previous store layout. Rebuild the store with data.build_store=true; octant light is now taken from events.parquet by base_event_id.")

        return pa_parquet.read_table(self.octants_path, columns=list(self.OCTANT_COLUMNS)).to_pandas()

    def _build_event_frame(self, light_vector: np.ndarray, sign_x: int, sign_y: int, sign_z: int) -> pd.DataFrame:
        """Rebuilds the per-channel table of one event in a chosen octant.

        Args:
            light_vector : Photon counts of the event, shape [C].
            sign_x       : Reflection sign applied to the sensor x, +1 or -1.
            sign_y       : Reflection sign applied to the sensor y, +1 or -1.
            sign_z       : Reflection sign applied to the sensor z, +1 or -1.

        Returns:
            A table with the columns ``bin``, ``x``, ``y``, ``z`` in metres and ``ly_amount`` in photon counts.
        """
        event_frame = pd.DataFrame({
            "bin"       : self.geometry_frame["bin"].values,
            "x"         : self.geometry_frame["x"].values * sign_x,
            "y"         : self.geometry_frame["y"].values * sign_y,
            "z"         : self.geometry_frame["z"].values * sign_z,
            "ly_amount" : light_vector,
        })
        return event_frame

    def iterate_corrected(self):
        """Yields every stored event as a per-channel table together with its vertex.

        Yields:
            The per-channel table of the event in the simulated octant and its vertex (x, y, z) in metres.
        """
        for event_id in range(len(self.light_matrix)):
            light_vector = self.light_matrix[event_id]
            event_frame  = self._build_event_frame(light_vector, 1, 1, 1)
            target       = tuple(float(value) for value in self.event_targets[event_id])
            yield event_frame, target


__all__ = [
    "ParquetDatasetWriter",
    "ParquetEventReader",
]
