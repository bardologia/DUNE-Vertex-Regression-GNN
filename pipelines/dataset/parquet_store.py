from __future__ import annotations

import hashlib
import itertools
from pathlib            import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pa_parquet

from configuration.data.general        import ChannelBins, StoreLayout
from configuration.shared              import IoNames, MeasurementUnits
from tools.monitoring.logger           import Logger
from pipelines.dataset.detector_layout import ChannelMapping, CoordinateTransform


class ParquetDatasetWriter:
    def __init__(self, input_directory: Path, output_directory: Path, worker_count: int, logger: Logger) -> None:
        self.input_directory = Path(input_directory)
        self.store_directory = Path(output_directory) / IoNames.parquet_dir
        self.geometry_path   = self.store_directory / IoNames.geometry_table_file
        self.events_path     = self.store_directory / IoNames.events_table_file
        self.octants_path    = self.store_directory / IoNames.octants_table_file

        self.worker_count    = worker_count
        self.logger          = logger
        self.channel_mapping = ChannelMapping(logger)

    def discover_files(self) -> list[Path]:
        self.logger.section("[Parquet Store]")
        discovered_files = sorted(self.input_directory.glob(IoNames.raw_event_glob))
        self.logger.subsection(f"Discovered {len(discovered_files)} raw CSV files in {self.input_directory}")
        return discovered_files

    def build_canonical_geometry(self, reference_file: Path) -> tuple[pd.DataFrame, str]:
        reference_frame = pd.read_csv(reference_file)
        raw_frame       = reference_frame[list(ChannelBins.raw_columns)].reset_index(drop=True)

        canonical_bytes = np.ascontiguousarray(raw_frame.values)
        geometry_hash   = hashlib.md5(canonical_bytes.tobytes()).hexdigest()

        geometry_frame  = self.channel_mapping.build(raw_frame)

        self.logger.subsection(f"Canonical geometry: {len(geometry_frame)} channels of {len(raw_frame)} raw bins (stored once)")
        return geometry_frame, geometry_hash

    def read_all_events(self, discovered_files: list[Path], geometry_hash: str, geometry_frame: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
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
                    position                                                = index_by_path[futures[future]]
                    source_name, sensor_x, sensor_y, sensor_z, light_vector = future.result()

                    source_names[position] = source_name
                    target_array[position] = (sensor_x, sensor_y, sensor_z)
                    light_matrix[position] = light_vector[kept_rows]
                    progress.advance(task_id)

        return source_names, target_array, light_matrix

    @staticmethod
    def read_event(source_path_string: str, geometry_hash: str) -> tuple[str, float, float, float, np.ndarray]:
        source_path = Path(source_path_string)
        transform   = CoordinateTransform()

        tokens   = source_path.stem.split("_")
        target_x = float(tokens[1])
        target_y = float(tokens[2])
        target_z = float(tokens[3])

        sensor_x, sensor_y, sensor_z = transform.apply_to_target(target_x, target_y, target_z)

        event_frame    = pd.read_csv(source_path)
        geometry_bytes = np.ascontiguousarray(event_frame[list(ChannelBins.raw_columns)].values)
        event_hash     = hashlib.md5(geometry_bytes.tobytes()).hexdigest()
        if event_hash != geometry_hash:
            raise ValueError(f"Geometry mismatch in {source_path.name}; the store assumes one shared raw layout.")

        light_values = event_frame["ly_amount"].values
        if not np.all(np.mod(light_values, 1.0) == 0.0):
            raise ValueError(f"Non-integer photon count in {source_path.name}; light is expected to be integer counts.")
        light_vector = light_values.astype(np.int32)

        return source_path.name, sensor_x, sensor_y, sensor_z, light_vector

    def build_octant_index(self, target_array: np.ndarray) -> list[tuple]:
        octant_rows = []
        symbol      = dict(StoreLayout.octant_symbols)
        for event_id in range(len(target_array)):
            base_x, base_y, base_z = target_array[event_id]
            emitted_targets        = set()
            for sign_x, sign_y, sign_z in itertools.product(StoreLayout.octant_signs, repeat=3):
                reflected_x = float(sign_x * base_x) + 0.0
                reflected_y = float(sign_y * base_y) + 0.0
                reflected_z = float(sign_z * base_z) + 0.0

                target_key = tuple(round(value, StoreLayout.reflection_decimals) for value in (reflected_x, reflected_y, reflected_z))
                if target_key in emitted_targets:
                    continue
                emitted_targets.add(target_key)

                octant_tag = symbol[sign_x] + symbol[sign_y] + symbol[sign_z]
                octant_rows.append((event_id, octant_tag, sign_x, sign_y, sign_z, reflected_x, reflected_y, reflected_z))

        self.logger.subsection(f"Octant index: {len(target_array)} events expanded to {len(octant_rows)} full-volume events")
        return octant_rows

    def write_parquet(self, geometry_frame: pd.DataFrame, source_names: list[str], target_array: np.ndarray, light_matrix: np.ndarray, octant_rows: list[tuple]) -> None:
        self.store_directory.mkdir(parents=True, exist_ok=True)

        pa_parquet.write_table(self.geometry_table(geometry_frame), self.geometry_path, compression=IoNames.parquet_compression)

        channel_count = light_matrix.shape[1]
        light_values  = pa.array(light_matrix.reshape(-1), type=pa.int32())
        light_list    = pa.FixedSizeListArray.from_arrays(light_values, channel_count)
        events_table  = pa.table({
            "event_id"        : pa.array(np.arange(len(source_names), dtype=np.int32)),
            "source_filename" : pa.array(source_names, type=pa.string()),
            "target_x"        : pa.array(target_array[:, 0], type=pa.float32()),
            "target_y"        : pa.array(target_array[:, 1], type=pa.float32()),
            "target_z"        : pa.array(target_array[:, 2], type=pa.float32()),
            "light"           : light_list,
        })
        pa_parquet.write_table(events_table, self.events_path, compression=IoNames.parquet_compression)

        octant_frame  = pd.DataFrame(octant_rows, columns=list(StoreLayout.octant_columns))
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
        pa_parquet.write_table(octants_table, self.octants_path, compression=IoNames.parquet_compression)

        self.logger.subsection(f"Geometry written to {self.geometry_path}")
        self.logger.subsection(f"Events written to {self.events_path}")
        self.logger.subsection(f"Octants written to {self.octants_path}")

    @classmethod
    def geometry_table(cls, geometry_frame: pd.DataFrame) -> pa.Table:
        columns = {}
        for name, alias in zip(StoreLayout.geometry_columns, StoreLayout.geometry_types):
            arrow_type    = pa.type_for_alias(alias)
            values        = geometry_frame[name].tolist() if pa.types.is_string(arrow_type) else geometry_frame[name].to_numpy()
            columns[name] = pa.array(values, type=arrow_type)
        return pa.table(columns)

    def report_sizes(self, octant_rows: list[tuple]) -> None:
        geometry_size = self.geometry_path.stat().st_size
        events_size   = self.events_path.stat().st_size
        octants_size  = self.octants_path.stat().st_size
        total_size    = geometry_size + events_size + octants_size

        self.logger.kv_table({
            IoNames.geometry_table_file : f"{geometry_size / MeasurementUnits.bytes_per_mb:.2f} MB",
            IoNames.events_table_file   : f"{events_size / MeasurementUnits.bytes_per_mb:.2f} MB",
            IoNames.octants_table_file  : f"{octants_size / MeasurementUnits.bytes_per_mb:.2f} MB",
            "total"                     : f"{total_size / MeasurementUnits.bytes_per_mb:.2f} MB",
            "full-volume events"        : f"{len(octant_rows)}",
        }, title="Parquet Store Summary")

    def run(self) -> Path:
        discovered_files = self.discover_files()

        geometry_frame, geometry_hash            = self.build_canonical_geometry(discovered_files[0])
        source_names, target_array, light_matrix = self.read_all_events(discovered_files, geometry_hash, geometry_frame)
        octant_rows                              = self.build_octant_index(target_array)

        self.write_parquet(geometry_frame, source_names, target_array, light_matrix, octant_rows)
        self.report_sizes(octant_rows)
        return self.store_directory


class ParquetEventReader:
    def __init__(self, store_directory: Path) -> None:
        self.store_directory = Path(store_directory)
        self.geometry_path   = self.store_directory / IoNames.geometry_table_file
        self.events_path     = self.store_directory / IoNames.events_table_file
        self.octants_path    = self.store_directory / IoNames.octants_table_file

        self.geometry_frame       = None
        self.geometry_fingerprint = None
        self.light_matrix         = None
        self.event_targets        = None
        self.octant_frame         = None

    def load_geometry(self) -> "ParquetEventReader":
        self.geometry_frame = pa_parquet.read_table(self.geometry_path).to_pandas()

        if tuple(self.geometry_frame.columns) != StoreLayout.geometry_columns:
            raise ValueError(f"The store {self.store_directory} has the geometry columns {list(self.geometry_frame.columns)}, expected {list(StoreLayout.geometry_columns)}. A store without raw_channel_index was built from the raw positions before the documented channel mapping, with misplaced sensors and the three face totals as channels; rebuild it with main/build_parquet_store.py.")

        self.geometry_fingerprint = self.fingerprint(self.geometry_frame)
        return self

    @classmethod
    def fingerprint(cls, geometry_frame: pd.DataFrame) -> str:
        table  = ParquetDatasetWriter.geometry_table(geometry_frame)
        digest = hashlib.sha256()
        for name in StoreLayout.geometry_columns:
            column = table.column(name)
            digest.update(name.encode("utf-8"))
            if pa.types.is_string(column.type):
                digest.update("\n".join(column.to_pylist()).encode("utf-8"))
            else:
                digest.update(np.ascontiguousarray(column.to_numpy()).tobytes())
        return digest.hexdigest()

    def load_store(self) -> "ParquetEventReader":
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
        schema = pa_parquet.read_schema(self.octants_path)
        if "light" in schema.names:
            raise ValueError(f"{self.octants_path} still carries a per-octant light column from the previous store layout. Rebuild the store with data.build_store=true; octant light is now taken from events.parquet by base_event_id.")

        return pa_parquet.read_table(self.octants_path, columns=list(StoreLayout.octant_columns)).to_pandas()

    def iterate_corrected(self):
        for event_id in range(len(self.light_matrix)):
            light_vector = self.light_matrix[event_id]
            event_frame  = self._build_event_frame(light_vector, 1, 1, 1)
            target       = tuple(float(value) for value in self.event_targets[event_id])
            yield event_frame, target

    def _build_event_frame(self, light_vector: np.ndarray, sign_x: int, sign_y: int, sign_z: int) -> pd.DataFrame:
        event_frame = pd.DataFrame({
            "bin"       : self.geometry_frame["bin"].values,
            "x"         : self.geometry_frame["x"].values * sign_x,
            "y"         : self.geometry_frame["y"].values * sign_y,
            "z"         : self.geometry_frame["z"].values * sign_z,
            "ly_amount" : light_vector,
        })
        return event_frame


__all__ = [
    "ParquetDatasetWriter",
    "ParquetEventReader",
]
