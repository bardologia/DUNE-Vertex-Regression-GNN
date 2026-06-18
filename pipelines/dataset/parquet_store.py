import os
import hashlib
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pa_parquet

from tools.monitoring.logger import Logger
from pipelines.dataset.coordinate_correction import CoordinateTransform
from pipelines.dataset.hot_channels          import HotChannelCorrector


class ParquetDatasetWriter:

    SIGN_COMBINATIONS = list(itertools.product((1, -1), repeat=3))

    def __init__(self, input_directory: Path, output_directory: Path, worker_count: int = 10, logger: Logger = None, hot_channels=None):
        self.input_directory  = Path(input_directory)
        self.store_directory  = Path(output_directory) / "parquet"
        self.geometry_path    = self.store_directory / "geometry.parquet"
        self.events_path       = self.store_directory / "events.parquet"
        self.octants_path     = self.store_directory / "octants.parquet"

        if hot_channels is None:
            from configuration.data.general import HotChannelConfig
            hot_channels = HotChannelConfig()

        self.worker_count = worker_count
        self.logger       = logger
        self.hot_channels = hot_channels
        self.transform    = CoordinateTransform()

    def discover_files(self):
        self.logger.section("[Parquet Store]")
        discovered_files = sorted(self.input_directory.glob("*.csv"))
        self.logger.subsection(f"Discovered {len(discovered_files)} raw CSV files in {self.input_directory}")
        return discovered_files

    def build_canonical_geometry(self, reference_file: Path):
        reference_frame = pd.read_csv(reference_file)
        geometry_frame  = reference_frame[["bin", "x", "y", "z"]].reset_index(drop=True)
        geometry_frame.insert(0, "channel_index", np.arange(len(geometry_frame), dtype=np.int32))

        canonical_bytes = np.ascontiguousarray(geometry_frame[["bin", "x", "y", "z"]].values)
        geometry_hash   = hashlib.md5(canonical_bytes.tobytes()).hexdigest()

        self.logger.subsection(f"Canonical geometry: {len(geometry_frame)} channels (stored once)")
        return geometry_frame, geometry_hash

    @staticmethod
    def read_event(source_path_string: str, geometry_hash: str):
        source_path = Path(source_path_string)
        transform   = CoordinateTransform()

        tokens   = source_path.stem.split("_")
        target_x = float(tokens[1])
        target_y = float(tokens[2])
        target_z = float(tokens[3])
        sensor_x, sensor_y, sensor_z = transform.apply_to_target(target_x, target_y, target_z)

        event_frame    = pd.read_csv(source_path)
        geometry_bytes = np.ascontiguousarray(event_frame[["bin", "x", "y", "z"]].values)
        event_hash     = hashlib.md5(geometry_bytes.tobytes()).hexdigest()
        if event_hash != geometry_hash:
            raise ValueError(f"Geometry mismatch in {source_path.name}; the store assumes one shared channel layout.")

        light_values = event_frame["ly_amount"].values
        if not np.all(np.mod(light_values, 1.0) == 0.0):
            raise ValueError(f"Non-integer photon count in {source_path.name}; light is expected to be integer counts.")
        light_vector = light_values.astype(np.int32)

        return source_path.name, sensor_x, sensor_y, sensor_z, light_vector

    def read_all_events(self, discovered_files, geometry_hash, channel_count):
        source_names = [None] * len(discovered_files)
        target_array = np.zeros((len(discovered_files), 3), dtype=np.float32)
        light_matrix = np.zeros((len(discovered_files), channel_count), dtype=np.int32)

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
                    target_array[position]  = (sensor_x, sensor_y, sensor_z)
                    light_matrix[position]  = light_vector
                    progress.advance(task_id)

        return source_names, target_array, light_matrix

    def correct_hot_channels(self, geometry_frame, light_matrix):
        positions = geometry_frame[["x", "y", "z"]].values
        return HotChannelCorrector(positions, self.hot_channels, self.logger).run(light_matrix)

    def build_octant_index(self, target_array):
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

    def write_parquet(self, geometry_frame, source_names, target_array, light_matrix, octant_rows):
        os.makedirs(self.store_directory, exist_ok=True)

        geometry_table = pa.table({
            "channel_index" : pa.array(geometry_frame["channel_index"].values, type=pa.int32()),
            "bin"           : pa.array(geometry_frame["bin"].values, type=pa.float64()),
            "x"             : pa.array(geometry_frame["x"].values, type=pa.float32()),
            "y"             : pa.array(geometry_frame["y"].values, type=pa.float32()),
            "z"             : pa.array(geometry_frame["z"].values, type=pa.float32()),
        })
        pa_parquet.write_table(geometry_table, self.geometry_path, compression="zstd")

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

        octant_light        = light_matrix[octant_frame["base_event_id"].values]
        octant_light_values = pa.array(octant_light.reshape(-1), type=pa.int32())
        octant_light_list   = pa.FixedSizeListArray.from_arrays(octant_light_values, channel_count)

        octants_table = pa.table({
            "base_event_id" : pa.array(octant_frame["base_event_id"].values, type=pa.int32()),
            "octant"        : pa.array(octant_frame["octant"].values, type=pa.string()),
            "sign_x"        : pa.array(octant_frame["sign_x"].values, type=pa.int8()),
            "sign_y"        : pa.array(octant_frame["sign_y"].values, type=pa.int8()),
            "sign_z"        : pa.array(octant_frame["sign_z"].values, type=pa.int8()),
            "target_x"      : pa.array(octant_frame["target_x"].values, type=pa.float32()),
            "target_y"      : pa.array(octant_frame["target_y"].values, type=pa.float32()),
            "target_z"      : pa.array(octant_frame["target_z"].values, type=pa.float32()),
            "light"         : octant_light_list,
        })
        pa_parquet.write_table(octants_table, self.octants_path, compression="zstd")

        self.logger.subsection(f"Geometry written to {self.geometry_path}")
        self.logger.subsection(f"Events written to {self.events_path}")
        self.logger.subsection(f"Octants written to {self.octants_path}")

    def report_sizes(self, octant_rows):
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

    def run(self):
        discovered_files = self.discover_files()

        geometry_frame, geometry_hash = self.build_canonical_geometry(discovered_files[0])
        source_names, target_array, light_matrix = self.read_all_events(discovered_files, geometry_hash, len(geometry_frame))
        light_matrix = self.correct_hot_channels(geometry_frame, light_matrix)
        octant_rows  = self.build_octant_index(target_array)

        self.write_parquet(geometry_frame, source_names, target_array, light_matrix, octant_rows)
        self.report_sizes(octant_rows)
        return self.store_directory


class ParquetEventReader:

    def __init__(self, store_directory: Path):
        self.store_directory = Path(store_directory)
        self.geometry_path   = self.store_directory / "geometry.parquet"
        self.events_path     = self.store_directory / "events.parquet"
        self.octants_path    = self.store_directory / "octants.parquet"

        self.geometry_frame      = None
        self.light_matrix        = None
        self.event_targets       = None
        self.octant_frame        = None
        self.octant_light_matrix = None

    OCTANT_FRAME_COLUMNS = ["base_event_id", "octant", "sign_x", "sign_y", "sign_z", "target_x", "target_y", "target_z"]

    def load_store(self, load_octant_light=True):
        self.geometry_frame = pa_parquet.read_table(self.geometry_path).to_pandas()
        channel_count       = len(self.geometry_frame)

        events_table        = pa_parquet.read_table(self.events_path)
        event_count         = events_table.num_rows
        flat_light          = events_table.column("light").combine_chunks().values.to_numpy()
        self.light_matrix   = flat_light.reshape(event_count, channel_count)
        self.event_targets  = np.column_stack([
            events_table.column("target_x").to_numpy(),
            events_table.column("target_y").to_numpy(),
            events_table.column("target_z").to_numpy(),
        ])

        if load_octant_light:
            octants_table            = pa_parquet.read_table(self.octants_path)
            octant_count             = octants_table.num_rows
            octant_flat_light        = octants_table.column("light").combine_chunks().values.to_numpy()
            self.octant_light_matrix = octant_flat_light.reshape(octant_count, channel_count)
            self.octant_frame        = octants_table.drop(["light"]).to_pandas()
        else:
            self.octant_frame = pa_parquet.read_table(self.octants_path, columns=self.OCTANT_FRAME_COLUMNS).to_pandas()
        return self

    def _build_event_frame(self, light_vector, sign_x, sign_y, sign_z):
        event_frame = pd.DataFrame({
            "bin"       : self.geometry_frame["bin"].values,
            "x"         : self.geometry_frame["x"].values * sign_x,
            "y"         : self.geometry_frame["y"].values * sign_y,
            "z"         : self.geometry_frame["z"].values * sign_z,
            "ly_amount" : light_vector,
        })
        return event_frame

    def iterate_corrected(self):
        for event_id in range(len(self.light_matrix)):
            light_vector = self.light_matrix[event_id]
            event_frame  = self._build_event_frame(light_vector, 1, 1, 1)
            target       = tuple(float(value) for value in self.event_targets[event_id])
            yield event_frame, target

    def iterate_augmented(self):
        for row_index, row in enumerate(self.octant_frame.itertuples(index=False)):
            light_vector = self.octant_light_matrix[row_index]
            event_frame  = self._build_event_frame(light_vector, row.sign_x, row.sign_y, row.sign_z)
            target       = (float(row.target_x), float(row.target_y), float(row.target_z))
            yield event_frame, target, row.octant


__all__ = [
    "ParquetDatasetWriter",
    "ParquetEventReader",
]
