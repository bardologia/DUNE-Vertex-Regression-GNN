from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.data.general      import DatasetConfig
from pipelines.dataset.parquet_store import ParquetDatasetWriter
from tools.monitoring.logger         import Logger


class RawLightExportPipeline:

    FILENAME = "raw_light.npz"

    def __init__(self, raw_input_dir, cache_directory, logger=None, dataset_config=None):
        self.raw_input_dir   = Path(raw_input_dir)
        self.cache_directory = Path(cache_directory)
        self.logger          = logger if logger is not None else Logger(log_dir="logs", name="raw_light")
        self.config          = dataset_config if dataset_config is not None else DatasetConfig()

        self.writer             = None
        self.discovered_files   = None
        self.geometry_positions = None
        self.light_matrix       = None
        self.event_targets      = None

    def _prepare_writer(self):
        self.writer           = ParquetDatasetWriter(self.raw_input_dir, self.cache_directory, worker_count=self.config.data.store_worker_count, logger=self.logger)
        self.discovered_files = self.writer.discover_files()

    def _read_events(self):
        geometry_frame, geometry_hash = self.writer.build_canonical_geometry(self.discovered_files[0])
        self.geometry_positions       = geometry_frame[["x", "y", "z"]].values.astype(np.float32)

        source_names, target_array, light_matrix = self.writer.read_all_events(self.discovered_files, geometry_hash, len(geometry_frame))
        self.light_matrix  = light_matrix.astype(np.int32)
        self.event_targets = target_array.astype(np.float32)

    def _write(self):
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        path = self.cache_directory / self.FILENAME
        np.savez_compressed(path, geometry_positions=self.geometry_positions, light_matrix=self.light_matrix, event_targets=self.event_targets)
        self.logger.subsection(f"Raw-light cache written to {path} ({self.light_matrix.shape[0]} events x {self.light_matrix.shape[1]} channels)")
        return path

    def run(self):
        self.logger.section("[Raw Light Export]")
        self._prepare_writer()
        self._read_events()
        return self._write()
