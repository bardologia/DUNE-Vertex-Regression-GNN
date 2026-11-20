from __future__ import annotations

from pathlib import Path

import numpy as np
from torch_geometric.loader import DataLoader as GraphDataLoader

from configuration.data.general      import DatasetConfig
from configuration.inference.general import DatasetExplorerSources, VertexCoordinates
from configuration.shared            import IoNames
from pipelines.dataset.graph         import Graph
from pipelines.dataset.graph_dataset import GraphDataset
from pipelines.dataset.parquet_store import ParquetDatasetWriter, ParquetEventReader
from pipelines.inference.pipeline    import InferencePipeline
from pipelines.shared.store_geometry import CacheGeometryStamp
from tools.monitoring.logger         import Logger


class RawLightExportPipeline:
    def __init__(self, raw_input_dir, cache_directory, logger=None, dataset_config=None):
        self.raw_input_dir   = Path(raw_input_dir)
        self.cache_directory = Path(cache_directory)
        self.logger          = logger if logger is not None else Logger(log_dir=IoNames.logs_dir, name=IoNames.raw_light_log)
        self.config          = dataset_config if dataset_config is not None else DatasetConfig()

        self.writer               = None
        self.discovered_files     = None
        self.geometry_positions   = None
        self.geometry_fingerprint = None
        self.light_matrix         = None
        self.event_targets        = None

    def _prepare_writer(self):
        self.writer           = ParquetDatasetWriter(self.raw_input_dir, self.cache_directory, worker_count=self.config.data.store_worker_count, logger=self.logger)
        self.discovered_files = self.writer.discover_files()

    def _read_events(self):
        geometry_frame, geometry_hash = self.writer.build_canonical_geometry(self.discovered_files[0])
        self.geometry_positions       = geometry_frame[["x", "y", "z"]].values.astype(np.float32)
        self.geometry_fingerprint     = ParquetEventReader.fingerprint(geometry_frame)

        source_names, target_array, light_matrix = self.writer.read_all_events(self.discovered_files, geometry_hash, geometry_frame)

        self.light_matrix  = light_matrix.astype(np.int32)
        self.event_targets = target_array.astype(np.float32)

    def _write(self):
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        path = self.cache_directory / IoNames.raw_light_cache_file
        np.savez_compressed(path, geometry_positions=self.geometry_positions, light_matrix=self.light_matrix, event_targets=self.event_targets, **CacheGeometryStamp.payload(self.geometry_fingerprint))
        self.logger.subsection(f"Raw-light cache written to {path} ({self.light_matrix.shape[0]} events x {self.light_matrix.shape[1]} channels, geometry {self.geometry_fingerprint})")
        return path

    def run(self):
        self.logger.section("[Raw Light Export]")
        self._prepare_writer()
        self._read_events()
        return self._write()


class DatasetExplorerExportPipeline:
    def __init__(self, kind, cache_directory, logger=None, dataset_config=None):
        if kind not in DatasetExplorerSources.kinds:
            raise ValueError(f"unknown dataset source: {kind} (expected one of {list(DatasetExplorerSources.kinds)})")

        self.kind            = kind
        self.cache_directory = Path(cache_directory)
        self.logger          = logger if logger is not None else Logger(log_dir=IoNames.logs_dir, name=IoNames.dataset_explorer_log)
        self.config          = dataset_config if dataset_config is not None else DatasetConfig()

        self.geometry_positions   = None
        self.geometry_fingerprint = None
        self.light_matrix         = None
        self.event_targets        = None
        self.octant_frame         = None
        self.base_dataset         = None
        self.node_selector        = None

    def _load_store(self):
        reader = ParquetEventReader(self.config.data.parquet_store_dir).load_store()

        self.geometry_positions   = reader.geometry_frame[["x", "y", "z"]].values.astype(np.float32)
        self.geometry_fingerprint = reader.geometry_fingerprint
        self.event_targets        = reader.event_targets
        self.light_matrix         = reader.light_matrix
        self.octant_frame         = reader.octant_frame

        self.logger.subsection(f"Store loaded: {len(self.event_targets)} base events | {self.geometry_positions.shape[0]} channels | octant rows {len(self.octant_frame)}")

    def _prepare_base(self):
        count        = len(self.light_matrix)
        base_ids     = np.arange(count, dtype=np.float32)
        ones         = np.ones(count, dtype=np.float32)
        base_samples = np.column_stack([base_ids, ones, ones, ones, self.event_targets, base_ids]).astype(np.float32)

        graph              = Graph(self.config)
        self.base_dataset  = GraphDataset(base_samples, self.geometry_positions, self.light_matrix, graph, self.config.physics, augmentation=None, stats=None)
        self.node_selector = graph.node_selector

    def _base_sensors(self):
        count       = len(self.light_matrix)
        n_active    = np.empty(count, dtype=np.int32)
        total_light = np.empty(count, dtype=np.float32)

        position_chunks = []
        light_chunks    = []
        offsets         = np.zeros(count + 1, dtype=np.int64)

        with self.logger.track() as progress:
            task_id = progress.add_task("Selecting base-event sensors", total=count)
            for base_event_id in range(count):
                light                          = self.base_dataset._light_for_sample(base_event_id, base_event_id)
                active_positions, active_light = self.node_selector.select(self.geometry_positions, light)

                n_active[base_event_id]      = active_positions.shape[0]
                total_light[base_event_id]   = float(active_light.sum())
                offsets[base_event_id + 1]   = offsets[base_event_id] + active_positions.shape[0]

                position_chunks.append(active_positions.astype(np.float32))
                light_chunks.append(active_light.astype(np.float32))
                progress.advance(task_id)

        base_positions = np.concatenate(position_chunks, axis=0)
        base_light     = np.concatenate(light_chunks,    axis=0)
        return base_positions, base_light, offsets, n_active, total_light

    def _instances(self):
        if self.kind == DatasetExplorerSources.raw:
            count   = len(self.event_targets)
            base_id = np.arange(count, dtype=np.int32)
            signs   = np.ones((count, len(VertexCoordinates.names)), dtype=np.int8)
            targets = self.event_targets.astype(np.float32)
            octant  = np.array([DatasetExplorerSources.unreflected_octant] * count)
            return base_id, signs, targets, octant

        frame   = self.octant_frame
        base_id = frame["base_event_id"].values.astype(np.int32)
        signs   = frame[["sign_x", "sign_y", "sign_z"]].values.astype(np.int8)
        targets = frame[["target_x", "target_y", "target_z"]].values.astype(np.float32)
        octant  = frame["octant"].values.astype(str)
        return base_id, signs, targets, octant

    def _write(self, payload):
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        path = self.cache_directory / f"{self.kind}{IoNames.npz_suffix}"
        np.savez_compressed(path, **payload)
        self.logger.subsection(f"Dataset explorer cache written to {path} ({len(payload['gt'])} events)")
        return path

    def run(self):
        self.logger.section(f"[Dataset Explorer Export: {self.kind}]")
        self._load_store()
        self._prepare_base()

        base_positions, base_light, base_offsets, base_n_active, base_total_light = self._base_sensors()
        base_event_id, signs, targets, octant                                     = self._instances()

        payload = {
            "has_prediction" : np.array(False),
            "gt"             : targets,
            "signs"          : signs,
            "octant"         : octant,
            "base_event_id"  : base_event_id,
            "n_active"       : base_n_active[base_event_id],
            "total_light"    : base_total_light[base_event_id],
            "base_offsets"   : base_offsets,
            "base_positions" : base_positions,
            "base_light"     : base_light,
            **CacheGeometryStamp.payload(self.geometry_fingerprint),
        }
        return self._write(payload)


class EventExportPipeline:
    def __init__(self, run_directory, device=None, splits=None, batch_size=None):
        self.inference          = InferencePipeline(run_directory, device=device, splits=splits, batch_size=batch_size)
        self.run_directory      = self.inference.run_directory
        self.logger             = self.inference.logger
        self.analysis_directory = self.inference.analysis_directory
        self.node_selector      = None

    def _load(self):
        self.inference._load_run_config()
        self.inference._prepare()
        self.node_selector = Graph(self.inference.entry.dataset).node_selector

    def _predict_split(self, split_name):
        dataset              = self.inference.dataset_splits[split_name]
        loader               = GraphDataLoader(dataset, batch_size=self.inference.batch_size, shuffle=False)
        predictions, targets = self.inference.predictor.predict(loader)
        return dataset, predictions.numpy().astype(np.float32), targets.numpy().astype(np.float32)

    def _event_sensors(self, base_dataset, index):
        sample        = base_dataset.samples[index]
        light_row     = int(sample[0])
        signs         = sample[1:4].astype(np.float32)
        base_event_id = int(sample[7])

        positions = base_dataset.geometry_positions * signs[None, :]
        light     = base_dataset._light_for_sample(light_row, base_event_id)

        active_positions, active_light = self.node_selector.select(positions, light)
        return signs, base_event_id, active_positions.astype(np.float32), active_light.astype(np.float32)

    def _assemble_split(self, split_name):
        dataset, predictions, targets = self._predict_split(split_name)
        base_dataset                  = dataset.base_dataset if hasattr(dataset, "base_dataset") else dataset
        count                         = len(base_dataset.samples)
        axes                          = len(VertexCoordinates.names)

        residual  = predictions - targets
        error_xyz = np.abs(residual).astype(np.float32)
        error     = np.linalg.norm(residual, axis=1).astype(np.float32)

        base_event_id = np.empty(count, dtype=np.int32)
        signs         = np.empty((count, axes), dtype=np.int8)
        node_count    = np.empty(count, dtype=np.int32)
        total_light   = np.empty(count, dtype=np.float32)

        position_chunks = []
        light_chunks    = []
        offsets         = np.zeros(count + 1, dtype=np.int64)

        with self.logger.track() as progress:
            task_id = progress.add_task(f"Exporting {split_name} events", total=count)
            for index in range(count):
                event_signs, event_base_id, active_positions, active_light = self._event_sensors(base_dataset, index)

                base_event_id[index] = event_base_id
                signs[index]         = event_signs.astype(np.int8)
                node_count[index]    = active_positions.shape[0]
                total_light[index]   = float(active_light.sum())

                position_chunks.append(active_positions)
                light_chunks.append(active_light)
                offsets[index + 1] = offsets[index] + active_positions.shape[0]

                progress.advance(task_id)

        sensor_positions = np.concatenate(position_chunks, axis=0) if position_chunks else np.zeros((0, axes), dtype=np.float32)
        sensor_light     = np.concatenate(light_chunks,    axis=0) if light_chunks    else np.zeros((0,),    dtype=np.float32)

        return {
            "has_prediction"   : np.array(True),
            "gt"               : targets,
            "pred"             : predictions,
            "error"            : error,
            "error_xyz"        : error_xyz,
            "base_event_id"    : base_event_id,
            "signs"            : signs,
            "n_active"         : node_count,
            "total_light"      : total_light,
            "sensor_positions" : sensor_positions,
            "sensor_light"     : sensor_light,
            "offsets"          : offsets,
            **CacheGeometryStamp.payload(self.inference.store_geometry["geometry_fingerprint"]),
        }

    def _write_split(self, split_name, payload):
        directory = self.analysis_directory / split_name
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / IoNames.event_explorer_cache_file
        np.savez_compressed(path, **payload)
        self.logger.subsection(f"Event explorer cache written to {path} ({len(payload['gt'])} events)")

    def run(self):
        self.logger.section("[Event Export Pipeline]")
        self._load()

        for split_name in self.inference.splits:
            self.logger.section(f"[Export: {split_name}]")
            self._write_split(split_name, self._assemble_split(split_name))
