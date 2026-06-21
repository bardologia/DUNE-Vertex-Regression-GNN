from __future__ import annotations

from pathlib import Path

import numpy as np

from configuration.data.general          import DatasetConfig
from pipelines.dataset.graph             import Graph
from pipelines.dataset.graph_dataset     import GraphDataset
from pipelines.dataset.parquet_store     import ParquetEventReader
from tools.monitoring.logger             import Logger


class DatasetExplorerExportPipeline:

    RAW       = "raw"
    AUGMENTED = "augmented"
    FILENAME  = {RAW: "raw.npz", AUGMENTED: "augmented.npz"}

    def __init__(self, kind, cache_directory, logger=None, dataset_config=None):
        if kind not in self.FILENAME:
            raise ValueError(f"unknown dataset source: {kind} (expected one of {list(self.FILENAME)})")

        self.kind            = kind
        self.cache_directory = Path(cache_directory)
        self.logger          = logger if logger is not None else Logger(log_dir="logs", name="dataset_explorer")
        self.config          = dataset_config if dataset_config is not None else DatasetConfig()

        self.geometry_positions = None
        self.light_matrix       = None
        self.event_targets      = None
        self.octant_frame       = None
        self.base_dataset       = None
        self.node_selector      = None

    def _load_store(self):
        reader = ParquetEventReader(self.config.data.parquet_store_dir).load_store(load_octant_light=False)

        self.geometry_positions = reader.geometry_frame[["x", "y", "z"]].values.astype(np.float32)
        self.event_targets      = reader.event_targets
        self.light_matrix       = reader.light_matrix
        self.octant_frame       = reader.octant_frame

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
        if self.kind == self.RAW:
            count   = len(self.event_targets)
            base_id = np.arange(count, dtype=np.int32)
            signs   = np.ones((count, 3), dtype=np.int8)
            targets = self.event_targets.astype(np.float32)
            octant  = np.array(["ppp"] * count)
            return base_id, signs, targets, octant

        frame   = self.octant_frame
        base_id = frame["base_event_id"].values.astype(np.int32)
        signs   = frame[["sign_x", "sign_y", "sign_z"]].values.astype(np.int8)
        targets = frame[["target_x", "target_y", "target_z"]].values.astype(np.float32)
        octant  = frame["octant"].values.astype(str)
        return base_id, signs, targets, octant

    def _write(self, payload):
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        path = self.cache_directory / self.FILENAME[self.kind]
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
        }
        return self._write(payload)
