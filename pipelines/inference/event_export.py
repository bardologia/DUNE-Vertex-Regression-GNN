from __future__ import annotations

from pathlib import Path

import numpy as np
from torch_geometric.loader import DataLoader as GraphDataLoader

from pipelines.dataset.graph     import Graph
from pipelines.inference.pipeline import InferencePipeline


class EventExportPipeline:

    FILENAME = "event_explorer.npz"

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

        residual  = predictions - targets
        error_xyz = np.abs(residual).astype(np.float32)
        error     = np.linalg.norm(residual, axis=1).astype(np.float32)

        base_event_id = np.empty(count, dtype=np.int32)
        signs         = np.empty((count, 3), dtype=np.int8)
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

        sensor_positions = np.concatenate(position_chunks, axis=0) if position_chunks else np.zeros((0, 3), dtype=np.float32)
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
        }

    def _write_split(self, split_name, payload):
        directory = self.analysis_directory / split_name
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / self.FILENAME
        np.savez_compressed(path, **payload)
        self.logger.subsection(f"Event explorer cache written to {path} ({len(payload['gt'])} events)")

    def run(self):
        self.logger.section("[Event Export Pipeline]")
        self._load()

        for split_name in self.inference.splits:
            if split_name in self.inference.dataset_splits:
                self.logger.section(f"[Export: {split_name}]")
                self._write_split(split_name, self._assemble_split(split_name))
