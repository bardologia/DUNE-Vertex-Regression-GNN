from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

import numpy as np
from torch_geometric.loader import DataLoader as GraphDataLoader

from configuration.entry             import TrainEntryConfig
from models                          import get_model
from tools.monitoring.logger         import Logger
from tools.runtime.config_cli        import ConfigCli
from pipelines.dataset.normalization import NormalizationStats
from pipelines.dataset.pipeline      import DatasetPipeline
from pipelines.inference.metrics     import InferenceMetrics
from pipelines.inference.predictor   import Predictor

from pipelines.sweep.grid   import SweepGrid
from pipelines.sweep.plots  import SweepPlots
from pipelines.sweep.report import SweepReport


class EnergyEfficiencySweepPipeline:
    GRID_METRIC_KEYS = ("euclidean_mean", "euclidean_median", "mae", "rmse", "r2")

    def __init__(self, config, logger=None):
        self.config           = config
        self.run_directory    = Path(config.run_directory)
        self.logger           = logger if logger is not None else Logger(log_dir=str(self.run_directory / "logs"), name="sweep")
        self.device           = config.device
        self.split            = config.split
        self.batch_size       = config.batch_size

        self.entry            = None
        self.stats            = None
        self.split_base_ids   = None
        self.baseline         = None
        self.efficiency_seed  = None
        self.predictor        = None
        self.dataset_pipeline = None
        self.grid             = None
        self.records          = None
        self.output_directory = None

    def _resolve_output_directory(self):
        stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        label     = f"energy_efficiency_{self.config.run_name}_{stamp}" if self.config.run_name else f"energy_efficiency_{stamp}"
        directory = self.run_directory / "sweeps" / label
        directory.mkdir(parents=True, exist_ok=True)
        self.output_directory = directory

    def _load_run(self):
        resolved_path = self.run_directory / "metadata" / "resolved_config.json"
        if not resolved_path.exists():
            raise FileNotFoundError(f"No training run found at {self.run_directory}: expected {resolved_path}. Pass --run_directory pointing at a completed run (runs/<model>_<stamp>).")

        self.entry          = ConfigCli.load_resolved(TrainEntryConfig(), resolved_path)
        self.stats          = NormalizationStats.load(self.run_directory / "metadata")
        self.split_base_ids = json.loads((self.run_directory / "metadata" / "split_base_ids.json").read_text(encoding="utf-8"))

        self.baseline        = (self.entry.dataset.physics.scale_factor, self.entry.dataset.physics.detection_efficiency)
        self.efficiency_seed = self.entry.dataset.physics.efficiency_seed

        self.logger.kv_table({
            "Run directory" : str(self.run_directory),
            "Model"         : self.entry.model_name,
            "Device"        : self.device,
            "Split"         : self.split,
            "Batch size"    : self.batch_size,
            "Output"        : str(self.output_directory),
        }, title="Energy/Efficiency Sweep")

    def _build_predictor(self):
        model, _        = get_model(self.entry.model_name, **self.entry.model_overrides)
        checkpoint_path = self.run_directory / "checkpoints" / self.entry.training.io.checkpoint_name
        self.predictor  = Predictor(model, checkpoint_path, self.stats, self.device, self.logger).prepare()

    def _prepare_dataset_pipeline(self):
        self.dataset_pipeline = DatasetPipeline(self.entry.dataset, self.logger, stats=self.stats, evaluation_mode=True)
        self.dataset_pipeline.prepare_samples()

        self.grid = SweepGrid(self.config.scale, self.config.efficiency)

    def _evaluate_cell(self, scale_factor, detection_efficiency):
        self.entry.dataset.physics.scale_factor         = scale_factor
        self.entry.dataset.physics.detection_efficiency = detection_efficiency

        dataset              = self.dataset_pipeline.build_evaluation_split(self.split_base_ids, self.split)
        loader               = GraphDataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        predictions, targets = self.predictor.predict(loader)

        metrics = InferenceMetrics(self.logger).compute(predictions, targets, stage=f"scale={scale_factor:g} eff={detection_efficiency:g}")
        return {"scale_factor": float(scale_factor), "detection_efficiency": float(detection_efficiency), "metrics": metrics}

    def _run_grid(self):
        cells        = self.grid.cells()
        self.records = []
        for index, (scale_factor, detection_efficiency) in enumerate(cells, start=1):
            self.logger.section(f"[Sweep cell {index}/{len(cells)}] scale={scale_factor:g} efficiency={detection_efficiency:g}")
            self.records.append(self._evaluate_cell(scale_factor, detection_efficiency))

    def _metric_grids(self):
        rows, columns = self.grid.shape()
        grids         = {}
        for key in self.GRID_METRIC_KEYS:
            values     = np.array([record["metrics"][key] for record in self.records], dtype=np.float64)
            grids[key] = values.reshape(rows, columns)
        return grids

    def _axes_context(self):
        return [
            {"name": "scale_factor",         "minimum": self.grid.scale_axis.minimum,      "maximum": self.grid.scale_axis.maximum,      "step": self.grid.scale_axis.step,      "count": int(self.grid.shape()[1]), "values": self.grid.scale_values().tolist()},
            {"name": "detection_efficiency", "minimum": self.grid.efficiency_axis.minimum, "maximum": self.grid.efficiency_axis.maximum, "step": self.grid.efficiency_axis.step, "count": int(self.grid.shape()[0]), "values": self.grid.efficiency_values().tolist()},
        ]

    def _build_outputs(self):
        metric_grids = self._metric_grids()

        plots = SweepPlots(self.output_directory / "plots", self.logger).run(
            self.grid.scale_values(),
            self.grid.efficiency_values(),
            metric_grids,
            self.baseline,
            self.config.annotate,
        )

        context = {
            "run_directory"       : str(self.run_directory),
            "model_name"          : self.entry.model_name,
            "checkpoint"          : self.entry.training.io.checkpoint_name,
            "split"               : self.split,
            "split_event_count"   : len(self.split_base_ids[self._split_key()]),
            "baseline_scale"      : self.baseline[0],
            "baseline_efficiency" : self.baseline[1],
            "efficiency_seed"     : self.efficiency_seed,
            "grid_cells"          : len(self.records),
            "axes"                : self._axes_context(),
        }
        SweepReport(self.output_directory, self.logger).build(context, self.records, plots)

    def _split_key(self):
        return "validation" if self.split == "val" else self.split

    def run(self):
        self.logger.section("[Energy and Efficiency Sweep]")
        self._resolve_output_directory()
        self._load_run()
        self._build_predictor()
        self._prepare_dataset_pipeline()
        self._run_grid()
        self._build_outputs()
        self.logger.ok(f"Sweep complete: {len(self.records)} cells -> {self.output_directory}")
        return self.records
