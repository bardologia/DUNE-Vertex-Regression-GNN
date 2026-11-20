from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

from torch.utils.tensorboard import SummaryWriter

from configuration.shared     import IoNames
from tools.monitoring.logger  import Logger
from tools.monitoring.tracker import NullTracker, Tracker
from tools.runtime.config_cli import ConfigCli

from pipelines.shared.run_paths      import RunPaths
from pipelines.shared.store_geometry import StoreGeometryRecord


class LightweightRunContext:
    def __init__(self, logger, checkpoint_path, tracker=None):
        self.logger             = logger
        self.tracker            = tracker if tracker is not None else NullTracker()
        self.checkpoint_path    = Path(checkpoint_path)
        self.metadata_directory = self.checkpoint_path.parent
        self.run_directory      = self.checkpoint_path.parent

        self.metadata_directory.mkdir(parents=True, exist_ok=True)


class TrainingRunMetadata:
    def __init__(self, training_config, model_name, base_logdir, run_name=None, logger=None):
        self.training_config = training_config
        self.model_name      = model_name

        stamp         = datetime.now().strftime(IoNames.run_stamp_format)
        resolved_name = run_name if run_name else stamp
        run_paths     = RunPaths(base_logdir, model_name, resolved_name)

        self.run_directory      = run_paths.run_directory
        self.tensorboard_dir    = run_paths.tensorboard_dir
        self.logs_directory     = run_paths.logs_directory
        self.metadata_directory = run_paths.metadata_directory
        self.checkpoint_dir     = run_paths.checkpoint_dir
        self.resuming           = self.run_directory.exists() and any(self.run_directory.iterdir())

        if self.resuming and training_config.loop.resume:
            StoreGeometryRecord(self.metadata_directory).recorded()
        run_paths.create(allow_existing=training_config.loop.resume)

        self.checkpoint_path = self.checkpoint_dir / training_config.io.checkpoint_name
        self.writer          = SummaryWriter(log_dir=str(self.tensorboard_dir))
        self.logger          = logger if logger is not None else Logger(log_dir=str(self.logs_directory), name=model_name)
        self.tracker         = Tracker(writer=self.writer, debug=training_config.loop.log_debug)

        self.logger.section("[Run Setup]")
        self.logger.kv_table({
            "Model"         : model_name,
            "Run directory" : str(self.run_directory),
            "Logs"          : str(self.logs_directory),
            "Tensorboard"   : str(self.tensorboard_dir),
            "Checkpoint"    : str(self.checkpoint_path),
        })

    def save_store_geometry(self, store_geometry: dict) -> Path:
        record = StoreGeometryRecord(self.metadata_directory)
        if self.resuming:
            record.require(store_geometry)
        return record.save(store_geometry)

    def save_resolved_config(self, config) -> Path:
        return ConfigCli.save_resolved(config, self.metadata_directory / IoNames.resolved_config_file)

    def save_normalization_stats(self, stats) -> Path:
        return stats.save(self.metadata_directory)

    def save_split(self, split_base_ids) -> Path:
        path = self.metadata_directory / IoNames.split_base_ids_file
        path.write_text(json.dumps(split_base_ids, indent=IoNames.json_indent), encoding="utf-8")
        return path

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
