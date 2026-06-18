from __future__ import annotations

import json
from datetime import datetime
from pathlib  import Path

from torch.utils.tensorboard import SummaryWriter

from tools.monitoring.logger  import Logger
from tools.monitoring.tracker import NullTracker, Tracker
from tools.runtime.config_cli import ConfigCli


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

        stamp                  = datetime.now().strftime("%Y%m%d-%H%M%S")
        resolved_name          = run_name if run_name else stamp
        self.run_directory     = Path(base_logdir) / f"{model_name}_{resolved_name}"
        self.tensorboard_dir   = self.run_directory / "tensorboard"
        self.logs_directory    = self.run_directory / "logs"
        self.metadata_directory = self.run_directory / "metadata"
        self.checkpoint_dir    = self.run_directory / "checkpoints"

        for directory in (self.tensorboard_dir, self.logs_directory, self.metadata_directory, self.checkpoint_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.checkpoint_dir / training_config.io.checkpoint_name
        self.writer          = SummaryWriter(log_dir=str(self.tensorboard_dir))
        self.logger          = logger if logger is not None else Logger(log_dir=str(self.logs_directory), name=model_name)
        self.tracker         = Tracker(writer=self.writer)

    def save_resolved_config(self, config) -> Path:
        return ConfigCli.save_resolved(config, self.metadata_directory / "resolved_config.json")

    def save_normalization_stats(self, stats) -> Path:
        return stats.save(self.metadata_directory)

    def save_split(self, split_base_ids) -> Path:
        path = self.metadata_directory / "split_base_ids.json"
        path.write_text(json.dumps(split_base_ids, indent=2), encoding="utf-8")
        return path

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
