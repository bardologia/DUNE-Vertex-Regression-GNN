from __future__ import annotations

from pathlib import Path


class RunPaths:
    def __init__(self, base_logdir, model_name, run_name):
        self.run_directory      = Path(base_logdir) / f"{model_name}_{run_name}"
        self.tensorboard_dir    = self.run_directory / "tensorboard"
        self.logs_directory     = self.run_directory / "logs"
        self.metadata_directory = self.run_directory / "metadata"
        self.checkpoint_dir     = self.run_directory / "checkpoints"

    def create(self) -> None:
        for directory in (self.tensorboard_dir, self.logs_directory, self.metadata_directory, self.checkpoint_dir):
            directory.mkdir(parents=True, exist_ok=True)
