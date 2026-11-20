from __future__ import annotations

from pathlib import Path

from configuration.shared import IoNames


class RunPaths:
    def __init__(self, base_logdir, model_name, run_name):
        self.run_directory      = Path(base_logdir) / IoNames.run_dir.format(model=model_name, run=run_name)
        self.tensorboard_dir    = self.run_directory / IoNames.tensorboard_dir
        self.logs_directory     = self.run_directory / IoNames.logs_dir
        self.metadata_directory = self.run_directory / IoNames.metadata_dir
        self.checkpoint_dir     = self.run_directory / IoNames.checkpoints_dir

    def create(self, allow_existing: bool = False) -> None:
        if not allow_existing and self.run_directory.exists() and any(self.run_directory.iterdir()):
            raise FileExistsError(f"Run directory {self.run_directory} already exists and is not empty; remove it or choose a different run_name so metrics, checkpoint, and logs cannot be mixed across separate runs.")

        for directory in (self.tensorboard_dir, self.logs_directory, self.metadata_directory, self.checkpoint_dir):
            directory.mkdir(parents=True, exist_ok=True)
