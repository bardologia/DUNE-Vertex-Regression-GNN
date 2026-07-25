from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from pathlib import Path

from configuration.entry             import TrainEntryConfig
from tools.runtime.config_cli        import ConfigCli
from pipelines.shared                import PretrainPreflight
from pipelines.training              import TrainingPipeline


class TrainEntry:
    def __init__(self):
        self.config = TrainEntryConfig()

    def _pipeline(self, explicit_paths):
        return TrainingPipeline(self.config, explicit_paths=explicit_paths)

    def _preflight(self, explicit_paths):
        run_directory = Path(self.config.training.io.log_base_dir) / "pretrain" / self.config.model_name

        PretrainPreflight(
            pretrain_config = self.config.training.pretrain,
            loop_config     = self.config.training.loop,
            build_trainer   = lambda logger: self._pipeline(explicit_paths).build_pretrain_trainer(run_directory / "probe", logger),
            run_directory   = run_directory,
            label           = self.config.model_name,
        ).run()

    def run(self):
        cli            = ConfigCli(self.config, description="Train a DUNE-GNN model")
        self.config    = cli.apply()
        explicit_paths = set(cli.overrides)

        self._preflight(explicit_paths)

        return self._pipeline(explicit_paths).run()


if __name__ == "__main__":
    TrainEntry().run()
