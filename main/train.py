from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry      import TrainEntryConfig
from tools.runtime.config_cli import ConfigCli
from pipelines.training       import TrainingPipeline


class TrainEntry:
    def __init__(self):
        self.config = TrainEntryConfig()

    def run(self):
        cli         = ConfigCli(self.config, description="Train a DUNE-GNN model")
        self.config = cli.apply()
        return TrainingPipeline(self.config, explicit_paths=set(cli.overrides)).run()


if __name__ == "__main__":
    TrainEntry().run()
