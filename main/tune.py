from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry      import TuneEntryConfig
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from pipelines.tuning         import Tuner


class TuneEntry:
    def __init__(self):
        self.config = TuneEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Tune a DUNE-GNN model with Optuna").apply()

        logger = Logger(log_dir="logs", name=f"tune_{self.config.model_name}")
        tuner  = Tuner(self.config, logger)
        study  = tuner.optimize()
        logger.close()
        return study


if __name__ == "__main__":
    TuneEntry().run()
