from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry      import BaselineEntryConfig
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from pipelines.baseline       import BaselinePipeline


class BaselineEntry:
    def __init__(self):
        self.config = BaselineEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Tune and train the gradient-boosted baselines (LightGBM, XGBoost)").apply()

        logger  = Logger(log_dir="logs", name="baseline")
        summary = BaselinePipeline(self.config, logger).run()
        logger.close()
        return summary


if __name__ == "__main__":
    BaselineEntry().run()
