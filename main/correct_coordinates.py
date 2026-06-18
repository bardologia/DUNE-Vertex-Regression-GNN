from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry      import DatasetEntryConfig
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from pipelines.dataset        import DatasetCorrector


class CorrectionEntry:
    def __init__(self):
        self.config = DatasetEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Correct raw CSV coordinates into the sensor frame").apply()

        logger = Logger(log_dir="logs", name="coordinate_correction")
        DatasetCorrector(self.config.raw_input_dir, self.config.output_dir, logger=logger).run()
        logger.close()


if __name__ == "__main__":
    CorrectionEntry().run()
