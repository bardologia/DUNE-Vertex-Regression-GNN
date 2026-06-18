from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry      import DatasetEntryConfig
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from pipelines.dataset        import DatasetAugmentor


class AugmentationEntry:
    def __init__(self):
        self.config = DatasetEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Reflect raw CSV events into the eight octants").apply()

        logger = Logger(log_dir="logs", name="octant_augmentation")
        DatasetAugmentor(self.config.raw_input_dir, self.config.output_dir, worker_count=self.config.worker_count, logger=logger).run()
        logger.close()


if __name__ == "__main__":
    AugmentationEntry().run()
