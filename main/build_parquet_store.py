from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry      import DatasetEntryConfig
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from pipelines.dataset        import ParquetDatasetWriter


class ParquetStoreEntry:
    def __init__(self):
        self.config = DatasetEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Build the Parquet store from raw CSVs").apply()

        logger = Logger(log_dir="logs", name="parquet_store")
        ParquetDatasetWriter(self.config.raw_input_dir, self.config.output_dir, worker_count=self.config.worker_count, logger=logger, hot_channels=self.config.hot_channels).run()
        logger.close()


if __name__ == "__main__":
    ParquetStoreEntry().run()
