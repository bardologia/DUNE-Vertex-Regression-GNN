from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry                       import DatasetExplorerEntryConfig
from tools.monitoring.logger                   import Logger
from tools.runtime.config_cli                  import ConfigCli
from pipelines.inference.dataset_explorer_export import DatasetExplorerExportPipeline


class DatasetExplorerExportEntry:
    def __init__(self):
        self.config = DatasetExplorerEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Export raw or octant-augmented events for the event explorer").apply()

        logger = Logger(log_dir="logs", name="dataset_explorer")
        path   = DatasetExplorerExportPipeline(self.config.kind, self.config.cache_dir, logger=logger, dataset_config=self.config.dataset).run()
        logger.close()
        return path


if __name__ == "__main__":
    DatasetExplorerExportEntry().run()
