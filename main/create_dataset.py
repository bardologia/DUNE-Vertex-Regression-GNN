from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from pathlib import Path

from configuration.entry      import DatasetEntryConfig
from tools.monitoring.logger  import Logger
from tools.runtime.config_cli import ConfigCli
from pipelines.dataset        import DatasetBuilder


class CreateDatasetEntry:
    def __init__(self):
        self.config = DatasetEntryConfig()

    def _output_directory(self) -> Path:
        scale_factor = self.config.dataset.preprocessing.scale_factor
        k_neighbors  = self.config.dataset.graph.k_neighbors
        return Path(self.config.output_base_dir) / f"E_{scale_factor * 100:.1f}MeV_K_{k_neighbors}"

    def run(self):
        self.config = ConfigCli(self.config, description="Build the preprocessed graph dataset").apply()

        logger           = Logger(log_dir="logs", name="create_dataset")
        output_directory = self._output_directory()
        DatasetBuilder(output_directory, logger, self.config.dataset).run()
        logger.close()
        return output_directory


if __name__ == "__main__":
    CreateDatasetEntry().run()
