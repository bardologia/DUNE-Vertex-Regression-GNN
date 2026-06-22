from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry                  import RawLightExportEntryConfig
from tools.monitoring.logger              import Logger
from tools.runtime.config_cli             import ConfigCli
from pipelines.inference.raw_light_export import RawLightExportPipeline


class RawLightExportEntry:
    def __init__(self):
        self.config = RawLightExportEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Export uncorrected raw photon counts for the preprocessing preview").apply()

        logger = Logger(log_dir="logs", name="raw_light")
        path   = RawLightExportPipeline(self.config.raw_input_dir, self.config.cache_dir, logger=logger, dataset_config=self.config.dataset).run()
        logger.close()
        return path


if __name__ == "__main__":
    RawLightExportEntry().run()
