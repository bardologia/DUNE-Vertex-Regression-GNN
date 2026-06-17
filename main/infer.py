from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry      import InferenceEntryConfig
from tools.runtime.config_cli import ConfigCli
from pipelines.inference      import InferencePipeline


class InferenceEntry:
    def __init__(self):
        self.config = InferenceEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Run full inference analysis on a training run").apply()

        return InferencePipeline(
            self.config.run_directory,
            device     = self.config.device,
            splits     = self.config.splits,
            batch_size = self.config.batch_size,
        ).run()


if __name__ == "__main__":
    InferenceEntry().run()
