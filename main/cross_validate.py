from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry        import CrossValidationEntryConfig
from tools.runtime.config_cli   import ConfigCli
from pipelines.cross_validation import CrossValidationPipeline


class CrossValidateEntry:
    def __init__(self):
        self.config = CrossValidationEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Cross-validate a DUNE-GNN model").apply()
        return CrossValidationPipeline(self.config).run()


if __name__ == "__main__":
    CrossValidateEntry().run()
