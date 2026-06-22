from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.explainability import FeatureImportanceConfig
from tools.runtime.config_cli     import ConfigCli
from pipelines.explainability     import FeatureImportancePipeline


class ExplainEntry:
    def __init__(self):
        self.config = FeatureImportanceConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Attribute the graph node and edge features of a trained DUNE-GNN run via permutation, occlusion, and gradient saliency").apply()

        return FeatureImportancePipeline(self.config).run()


if __name__ == "__main__":
    ExplainEntry().run()
