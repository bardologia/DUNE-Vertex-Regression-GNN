from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.sweep      import EnergyEfficiencySweepConfig
from tools.runtime.config_cli import ConfigCli
from pipelines.sweep          import EnergyEfficiencySweepPipeline


class SweepEntry:
    def __init__(self):
        self.config = EnergyEfficiencySweepConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Sweep light scale factor and binomial detection efficiency over a trained run's evaluation split").apply()

        return EnergyEfficiencySweepPipeline(self.config).run()


if __name__ == "__main__":
    SweepEntry().run()
