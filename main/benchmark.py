from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.benchmark    import BenchmarkConfig
from tools.runtime.config_cli   import ConfigCli
from pipelines.benchmark        import BenchmarkPipeline


class BenchmarkEntry:
    def __init__(self):
        self.config = BenchmarkConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Benchmark the DUNE-GNN model zoo").apply()
        return BenchmarkPipeline(self.config).run()


if __name__ == "__main__":
    BenchmarkEntry().run()
