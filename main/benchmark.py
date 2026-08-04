from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.benchmark    import BenchmarkConfig
from tools.runtime.config_cli   import ConfigCli
from pipelines.benchmark        import BenchmarkPipeline


class BenchmarkEntry:
    def __init__(self):
        self.config = BenchmarkConfig()

    def run(self):
        cli         = ConfigCli(self.config, description="Benchmark the DUNE-GNN model zoo")
        self.config = cli.apply()
        return BenchmarkPipeline(self.config, explicit_paths=set(cli.overrides)).run()


if __name__ == "__main__":
    BenchmarkEntry().run()
