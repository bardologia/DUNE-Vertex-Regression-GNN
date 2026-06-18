from _bootstrap import EnvironmentPinner

EnvironmentPinner.pin()

from configuration.entry            import InferenceEntryConfig
from tools.runtime.config_cli       import ConfigCli
from pipelines.inference.event_export import EventExportPipeline


class EventExportEntry:
    def __init__(self):
        self.config = InferenceEntryConfig()

    def run(self):
        self.config = ConfigCli(self.config, description="Export per-event predictions and active sensors for the event explorer").apply()

        return EventExportPipeline(
            self.config.run_directory,
            device     = self.config.device,
            splits     = self.config.splits,
            batch_size = self.config.batch_size,
        ).run()


if __name__ == "__main__":
    EventExportEntry().run()
