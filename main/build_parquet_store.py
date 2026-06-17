import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import Logger
from core.parquet_store import ParquetDatasetWriter


class ParquetStoreEntryPoint:

    def __init__(self):
        self.input_directory  = PROJECT_ROOT / "data"
        self.output_directory = PROJECT_ROOT / "data_frames"
        self.worker_count     = 10
        self.logger           = Logger(log_dir=str(PROJECT_ROOT / "logs"), name="parquet_store")

    def run(self):
        writer = ParquetDatasetWriter(
            input_directory  = self.input_directory,
            output_directory = self.output_directory,
            worker_count     = self.worker_count,
            logger           = self.logger,
        )
        writer.run()
        self.logger.close()


if __name__ == "__main__":
    ParquetStoreEntryPoint().run()
