import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.monitoring.logger import Logger
from pipelines.dataset import DatasetAugmentor


class AugmentationEntryPoint:

    def __init__(self):
        self.input_directory  = PROJECT_ROOT / "data"
        self.output_directory = PROJECT_ROOT / "data_frames"
        self.worker_count     = 10
        self.logger           = Logger(log_dir=str(PROJECT_ROOT / "logs"), name="octant_augmentation")

    def run(self):
        augmentor = DatasetAugmentor(
            input_directory  = self.input_directory,
            output_directory = self.output_directory,
            worker_count     = self.worker_count,
            logger           = self.logger,
        )
        augmentor.run()
        self.logger.close()


if __name__ == "__main__":
    AugmentationEntryPoint().run()
