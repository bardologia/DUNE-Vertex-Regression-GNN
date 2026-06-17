import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.logger import Logger
from core.coordinate_correction import DatasetCorrector


class CorrectionEntryPoint:

    def __init__(self):
        self.input_directory  = PROJECT_ROOT / "data"
        self.output_directory = PROJECT_ROOT / "data_frames"
        self.logger           = Logger(log_dir=str(PROJECT_ROOT / "logs"), name="coordinate_correction")

    def run(self):
        corrector = DatasetCorrector(
            input_directory  = self.input_directory,
            output_directory = self.output_directory,
            logger           = self.logger,
        )
        corrector.run()
        self.logger.close()


if __name__ == "__main__":
    CorrectionEntryPoint().run()
