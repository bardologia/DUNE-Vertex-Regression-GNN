import cProfile
import pstats
import sys
from pathlib import Path
import os
import warnings
import traceback
import ray

os.environ['LOKY_MAX_CPU_COUNT'] = '8'

warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import DatasetBuilder, DatasetLoader
from core.config import config
from core.logger import Logger
from core.runtime import RuntimeGuard

def main(config):
    dataset_dir = (f"./datasets/"f"E_{config.preprocessing.scale_factor * 100}MeV_K_{config.graph.k_neighbors}/")
    logger = Logger(log_dir=dataset_dir, config=config, name="dataset_creation")
    logger.section("[Dataset Creation]")
    RuntimeGuard.cleanup_stale_ray_processes(logger)

    logger.section("[Profiler]")
    logger.info("Profiling enabled")
    profiler = cProfile.Profile()
    profiler.enable()

    exit_code = 0
    try:
        builder = DatasetBuilder(dataset_dir, logger, config=config)
        builder.run()

        logger.section("[Dataset Test Loading]")
        loader = DatasetLoader(dataset_dir, logger, config=config)
        loader.run()

        logger.info("End of dataset creation.")
    except KeyboardInterrupt:
        exit_code = 130
        logger.warning("Dataset creation interrupted by user (KeyboardInterrupt).")
    except Exception as exception:
        exit_code = 1
        logger.error(f"Dataset creation failed: {exception}")
        logger.error(traceback.format_exc())
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        logger.save_profiler_results(stats, output=f"{dataset_dir}/profiler_results.md")
        logger.info(f"Profiler results saved to {dataset_dir}/profiler_results.md")

        if ray.is_initialized():
            ray.shutdown()

        if exit_code != 0:
            raise SystemExit(exit_code)

if __name__ == "__main__":
    main(config)
