import cProfile
import pstats
import sys
from pathlib import Path
import os
import warnings

os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'
os.environ['LOKY_MAX_CPU_COUNT'] = '8'
os.environ['RAY_ENABLE_METRICS'] = '0'

warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

sys.path.insert(0, str(Path.cwd()))

from core.dataset import DatasetBuilder, DatasetLoader
from core.config import config
from core.logger import Logger
import ray 

def main(config):
    dataset_dir = (f"./datasets/"f"E_{config.preprocessing.scale_factor * 100}MeV_K_{config.graph.k_neighbors}/")
    logger = Logger(log_dir=dataset_dir, config=config, name="dataset_creation")
    logger.section("[Dataset Creation]")

    logger.section("[Profiler]")
    logger.info("Profiling enabled")
    profiler = cProfile.Profile()
    profiler.enable()

    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=4,
        _metrics_export_port=None,
        object_store_memory=8*1024**3,
    )
    
    builder = DatasetBuilder(dataset_dir, logger)
    builder.run()
    
    logger.section("[Dataset Test Loading]")
    loader = DatasetLoader(dataset_dir, logger)
    norm_split, norm_metrics, saved_config = loader.run()

    logger.info("End of dataset creation.")
    if ray.is_initialized():
        ray.shutdown()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    logger.save_profiler_results(stats, output=f"{dataset_dir}/profiler_results.md")
    logger.info(f"Profiler results saved to {dataset_dir}/profiler_results.md")

if __name__ == "__main__":
    main(config)
