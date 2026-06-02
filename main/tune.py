import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import DatasetLoader
from core.config import config
from core.logger import Logger
from core.runtime import RuntimeGuard
from core.tuner import Tuner

def main(dataset_path, config):
    logger = Logger(log_dir=config.io.logdir, config=config, name="tuning", level="INFO")
    RuntimeGuard.cleanup_stale_ray_processes(logger)
    loader = DatasetLoader(preprocessed_dir=dataset_path, logger=logger, config=config)

    norm_split, norm_metrics, data_config = loader.run()
    
    train_dataset = norm_split["train"]
    val_dataset   = norm_split["val"]
    test_dataset  = norm_split["test"]

    tuner = Tuner(norm_metrics=norm_metrics, dataset_config=data_config, config=config, logger=logger)
    tuner.optimize(train_dataset=train_dataset, val_dataset=val_dataset, n_trials=config.tuning.n_trials)
    best_configs = tuner.build_best_configs(epochs=config.tuning.epochs)

    return best_configs

if __name__ == "__main__":
    dataset_path = "C:\\Users\\victo\\Desktop\\dune-gnn\\datasets\\E_10.0MeV_K_4"
    best_configs = main(dataset_path=dataset_path, config=config)