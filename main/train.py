import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader  # type: ignore
import cProfile
import pstats
import argparse
import subprocess
import webbrowser
import atexit

sys.path.insert(0, str(Path.cwd()))

from core.dataset import DatasetLoader
from core.trainer import Trainer
from core.config import config
from core.model import Model
from core.logger import Logger

tensorboard_process = None

def start_tensorboard(logdir, port=6006):
    global tensorboard_process
    tensorboard_process = subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(Path(logdir) / "tensorboard"), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    webbrowser.open(f"http://localhost:{port}")
    return tensorboard_process

def stop_tensorboard():
    global tensorboard_process
    if tensorboard_process:
        tensorboard_process.terminate()

atexit.register(stop_tensorboard)

def main(dataset_path, config):
    logger = Logger(log_dir=config.io.logdir, config=config, name="training")
    loader = DatasetLoader(preprocessed_dir=dataset_path, logger=logger, config=config)

    norm_split, norm_metrics, data_config = loader.run()
    
    train_dataset = norm_split["train"]
    val_dataset   = norm_split["val"]
    test_dataset  = norm_split["test"]
    
    model = Model(config)
    batch_size = config.training.batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=config.training.pin_memory,
        num_workers=config.training.num_workers,
        persistent_workers=True if config.training.num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=config.training.pin_memory,
        num_workers=config.training.num_workers,
        persistent_workers=True if config.training.num_workers > 0 else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=config.training.pin_memory,
        num_workers=config.training.num_workers,
        persistent_workers=True if config.training.num_workers > 0 else False,
    )
    
    logger.section("[Profiler]")
    if config.training.profile:
        logger.info("Profiling enabled")
        profiler = cProfile.Profile()
        profiler.enable()
    else:
        logger.info("Profiling disabled")
    
    trainer = Trainer(
        model,
        norm           = norm_metrics,
        config         = config,
        dataset_config = data_config,
        run_dir        = config.io.logdir,
        logger         = logger,
    )
    
    train_results, validation_results, test_results = trainer.train(
        train_loader,
        val_loader,
        test_loader=test_loader,
    )
    
    if config.training.profile: 
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        logger.save_profiler_results(stats, output=f"{config.io.logdir}/profiler_results.md")
        logger.info(f"Profiler results saved to {config.io.logdir}/profiler_results.md")

    return train_results, validation_results, test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="./datasets/E_10.0MeV_K_4/")
    parser.add_argument("--tensorboard", "-tb", action="store_true")
    parser.add_argument("--port", "-p", type=int, default=6006)
    args = parser.parse_args()
    
    if args.tensorboard:
        start_tensorboard(config.io.logdir, args.port)
    
    train_results, validation_results, test_results = main(args.dataset, config)
