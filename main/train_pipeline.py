import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader  # type: ignore
import cProfile
import pstats

sys.path.insert(0, str(Path.cwd()))

from core.dataset import DatasetLoader
from core.trainer import GNNTrainer
from core.config import config
from core.model import Model

from tqdm import tqdm

plt.style.use('seaborn-v0_8-darkgrid')

# Tensorboard setup (uncomment if needed)
# from torch.utils.tensorboard import SummaryWriter
# tensorboard --logdir {config.training.log_dir} --port 6014 --reload_interval 30


def main():
    dataset_path = f"./datasets/E_{config.preprocessing.scale_factor * 100}MeV_K_{config.graph.k_neighbors}/"
    
    loader = DatasetLoader(preprocessed_dir=dataset_path)
    
    graphs, target_df, saved_config = loader._load_artifacts()
    data_config = saved_config["data_config"]
    
    train_idx, val_idx, test_idx, Y, N = loader._balance_targets(target_df)
    idx = (train_idx, val_idx, test_idx)
    
    train_balanced_graphs = [graphs[i] for i in train_idx]
    val_balanced_graphs   = [graphs[i] for i in val_idx]
    test_balanced_graphs  = [graphs[i] for i in test_idx]
    
    balanced_graphs = (train_balanced_graphs, val_balanced_graphs, test_balanced_graphs)
    
    raw_train_dataset, raw_val_dataset, raw_test_dataset = loader._attach_targets(balanced_graphs, idx, Y)
    raw_split = {"train": raw_train_dataset, "val": raw_val_dataset, "test": raw_test_dataset}
    norm_split, norm_metrics = loader._normalize(raw_split, data_config)
    
    train_dataset = norm_split["train"]
    val_dataset   = norm_split["val"]
    test_dataset  = norm_split["test"]
    
    model = Model()
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
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Train model
    print("\nStarting training...")
    trainer = GNNTrainer(
        model,
        norm_metrics=norm_metrics,
        model_configs=config.model,
        dataset_config_dict={
            'data_config': config.data,
            'graph_config': config.graph,
            'preprocessing_config': config.preprocessing,
        }
    )
    
    train_losses, val_losses, test_results = trainer.train(
        train_loader,
        val_loader,
        test_loader=test_loader,
    )
    
    profiler.disable()
    
    print("\nProfiling results:")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    return train_losses, val_losses, test_results

if __name__ == "__main__":
    train_losses, val_losses, test_results = main()
