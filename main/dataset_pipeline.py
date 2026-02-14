import sys
from pathlib import Path
import os
import warnings

os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'
os.environ['LOKY_MAX_CPU_COUNT'] = '8'

warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

sys.path.insert(0, str(Path.cwd()))

from core.dataset import DatasetBuilder, DatasetLoader
from core.config import config
import numpy as np
import ray # type: ignore


def main():
    dataset_dir = (
        f"./datasets/"
        f"E_{config.preprocessing.scale_factor * 100}MeV_K_{config.graph.k_neighbors}/"
    )
    
    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=8,
        _metrics_export_port=None,
        object_store_memory=8*1024**3,
    )
    
    print("Building dataset...")
    builder = DatasetBuilder(dataset_dir)
    builder.run()
    
    print("\nLoading and processing dataset...")
    loader = DatasetLoader(preprocessed_dir=dataset_dir)

    graphs, target_df, config_dict = loader._load_artifacts()
    data_config = config_dict["data_config"]

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

    print(f"\nDataset Sizes:")
    print(f"  Training:   {len(train_dataset)} graphs")
    print(f"  Validation: {len(val_dataset)} graphs")
    print(f"  Test:       {len(test_dataset)} graphs")
    print(f"  Total:      {len(train_dataset) + len(val_dataset) + len(test_dataset)} graphs")
    
    def analyze_dataset(dataset, name):
        num_nodes = [graph.x.shape[0] for graph in dataset]
        num_edges = [graph.edge_index.shape[1] for graph in dataset]
        
        print(f"\n{name}:")
        print(f"  Nodes:  min={min(num_nodes)}, max={max(num_nodes)}, mean={np.mean(num_nodes):.2f}")
        print(f"  Edges:  min={min(num_edges)}, max={max(num_edges)}, mean={np.mean(num_edges):.2f}")
        print(f"  Samples: {len(dataset)}")

    analyze_dataset(train_dataset, "Training Set")
    analyze_dataset(val_dataset, "Validation Set")
    analyze_dataset(test_dataset, "Test Set")

    sample_graph = train_dataset[0]

    print(f"\nSample Graph (first from training set):")
    print(f"  Node features shape: {sample_graph.x.shape}")
    print(f"  Edge index shape: {sample_graph.edge_index.shape}")
    print(f"  Target (y) shape: {sample_graph.y.shape}")
    print(f"\n  First 5 nodes (features):")
    print(sample_graph.x[:5])
    print(f"\n  Target coordinates (graph-level):")
    print(sample_graph.y)

    batch_stats = DatasetLoader.estimate_batch_size(
        graphs=train_dataset,
        available_vram_gb=4,
        safety_factor=0.6
    )

    print("\nBatch Size Estimation:")
    print(f"  Average graph size: {batch_stats['avg_graph_size_mb']:.2f} MB")
    print(f"  Average nodes per graph: {batch_stats['avg_nodes']:.0f}")
    print(f"  Average edges per graph: {batch_stats['avg_edges']:.0f}")
    print(f"  Available VRAM: {batch_stats['available_vram_gb']} GB")
    print(f"  Usable VRAM ({batch_stats['safety_factor']*100:.0f}%): {batch_stats['usable_vram_mb']:.0f} MB")
    print(f"  Estimated batch size: {batch_stats['estimated_batch_size']}")
    print(f"  Memory per batch: {batch_stats['memory_per_batch_mb']:.2f} MB")
    
    if ray.is_initialized():
        ray.shutdown()
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = main()
