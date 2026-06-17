from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from pipelines.dataset.normalization import Normalizer
from pipelines.dataset.splitting     import TargetBalancer


class DatasetPipeline:
    def __init__(self, preprocessed_dir, logger, split_config, subset_fraction: float = 1.0):
        self.logger             = logger
        self.preprocessed_dir   = Path(preprocessed_dir)
        self.split_config       = split_config
        self.subset_fraction    = subset_fraction
        self.coordinate_columns = None

    def _load_artifacts(self):
        self.logger.section("[Loading Artifacts]")

        with open(self.preprocessed_dir / "dataset_metadata.pkl", "rb") as handle:
            metadata = pickle.load(handle)

        graphs = []
        with self.logger.track() as progress:
            task_id = progress.add_task("Loading chunks", total=metadata["num_chunks"])
            for chunk_index in range(metadata["num_chunks"]):
                chunk = torch.load(self.preprocessed_dir / "chunks" / f"dataset_chunk_{chunk_index}.pt", weights_only=False)
                graphs.extend(chunk)
                progress.advance(task_id)

        with open(self.preprocessed_dir / "target.pkl", "rb") as handle:
            target_dataframe = pickle.load(handle)

        with open(self.preprocessed_dir / "config.pkl", "rb") as handle:
            saved_config = pickle.load(handle)

        self.coordinate_columns = saved_config["data_config"].coordinate_columns

        if 0.0 < self.subset_fraction < 1.0:
            number_of_graphs = max(1, int(len(graphs) * self.subset_fraction))
            random_generator = np.random.RandomState(self.split_config.random_state)
            indices          = np.sort(random_generator.choice(len(graphs), size=number_of_graphs, replace=False))
            graphs           = [graphs[index] for index in indices]
            target_dataframe = target_dataframe.iloc[indices].reset_index(drop=True)
            self.logger.subsection(f"Subset fraction {self.subset_fraction}: using {number_of_graphs} graphs")

        self.logger.subsection(f"Loaded {len(graphs)} graphs from {self.preprocessed_dir}")
        return graphs, target_dataframe, saved_config

    def _split_and_attach(self, graphs, target_dataframe):
        balancer = TargetBalancer(self.logger, self.coordinate_columns, self.split_config)
        train_indices, validation_indices, test_indices, targets = balancer.balance(target_dataframe)

        def attach(selected_indices):
            selected_graphs = []
            for index in selected_indices:
                graph   = graphs[index]
                graph.y = torch.tensor(targets[index], dtype=torch.float32).unsqueeze(0)
                selected_graphs.append(graph)
            return selected_graphs

        return {"train": attach(train_indices), "val": attach(validation_indices), "test": attach(test_indices)}

    def _normalize(self, splits):
        normalizer     = Normalizer(self.logger, source_split="train")
        splits, stats  = normalizer.run(splits)
        return splits, stats

    def run(self):
        self.logger.section("[Dataset Pipeline]")
        graphs, target_dataframe, saved_config = self._load_artifacts()
        splits                                 = self._split_and_attach(graphs, target_dataframe)
        splits, stats                          = self._normalize(splits)
        return splits, stats, saved_config

    @staticmethod
    def build_loaders(splits, batch_size, num_workers=0, pin_memory=False, persistent_workers=False):
        persistent = persistent_workers and num_workers > 0
        train_loader = GraphDataLoader(splits["train"], batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
        val_loader   = GraphDataLoader(splits["val"],   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
        test_loader  = GraphDataLoader(splits["test"],  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
        return train_loader, val_loader, test_loader
