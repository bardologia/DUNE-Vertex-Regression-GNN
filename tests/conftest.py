import glob
import shutil
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from configuration import DatasetConfig, SplitConfig
from tools import Logger
from pipelines.dataset import DatasetBuilder, ParquetDatasetWriter


PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RAW_DATA_DIR  = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def quiet_logger():
    return Logger(log_dir="", name="tests", level="ERROR")


@pytest.fixture()
def synthetic_batch():
    from torch_geometric.loader import DataLoader

    def make_graph(node_count=40, neighbors=4):
        node_features = torch.randn(node_count, 8)
        source        = torch.arange(node_count).repeat_interleave(neighbors)
        destination   = torch.randint(0, node_count, (node_count * neighbors,))
        edge_index    = torch.stack([source, destination])
        edge_attr     = torch.randn(node_count * neighbors, 7)
        target        = torch.randn(1, 3)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=target)

    return next(iter(DataLoader([make_graph() for _ in range(6)], batch_size=3)))


@pytest.fixture(scope="session")
def sample_csv_dir(tmp_path_factory):
    directory   = tmp_path_factory.mktemp("raw_csv")
    source_files = sorted(glob.glob(str(RAW_DATA_DIR / "*.csv")))[:48]
    if not source_files:
        pytest.skip("No raw CSV data available")
    for source_file in source_files:
        shutil.copy(source_file, directory)
    return directory


@pytest.fixture(scope="session")
def parquet_store(tmp_path_factory, sample_csv_dir, quiet_logger):
    output_directory = tmp_path_factory.mktemp("store")
    ParquetDatasetWriter(sample_csv_dir, output_directory, worker_count=4, logger=quiet_logger).run()
    return output_directory / "parquet"


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory, parquet_store, quiet_logger):
    config = DatasetConfig()
    config.data.source            = "parquet"
    config.data.parquet_store_dir = parquet_store
    config.ray.num_cpus           = 4

    directory = tmp_path_factory.mktemp("dataset")
    DatasetBuilder(directory, quiet_logger, config).run()
    return directory


@pytest.fixture()
def split_config():
    return SplitConfig()
