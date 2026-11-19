import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from configuration import DatasetConfig
from tools import Logger
from pipelines.dataset import ParquetDatasetWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data"


class LightDerivedGeometry:
    """Channel geometry of the channel-defect sweep (variant V1-drop), derived from the 2024 raw table and the light, without the documented module layout.

    The rule of the sweep report (section 2.2): negate y and z on the endcap
    rows of the raw table, as the June endcap correction did; then every even
    raw row k keeps the position of row k, every odd row k takes the position
    of row k + 2, and rows 1, 5759 and 6911, the face totals, are removed.

    Attributes:
        FACE_TOTAL_ROWS : Raw rows that carry face totals.
        ENDCAP_ROWS     : Number of raw rows on the endcap faces.
        kept_rows       : Raw rows kept as channels, in raw order, shape [6909].
        bins            : Raw bins of the kept rows as int64, shape [6909].
        positions       : Sensor-frame position of every kept row, shape [6909, 3], in metres.
    """

    FACE_TOTAL_ROWS = (1, 5759, 6911)
    ENDCAP_ROWS     = 1152

    def __init__(self, raw_frame: pd.DataFrame) -> None:
        """Derives the geometry from one raw layout.

        Args:
            raw_frame : Raw layout with the columns ``bin``, ``x``, ``y`` and ``z``, one row per raw bin in file order.
        """
        table      = raw_frame[["x", "y", "z"]].to_numpy(dtype=np.float64).copy()
        on_endcap  = np.isclose(np.abs(table[:, 2]), np.abs(table[:, 2]).max())
        assert int(on_endcap.sum()) == self.ENDCAP_ROWS
        table[on_endcap, 1:] *= -1.0

        rows    = np.arange(table.shape[0])
        sources = np.where(rows % 2 == 0, rows, rows + 2)
        kept    = rows[~np.isin(rows, self.FACE_TOTAL_ROWS)]

        self.kept_rows = kept
        self.bins      = raw_frame["bin"].to_numpy()[kept].astype(np.int64)
        self.positions = table[sources[kept]]


@pytest.fixture(scope="session")
def quiet_logger():
    return Logger(log_dir="", name="tests", level="ERROR")


@pytest.fixture()
def synthetic_batch():
    from torch_geometric.loader import DataLoader

    def make_graph(node_count=40, neighbors=4):
        node_features = torch.randn(node_count, 17)
        source        = torch.arange(node_count).repeat_interleave(neighbors)
        destination   = torch.randint(0, node_count, (node_count * neighbors,))
        edge_index    = torch.stack([source, destination])
        edge_attr     = torch.randn(node_count * neighbors, 23)
        graph_attr    = torch.randn(1, 33)
        target        = torch.randn(1, 3)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, graph_attr=graph_attr, y=target)

    return next(iter(DataLoader([make_graph() for _ in range(6)], batch_size=3)))


@pytest.fixture(scope="session")
def sample_csv_dir(tmp_path_factory):
    directory    = tmp_path_factory.mktemp("raw_csv")
    available = sorted(glob.glob(str(RAW_DATA_DIR / "*.csv")))
    if not available:
        pytest.skip("No raw CSV data available")

    stride       = max(1, len(available) // 48)
    source_files = available[::stride][:48]
    for source_file in source_files:
        shutil.copy(source_file, directory)
    return directory


@pytest.fixture(scope="session")
def parquet_store(tmp_path_factory, sample_csv_dir, quiet_logger):
    output_directory = tmp_path_factory.mktemp("store")
    ParquetDatasetWriter(sample_csv_dir, output_directory, worker_count=4, logger=quiet_logger).run()
    return output_directory / "parquet"


@pytest.fixture()
def raw_frame(sample_csv_dir):
    first_file = sorted(sample_csv_dir.glob("*.csv"))[0]
    return pd.read_csv(first_file)[["bin", "x", "y", "z"]]


@pytest.fixture()
def light_derived(raw_frame):
    return LightDerivedGeometry(raw_frame)


@pytest.fixture()
def dataset_config(parquet_store):
    config = DatasetConfig()
    config.data.parquet_store_dir = parquet_store
    config.data.stats_sample_size = 32
    return config
