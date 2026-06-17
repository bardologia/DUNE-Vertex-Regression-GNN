import torch

from pipelines.dataset import DatasetPipeline, Graph


def test_graph_builder_shapes():
    import pandas as pd

    class GraphConfig:
        bidirectional = False
        k_neighbors   = 4
        knn_algorithm = "kd_tree"

    class DataConfig:
        light_column     = "light_amount"
        position_columns = ("x", "y", "z")

    class Config:
        graph = GraphConfig()
        data  = DataConfig()

    frame = pd.DataFrame({
        "bin"          : range(30),
        "x"            : torch.randn(30).numpy(),
        "y"            : torch.randn(30).numpy(),
        "z"            : torch.randn(30).numpy(),
        "light_amount" : torch.rand(30).numpy(),
    })

    graph = Graph(Config()).build_graph(frame)
    assert graph.x.shape == (30, 8)
    assert graph.edge_attr.shape[1] == 7


def test_dataset_pipeline_splits(tiny_dataset, quiet_logger, split_config):
    splits, stats, saved = DatasetPipeline(tiny_dataset, quiet_logger, split_config, subset_fraction=1.0).run()

    assert len(splits["train"]) > 0
    assert len(splits["val"])   > 0
    assert len(splits["test"])  > 0

    sample_graph = splits["train"][0]
    assert sample_graph.x.shape[1]         == 8
    assert sample_graph.edge_attr.shape[1] == 7
    assert sample_graph.y.shape            == (1, 3)
    assert stats.target_mean.shape         == (3,)
