import torch
from torch_geometric.data import Data

from pipelines.dataset import NormalizationStats, Normalizer


def _make_split(count=20):
    graphs = []
    for _ in range(count):
        node_count = 30
        graphs.append(Data(
            x         = torch.randn(node_count, 8) * 5 + 2,
            edge_index= torch.randint(0, node_count, (2, node_count * 4)),
            edge_attr = torch.randn(node_count * 4, 7) * 3 - 1,
            y         = torch.randn(1, 3) * 10,
        ))
    return graphs


def test_normalizer_zeroes_train_mean(quiet_logger):
    splits     = {"train": _make_split(), "val": _make_split(8), "test": _make_split(8)}
    normalizer = Normalizer(quiet_logger, source_split="train")
    splits, stats = normalizer.run(splits)

    node_mean = torch.cat([graph.x for graph in splits["train"]]).mean(dim=0).abs().max().item()
    assert node_mean < 1e-3
    assert stats.target_mean.shape == (3,)


def test_normalization_stats_roundtrip(tmp_path, quiet_logger):
    splits        = {"train": _make_split(), "val": _make_split(6), "test": _make_split(6)}
    _, stats      = Normalizer(quiet_logger).run(splits)
    stats.save(tmp_path)
    loaded        = NormalizationStats.load(tmp_path)

    assert torch.equal(torch.tensor(loaded.target_std), torch.tensor(stats.target_std))
    assert torch.equal(torch.tensor(loaded.node_mean), torch.tensor(stats.node_mean))
