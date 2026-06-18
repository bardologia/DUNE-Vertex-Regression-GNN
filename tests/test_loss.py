import numpy as np
import torch
from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader

from configuration import LossConfig
from pipelines.dataset.normalization import FeatureGroupNormalizer, NormalizationStats
from pipelines.training.loss import Loss
from tools import Logger, NullTracker


def _stats():
    generator = np.random.default_rng(0)
    node      = FeatureGroupNormalizer.fit(generator.normal(size=(500, 8)).astype(np.float32))
    edge      = FeatureGroupNormalizer.fit(generator.normal(size=(500, 7)).astype(np.float32))
    target    = FeatureGroupNormalizer.fit(generator.normal(size=(500, 3)).astype(np.float32))
    return NormalizationStats(node, edge, target)


def _batch():
    graphs = []
    for _ in range(3):
        node_count = 20
        graphs.append(Data(
            x          = torch.randn(node_count, 8),
            edge_index = torch.randint(0, node_count, (2, node_count * 4)),
            edge_attr  = torch.randn(node_count * 4, 7),
            y          = torch.randn(1, 3),
        ))
    return next(iter(DataLoader(graphs, batch_size=3)))


def _loss(config):
    return Loss(config, _stats(), Logger(log_dir="", name="loss", level="ERROR"), NullTracker())


def test_physics_loss_is_finite_and_differentiable():
    batch       = _batch()
    predictions = torch.randn(3, 3, requires_grad=True)
    config      = LossConfig(euclidean_weight=0.5, containment_weight=0.1, light_falloff_weight=0.5)

    result = _loss(config)(predictions, batch, step=0)
    result["total_loss"].backward()

    assert torch.isfinite(result["total_loss"])
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_data_only_loss_matches_mse():
    batch       = _batch()
    predictions = torch.randn(3, 3)
    result      = _loss(LossConfig())(predictions, batch, step=0)
    expected    = torch.nn.functional.mse_loss(predictions, batch.y)
    assert abs(result["total_loss"].item() - expected.item()) < 1e-5
