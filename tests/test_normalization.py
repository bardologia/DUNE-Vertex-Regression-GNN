import numpy as np
import torch

from pipelines.dataset.normalization import FeatureGroupNormalizer, NormalizationStats


def test_heavy_tailed_nonnegative_channel_uses_log():
    generator   = np.random.default_rng(0)
    heavy_tail  = generator.exponential(scale=50.0, size=(4000, 1)) ** 2
    group       = FeatureGroupNormalizer.fit(heavy_tail)
    assert group.methods[0] == "log1p_zscore"
    assert bool(group.log_mask[0]) is True


def test_signed_channel_never_uses_log():
    generator = np.random.default_rng(1)
    signed    = generator.normal(0.0, 5.0, size=(4000, 1))
    group     = FeatureGroupNormalizer.fit(signed)
    assert group.methods[0] in ("zscore", "robust_iqr")
    assert bool(group.log_mask[0]) is False


def test_forward_inverse_recovers_values():
    generator = np.random.default_rng(2)
    matrix    = np.column_stack([
        generator.exponential(scale=10.0, size=4000) ** 2,
        generator.normal(0.0, 3.0, size=4000),
        generator.normal(-20.0, 8.0, size=4000),
    ]).astype(np.float32)

    group      = FeatureGroupNormalizer.fit(matrix)
    normalized = group.forward_numpy(matrix)
    recovered  = group.inverse_torch(torch.tensor(normalized), "cpu").numpy()

    assert np.allclose(recovered, matrix, atol=1e-2, rtol=1e-3)


def test_zscore_channel_is_standardized():
    generator  = np.random.default_rng(3)
    matrix     = generator.normal(5.0, 2.0, size=(4000, 1)).astype(np.float32)
    group      = FeatureGroupNormalizer.fit(matrix)
    normalized = group.forward_numpy(matrix)
    assert abs(float(normalized.mean())) < 1e-2
    assert abs(float(normalized.std()) - 1.0) < 5e-2


def test_stats_roundtrip(tmp_path):
    generator = np.random.default_rng(4)
    node      = FeatureGroupNormalizer.fit(generator.normal(size=(500, 8)).astype(np.float32))
    edge      = FeatureGroupNormalizer.fit(generator.normal(size=(500, 7)).astype(np.float32))
    target    = FeatureGroupNormalizer.fit(generator.normal(size=(500, 3)).astype(np.float32))

    NormalizationStats(node, edge, target).save(tmp_path)
    loaded = NormalizationStats.load(tmp_path)

    assert loaded.node.methods   == node.methods
    assert loaded.target.methods == target.methods
    assert np.allclose(loaded.edge.center, edge.center)
    assert np.allclose(loaded.target.scale, target.scale)
