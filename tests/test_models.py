import pytest
import torch

from models import MODEL_REGISTRY, get_model


def _overrides_for(model_name):
    if model_name == "pna":
        return {"degree_histogram": (0, 0, 0, 0, 40)}
    return {}


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_model_forward_shape(model_name, synthetic_batch):
    model, config = get_model(model_name, **_overrides_for(model_name))
    model.eval()
    with torch.no_grad():
        predictions = model(synthetic_batch)

    assert tuple(predictions.shape) == (3, 3)
    assert torch.isfinite(predictions).all()


def test_sum_pooling_widens_the_readout():
    mean_max, _     = get_model("gatv2", pooling="mean_max")
    mean_max_sum, _ = get_model("gatv2", pooling="mean_max_sum")

    assert mean_max_sum.pool.out_dim == 3 * mean_max.pool.out_dim // 2


def test_graph_scalars_reach_the_head(synthetic_batch):
    model, config = get_model("gatv2")

    assert model.graph_scalars is not None
    assert model.regression_head.feature_proj[0].in_features == model.pool.out_dim + config.graph_embed_dim

    model.eval()
    with torch.no_grad():
        baseline = model(synthetic_batch)
        synthetic_batch.graph_attr = synthetic_batch.graph_attr + 5.0
        shifted  = model(synthetic_batch)

    assert not torch.allclose(baseline, shifted)


def test_missing_graph_scalars_raise(synthetic_batch):
    model, _ = get_model("gatv2")
    del synthetic_batch.graph_attr

    with pytest.raises(ValueError):
        model(synthetic_batch)


def test_graph_dim_zero_drops_the_scalar_branch(synthetic_batch):
    model, _ = get_model("gatv2", graph_dim=0)
    del synthetic_batch.graph_attr

    assert model.graph_scalars is None

    model.eval()
    with torch.no_grad():
        predictions = model(synthetic_batch)

    assert tuple(predictions.shape) == (3, 3)


def test_get_model_overrides():
    model, config = get_model("gatv2", hidden_dim=64, num_layers=2)
    assert config.hidden_dim == 64
    assert config.num_layers == 2


def test_model_exposes_named_submodules():
    model, _ = get_model("graphsage")
    assert hasattr(model, "encoder")
    assert hasattr(model, "pool")
    assert hasattr(model, "regression_head")


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_model("does_not_exist")
