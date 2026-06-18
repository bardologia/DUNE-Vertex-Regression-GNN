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
