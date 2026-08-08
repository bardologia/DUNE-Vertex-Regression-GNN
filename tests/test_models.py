import pytest
import torch

from models import MODEL_REGISTRY, get_model


CLAMP_BOUNDS = {"clamp_lower": (-2.0, -2.0, -2.0), "clamp_upper": (2.0, 2.0, 2.0)}


def _overrides_for(model_name):
    if model_name == "pna":
        return {**CLAMP_BOUNDS, "degree_histogram": (0, 0, 0, 0, 40)}
    return dict(CLAMP_BOUNDS)


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_model_forward_shape(model_name, synthetic_batch):
    model, config = get_model(model_name, **_overrides_for(model_name))
    model.eval()
    with torch.no_grad():
        predictions = model(synthetic_batch)

    assert tuple(predictions.shape) == (3, 3)
    assert torch.isfinite(predictions).all()


def test_sum_pooling_widens_the_readout():
    mean_max, _     = get_model("gatv2", pooling="mean_max", **CLAMP_BOUNDS)
    mean_max_sum, _ = get_model("gatv2", pooling="mean_max_sum", **CLAMP_BOUNDS)

    assert mean_max_sum.pool.out_dim == 3 * mean_max.pool.out_dim // 2


def test_graph_scalars_reach_the_head(synthetic_batch):
    model, config = get_model("gatv2", **CLAMP_BOUNDS)

    assert model.graph_scalars is not None
    assert model.graph_scalars.out_dim == config.graph_embed_dim + config.graph_dim
    assert model.regression_head.feature_proj[0].in_features == model.pool.out_dim + model.graph_scalars.out_dim

    model.eval()
    with torch.no_grad():
        baseline = model(synthetic_batch)
        synthetic_batch.graph_attr = synthetic_batch.graph_attr + 5.0
        shifted  = model(synthetic_batch)

    assert not torch.allclose(baseline, shifted)


def test_missing_graph_scalars_raise(synthetic_batch):
    model, _ = get_model("gatv2", **CLAMP_BOUNDS)
    del synthetic_batch.graph_attr

    with pytest.raises(ValueError):
        model(synthetic_batch)


def test_graph_dim_zero_drops_the_scalar_branch(synthetic_batch):
    model, _ = get_model("gatv2", graph_dim=0, **CLAMP_BOUNDS)
    del synthetic_batch.graph_attr

    assert model.graph_scalars is None

    model.eval()
    with torch.no_grad():
        predictions = model(synthetic_batch)

    assert tuple(predictions.shape) == (3, 3)


def test_output_clamp_leaks_beyond_the_envelope():
    model, config = get_model("gatv2", clamp_leak=0.05, **CLAMP_BOUNDS)
    raw           = torch.tensor([[0.5, -1.0, 3.0], [-6.0, 2.0, 2.0]])

    clamped = model.output_clamp(raw)

    assert torch.allclose(clamped[0, :2], raw[0, :2])
    assert clamped[0, 2].item() == pytest.approx(2.0 + 0.05 * 1.0)
    assert clamped[1, 0].item() == pytest.approx(-2.0 - 0.05 * 4.0)
    assert clamped[1, 1].item() == pytest.approx(2.0)
    assert clamped[1, 2].item() == pytest.approx(2.0)


def test_output_clamp_stays_differentiable_outside_the_envelope():
    model, _    = get_model("gatv2", clamp_leak=0.1, **CLAMP_BOUNDS)
    predictions = torch.tensor([[9.0, -9.0, 0.0]], requires_grad=True)

    model.output_clamp(predictions).sum().backward()

    assert predictions.grad.tolist() == [[pytest.approx(0.1), pytest.approx(0.1), pytest.approx(1.0)]]


def test_output_clamp_without_bounds_raises():
    with pytest.raises(ValueError):
        get_model("gatv2")


def test_output_clamp_can_be_disabled(synthetic_batch):
    model, _ = get_model("gatv2", output_clamp=False)

    assert model.output_clamp is None

    model.eval()
    with torch.no_grad():
        predictions = model(synthetic_batch)

    assert tuple(predictions.shape) == (3, 3)


def test_output_clamp_stays_out_of_the_state_dict():
    model, _ = get_model("gatv2", **CLAMP_BOUNDS)
    assert not [key for key in model.state_dict() if "output_clamp" in key]


def test_get_model_overrides():
    model, config = get_model("gatv2", hidden_dim=64, num_layers=2, **CLAMP_BOUNDS)
    assert config.hidden_dim == 64
    assert config.num_layers == 2


def test_model_exposes_named_submodules():
    model, _ = get_model("graphsage", **CLAMP_BOUNDS)
    assert hasattr(model, "encoder")
    assert hasattr(model, "pool")
    assert hasattr(model, "regression_head")


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_model("does_not_exist")
