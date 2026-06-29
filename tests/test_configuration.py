from configuration import MODEL_CONFIG_REGISTRY, TrainEntryConfig
from configuration.architectures import GPSConfig
from tools import ConfigCli


def test_model_config_registry_complete():
    expected = {
        "gps", "gps_lite", "gps_cascade", "gatv2", "gatv2_cascade", "gine", "gine_cascade",
        "graphsage", "pna", "gcn", "transformer", "edgeconv", "general_conv", "res_gated",
    }
    assert set(MODEL_CONFIG_REGISTRY) == expected


def test_cascade_head_type():
    from configuration.architectures import GPSCascadeConfig
    assert GPSCascadeConfig().head_type == "cascade"


def test_model_configs_declare_tunable_params():
    for name, config_class in MODEL_CONFIG_REGISTRY.items():
        space = config_class.tunable_params()
        assert isinstance(space, dict) and len(space) > 0
        for specification in space.values():
            assert specification["type"] in ("float", "int", "categorical")

    assert "heads" in MODEL_CONFIG_REGISTRY["gatv2"].tunable_params()
    assert "heads" not in MODEL_CONFIG_REGISTRY["gcn"].tunable_params()
    assert "sag_ratio" not in MODEL_CONFIG_REGISTRY["gps"].tunable_params()


def test_sagpool_config_exposes_sag_ratio():
    from dataclasses import dataclass

    @dataclass
    class SagPoolConfig(GPSConfig):
        pooling : str = "sagpool_multiscale"

    assert "sag_ratio" in SagPoolConfig.tunable_params()


def test_model_config_sections():
    from configuration.architectures import BaseGNNConfig
    assert "encoder" in BaseGNNConfig.SECTIONS
    assert "head" in BaseGNNConfig.SECTIONS


def test_gps_declares_lower_optimizer_overrides():
    overrides = GPSConfig().optimizer_overrides
    assert overrides["learning_rate_encoder"] < 1e-3
    assert overrides["learning_rate_regression_head"] < 1e-2
    assert MODEL_CONFIG_REGISTRY["gcn"]().optimizer_overrides == {}


def test_optimizer_overrides_applied_and_cli_wins():
    from pipelines.training.optimizer_overrides import OptimizerOverrideApplier

    class SilentLogger:
        def section(self, *args): pass
        def kv_table(self, *args): pass
        def subsection(self, *args): pass

    entry = TrainEntryConfig()
    OptimizerOverrideApplier(entry.training, GPSConfig(), explicit_paths=set(), logger=SilentLogger()).run()
    assert entry.training.optimizer.learning_rate_encoder == GPSConfig().optimizer_overrides["learning_rate_encoder"]

    entry = TrainEntryConfig()
    entry.training.optimizer.learning_rate_encoder = 5e-4
    OptimizerOverrideApplier(entry.training, GPSConfig(), explicit_paths={"training.optimizer.learning_rate_encoder"}, logger=SilentLogger()).run()
    assert entry.training.optimizer.learning_rate_encoder == 5e-4
    assert entry.training.optimizer.learning_rate_pool == GPSConfig().optimizer_overrides["learning_rate_pool"]


def test_optimizer_overrides_reject_unknown_field():
    import pytest
    from dataclasses import dataclass, field
    from pipelines.training.optimizer_overrides import OptimizerOverrideApplier

    @dataclass
    class BadConfig(GPSConfig):
        optimizer_overrides : dict = field(default_factory=lambda: {"nonexistent_lr": 1.0})

    class SilentLogger:
        def section(self, *args): pass
        def kv_table(self, *args): pass
        def subsection(self, *args): pass

    with pytest.raises(ValueError):
        OptimizerOverrideApplier(TrainEntryConfig().training, BadConfig(), set(), SilentLogger()).run()


def test_default_gps_config_dimensions():
    config = GPSConfig()
    assert config.input_dim  == 17
    assert config.edge_dim   == 23
    assert config.output_dim == 3
    assert config.pooling    == "mean_max"
    assert config.drop_path_rate == 0.0


def test_configcli_nested_overrides():
    config = TrainEntryConfig()
    argv   = [
        "--model_name", "gatv2",
        "--training.loop.epochs", "7",
        "--training.optimizer.learning_rate_encoder", "2e-3",
        "--dataset.split.test_size", "0.2",
        "--dataset.data.subset_fraction", "0.5",
        "--dataset.augmentation.enabled", "true",
        "--model_overrides", "{'hidden_dim': 64, 'num_layers': 3}",
    ]
    config = ConfigCli(config).apply(argv)

    assert config.model_name                                == "gatv2"
    assert config.training.loop.epochs                      == 7
    assert abs(config.training.optimizer.learning_rate_encoder - 2e-3) < 1e-12
    assert abs(config.dataset.split.test_size - 0.2) < 1e-12
    assert abs(config.dataset.data.subset_fraction - 0.5) < 1e-12
    assert config.dataset.augmentation.enabled is True
    assert config.model_overrides                           == {"hidden_dim": 64, "num_layers": 3}


def test_configcli_mapping_roundtrip():
    config  = TrainEntryConfig()
    mapping = ConfigCli.to_mapping(config)
    assert "training.loop.epochs" in mapping
    assert "model_name" in mapping
