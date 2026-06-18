from configuration import MODEL_CONFIG_REGISTRY, TrainEntryConfig
from configuration.architectures import GPSConfig
from tools import ConfigCli


def test_model_config_registry_complete():
    expected = {
        "gps", "gps_lite", "gps_cascade", "gatv2", "gatv2_cascade", "gine", "gine_cascade",
        "graphsage", "pna", "gcn", "transformer", "edgeconv", "general_conv", "res_gated", "supergat",
    }
    assert set(MODEL_CONFIG_REGISTRY) == expected


def test_cascade_head_type():
    from configuration.architectures import GPSCascadeConfig
    assert GPSCascadeConfig().head_type == "cascade"


def test_default_gps_config_dimensions():
    config = GPSConfig()
    assert config.input_dim  == 8
    assert config.edge_dim   == 7
    assert config.output_dim == 3
    assert config.pooling    == "sagpool_multiscale"


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
