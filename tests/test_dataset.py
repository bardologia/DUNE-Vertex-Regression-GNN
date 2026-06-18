import numpy as np

from configuration import DatasetConfig
from pipelines.dataset import DatasetPipeline, Graph


def test_graph_builder_shapes():
    generator = np.random.default_rng(0)
    positions = generator.normal(size=(30, 3)).astype(np.float32)
    light     = generator.exponential(size=30).astype(np.float32)

    graph = Graph(DatasetConfig()).build_from_arrays(positions, light)
    assert graph.x.shape         == (30, 17)
    assert graph.edge_attr.shape[1] == 23


def test_feature_toggles_change_dimensions():
    from configuration import DatasetConfig as _DatasetConfig
    from pipelines.dataset import FeatureSchema

    assert FeatureSchema(_DatasetConfig()).dimensions() == (17, 23)

    minimal = _DatasetConfig()
    minimal.graph.direction_features = False
    minimal.graph.inertia_features   = False
    minimal.graph.rank_features      = False
    minimal.graph.edge_rbf_count     = 0
    assert FeatureSchema(minimal).dimensions() == (7, 7)


def test_active_only_prunes_dark_nodes():
    positions = np.random.default_rng(1).normal(size=(40, 3)).astype(np.float32)
    light     = np.zeros(40, dtype=np.float32)
    light[:12] = np.arange(1, 13, dtype=np.float32)

    config        = DatasetConfig()
    config.graph.active_only = True
    graph         = Graph(config).build_from_arrays(positions, light)
    assert graph.x.shape[0] == 12


def test_max_active_nodes_caps_to_brightest():
    positions = np.random.default_rng(2).normal(size=(40, 3)).astype(np.float32)
    light     = np.arange(1, 41, dtype=np.float32)

    config        = DatasetConfig()
    config.graph.active_only      = True
    config.graph.max_active_nodes = 8
    graph         = Graph(config).build_from_arrays(positions, light)
    assert graph.x.shape[0] == 8


def test_dataset_pipeline_produces_splits(dataset_config, quiet_logger):
    datasets, stats = DatasetPipeline(dataset_config, quiet_logger).run()

    assert len(datasets["train"]) > 0
    assert len(datasets["val"])   > 0
    assert len(datasets["test"])  > 0

    sample = datasets["train"][0]
    assert sample.x.shape[1]         == 17
    assert sample.edge_attr.shape[1] == 23
    assert sample.y.shape            == (1, 3)
    assert len(stats.target.methods) == 3
    assert len(stats.node.methods)   == 17


def test_octant_split_has_no_base_event_leakage(parquet_store, quiet_logger):
    config = DatasetConfig()
    config.data.parquet_store_dir = parquet_store
    config.data.stats_sample_size = 32
    config.data.augment_octants   = True

    datasets, _ = DatasetPipeline(config, quiet_logger).run()

    def base_ids(dataset):
        return set(int(sample[7]) for sample in dataset.samples)

    train_ids = base_ids(datasets["train"])
    val_ids   = base_ids(datasets["val"])
    test_ids  = base_ids(datasets["test"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    total_samples = len(datasets["train"]) + len(datasets["val"]) + len(datasets["test"])
    assert total_samples > len(train_ids | val_ids | test_ids)


def test_missing_store_raises(quiet_logger, tmp_path):
    config = DatasetConfig()
    config.data.parquet_store_dir = tmp_path / "absent"
    config.data.build_store       = False

    try:
        DatasetPipeline(config, quiet_logger).run()
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
