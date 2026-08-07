import numpy as np
import pytest
import torch

from configuration import DatasetConfig
from pipelines.dataset import DatasetPipeline, FeatureLayout, Graph


def test_graph_builder_shapes():
    generator = np.random.default_rng(0)
    positions = generator.normal(size=(30, 3)).astype(np.float32)
    light     = generator.exponential(size=30).astype(np.float32)

    graph = Graph(DatasetConfig()).build_from_arrays(positions, light)
    assert graph.x.shape            == (30, 17)
    assert graph.edge_attr.shape[1] == 23
    assert graph.graph_attr.shape   == (1, 33)


def test_graph_scalars_carry_extensive_quantities():
    positions = np.random.default_rng(3).normal(size=(20, 3)).astype(np.float32)
    light     = np.arange(1, 21, dtype=np.float32)

    scalars = Graph(DatasetConfig()).build_from_arrays(positions, light).graph_attr[0].numpy()

    assert scalars[3] == light.sum()
    assert scalars[4] == 20.0
    assert scalars[5] == light.max()
    assert np.all(scalars[6:9] >= 0.0)


def test_graph_scalars_carry_the_summary_moments():
    positions = np.random.default_rng(7).normal(size=(40, 3)).astype(np.float32)
    light     = np.random.default_rng(8).exponential(size=40).astype(np.float32)

    config  = DatasetConfig()
    scalars = Graph(config).build_from_arrays(positions, light).graph_attr[0].numpy()
    groups  = FeatureLayout.group_index(FeatureLayout(config).graph_features())

    sharp_weights  = light.astype(np.float64) ** 2
    sharp_centroid = (sharp_weights[:, None] * positions).sum(axis=0) / sharp_weights.sum()
    centroid       = (light[:, None] * positions).sum(axis=0) / light.sum()
    spread         = np.sqrt((light[:, None] * (positions - centroid) ** 2).sum(axis=0) / light.sum())

    assert np.allclose(scalars[groups["sharp_centroid"]], sharp_centroid, atol=1e-4)
    assert np.allclose(scalars[groups["axis_spread"]],    spread,         atol=1e-4)
    assert len(groups["axis_skew"])     == 3
    assert len(groups["light_quantile"]) == 3 * len(config.graph.graph_quantiles)


def test_graph_moment_toggles_change_dimensions():
    from pipelines.dataset import FeatureSchema

    config = DatasetConfig()
    config.graph.graph_moment_features = False
    config.graph.graph_quantiles       = ()

    assert FeatureSchema(config).dimensions()[2] == 9
    assert len(FeatureLayout(config).graph_features()) == 9


def test_feature_toggles_change_dimensions():
    from configuration import DatasetConfig as _DatasetConfig
    from pipelines.dataset import FeatureSchema

    assert FeatureSchema(_DatasetConfig()).dimensions() == (17, 23, 33)

    minimal = _DatasetConfig()
    minimal.graph.direction_features    = False
    minimal.graph.inertia_features      = False
    minimal.graph.rank_features         = False
    minimal.graph.graph_scalar_features = False
    minimal.graph.edge_rbf_count        = 0

    assert FeatureSchema(minimal).dimensions() == (7, 7, 0)
    assert "graph_attr" not in Graph(minimal).build_from_arrays(np.random.default_rng(4).normal(size=(10, 3)).astype(np.float32), np.arange(1, 11, dtype=np.float32))


def test_inertia_eigenvalues_are_not_normalized():
    positions = np.random.default_rng(5).normal(size=(50, 3)).astype(np.float32) * np.array([4.0, 1.0, 1.0], dtype=np.float32)
    light     = np.ones(50, dtype=np.float32)

    graph       = Graph(DatasetConfig()).build_from_arrays(positions, light)
    eigenvalues = graph.x[0, 13:16].numpy()

    assert eigenvalues.sum() > 1.5
    assert eigenvalues[0] > eigenvalues[1] >= eigenvalues[2]


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
    assert sample.x.shape[1]          == 17
    assert sample.edge_attr.shape[1]  == 23
    assert sample.graph_attr.shape    == (1, 33)
    assert sample.y.shape             == (1, 3)
    assert len(stats.target.methods)  == 3
    assert len(stats.node.methods)    == 17
    assert len(stats.graph.methods)   == 33


def test_global_node_reaches_every_node_in_two_hops():
    generator = np.random.default_rng(0)
    positions = generator.normal(size=(40, 3)).astype(np.float32)
    light     = generator.exponential(size=40).astype(np.float32)

    config = DatasetConfig()
    plain  = Graph(config).build_from_arrays(positions, light)

    config.graph.global_node = True
    with_global              = Graph(config).build_from_arrays(positions, light)

    global_index = with_global.x.shape[0] - 1
    source       = with_global.edge_index[0]
    target       = with_global.edge_index[1]

    assert with_global.x.shape[0]  == plain.x.shape[0] + 1
    assert with_global.x.shape[1]  == plain.x.shape[1]
    assert with_global.edge_attr.shape[1] == plain.edge_attr.shape[1]
    assert int((target == global_index).sum()) == global_index
    assert int((source == global_index).sum()) == global_index
    assert torch.allclose(with_global.graph_attr, plain.graph_attr)


def test_global_node_sits_at_the_light_centroid():
    generator = np.random.default_rng(1)
    positions = generator.normal(size=(30, 3)).astype(np.float32)
    light     = generator.exponential(size=30).astype(np.float32)

    config = DatasetConfig()
    config.graph.active_only = False
    config.graph.global_node = True

    data     = Graph(config).build_from_arrays(positions, light)
    centroid = (light[:, None] * positions).sum(axis=0) / light.sum()

    assert np.allclose(data.x[-1, 0:3].numpy(), centroid, atol=1e-4)


def test_isotropic_target_frame_shares_one_scale(dataset_config, quiet_logger):
    dataset_config.data.target_frame = "isotropic"
    _, stats = DatasetPipeline(dataset_config, quiet_logger).run()

    layout          = FeatureLayout(dataset_config)
    node_positions  = FeatureLayout.group_index(layout.node_features())["position"]
    graph_centroid  = FeatureLayout.group_index(layout.graph_features())["light_centroid"]
    target_scale    = float(stats.target.scale[0])

    assert np.allclose(stats.node.scale[node_positions],  target_scale)
    assert np.allclose(stats.graph.scale[graph_centroid], target_scale)
    assert not np.allclose(stats.node.scale[node_positions[0]], stats.node.scale[len(node_positions)])


def test_spatial_channels_adopt_the_target_frame(dataset_config, quiet_logger):
    _, stats = DatasetPipeline(dataset_config, quiet_logger).run()

    layout         = FeatureLayout(dataset_config)
    node_positions = FeatureLayout.group_index(layout.node_features())["position"]
    graph_centroid = FeatureLayout.group_index(layout.graph_features())["light_centroid"]

    assert np.allclose(stats.node.scale[node_positions],   stats.target.scale)
    assert np.allclose(stats.graph.scale[graph_centroid],  stats.target.scale)
    assert np.allclose(stats.node.center[node_positions],  stats.target.center)
    assert np.allclose(stats.graph.center[graph_centroid], stats.target.center)


def test_axiswise_target_frame_whitens_each_axis(dataset_config, quiet_logger):
    dataset_config.data.target_frame = "axiswise"
    datasets, stats = DatasetPipeline(dataset_config, quiet_logger).run()

    layout         = FeatureLayout(dataset_config)
    node_positions = FeatureLayout.group_index(layout.node_features())["position"]
    targets        = np.concatenate([datasets["train"][index].y.numpy() for index in range(len(datasets["train"]))])

    assert stats.target.methods == ["axiswise"] * 3
    assert len(set(stats.target.scale.tolist())) == 3
    assert np.allclose(targets.mean(axis=0), 0.0, atol=0.2)
    assert np.allclose(targets.std(axis=0),  1.0, atol=0.3)
    assert np.allclose(stats.node.scale[node_positions], stats.target.scale)


def test_unknown_target_frame_raises(dataset_config, quiet_logger):
    dataset_config.data.target_frame = "spherical"

    with pytest.raises(ValueError, match="Unknown data.target_frame"):
        DatasetPipeline(dataset_config, quiet_logger).run()


def test_spatial_graph_scalars_all_adopt_the_target_frame(dataset_config, quiet_logger):
    _, stats = DatasetPipeline(dataset_config, quiet_logger).run()

    groups = FeatureLayout.group_index(FeatureLayout(dataset_config).graph_features())

    for group_name in ("light_centroid", "sharp_centroid", "light_quantile"):
        channels = groups[group_name]
        assert np.allclose(stats.graph.scale[channels],  np.tile(stats.target.scale,  len(channels) // 3))
        assert np.allclose(stats.graph.center[channels], np.tile(stats.target.center, len(channels) // 3))


def test_light_centroid_maps_onto_the_normalized_target(dataset_config, quiet_logger):
    datasets, stats = DatasetPipeline(dataset_config, quiet_logger).run()

    layout         = FeatureLayout(dataset_config)
    graph_centroid = FeatureLayout.group_index(layout.graph_features())["light_centroid"]
    sample         = datasets["test"][0]

    centroid  = sample.graph_attr[0, graph_centroid].numpy()
    recovered = stats.target.inverse_torch(torch.tensor(centroid).unsqueeze(0), "cpu").numpy()[0]
    raw       = stats.graph.inverse_torch(sample.graph_attr, "cpu").numpy()[0, graph_centroid]

    assert np.allclose(recovered, raw, atol=1e-3, rtol=1e-3)


def test_augmentation_reaches_every_split(parquet_store, quiet_logger):
    def build(enabled):
        config = DatasetConfig()
        config.data.parquet_store_dir = parquet_store
        config.data.stats_sample_size = 32
        config.augmentation.enabled   = enabled
        return DatasetPipeline(config, quiet_logger).run()[0]

    clean     = build(False)
    augmented = build(True)
    repeated  = build(True)

    for split in ("val", "test"):
        clean_nodes     = [clean[split][index].x.shape[0]     for index in range(len(clean[split]))]
        augmented_nodes = [augmented[split][index].x.shape[0] for index in range(len(augmented[split]))]
        assert augmented_nodes != clean_nodes

    assert torch.equal(augmented["val"][0].x,          repeated["val"][0].x)
    assert torch.equal(augmented["test"][0].graph_attr, repeated["test"][0].graph_attr)


def test_train_augmentation_toggle_takes_effect_per_item(parquet_store, quiet_logger):
    config = DatasetConfig()
    config.data.parquet_store_dir = parquet_store
    config.data.stats_sample_size = 32
    config.augmentation.enabled   = True

    datasets, _ = DatasetPipeline(config, quiet_logger).run()
    train       = datasets["train"]

    augmented          = [train[index].x.clone() for index in range(4)]
    train.augmentation = None
    plain              = [train[index].x.clone() for index in range(4)]

    assert any(not torch.equal(before, after) for before, after in zip(augmented, plain))
    assert all(torch.equal(after, train[index].x) for index, after in enumerate(plain))


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
