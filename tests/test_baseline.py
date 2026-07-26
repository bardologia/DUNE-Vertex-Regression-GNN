import numpy as np
import pytest

from configuration import BaselineEntryConfig, DatasetConfig
from configuration.baseline import BaselineBoosterConfig, BaselineFeatureConfig, BaselineSearchConfig
from pipelines.baseline import BASELINE_MODEL_FAMILIES, BoosterTuner, SummaryFeatureExtractor, TabularSplitBuilder


def _synthetic_event(seed=0, sensors=40):
    generator = np.random.default_rng(seed)
    positions = generator.normal(size=(sensors, 3)).astype(np.float32)
    light     = generator.exponential(size=sensors).astype(np.float32)
    light[generator.random(sensors) < 0.3] = 0.0
    light[0] = 5.0
    return positions, light


def _synthetic_splits(seed=0, rows=240, features=12):
    generator = np.random.default_rng(seed)

    def make(count):
        X = generator.normal(size=(count, features)).astype(np.float32)
        y = np.column_stack([X[:, 0] * 2.0, X[:, 1] - X[:, 2], X[:, 3]]).astype(np.float32)
        return X, y + 0.01 * generator.normal(size=(count, 3)).astype(np.float32)

    return {"train": make(rows), "val": make(60), "test": make(60)}


def test_feature_vector_matches_names():
    extractor        = SummaryFeatureExtractor(BaselineFeatureConfig())
    positions, light = _synthetic_event()

    vector = extractor.extract(positions, light)
    assert vector.shape == (len(extractor.names()),)
    assert np.isfinite(vector).all()


def test_feature_extraction_is_deterministic():
    extractor        = SummaryFeatureExtractor(BaselineFeatureConfig())
    positions, light = _synthetic_event(seed=7)

    assert np.array_equal(extractor.extract(positions, light), extractor.extract(positions, light))


def test_octant_reflection_flips_sign_carrying_features():
    extractor        = SummaryFeatureExtractor(BaselineFeatureConfig())
    positions, light = _synthetic_event(seed=3)

    names     = extractor.names()
    original  = extractor.extract(positions, light)
    reflected = extractor.extract(positions * np.array([-1.0, -1.0, -1.0], dtype=np.float32), light)

    centroid = slice(names.index("centroid_x"), names.index("centroid_z") + 1)
    assert np.allclose(reflected[centroid], -original[centroid], atol=1e-5)
    assert reflected[names.index("total_light")] == original[names.index("total_light")]
    assert np.allclose(reflected[names.index("spread_x")], original[names.index("spread_x")], atol=1e-5)


def test_feature_extraction_requires_light():
    extractor = SummaryFeatureExtractor(BaselineFeatureConfig())
    with pytest.raises(ValueError):
        extractor.extract(np.zeros((10, 3), dtype=np.float32), np.zeros(10, dtype=np.float32))


def test_tabular_split_builder_expands_train_octants(parquet_store, quiet_logger):
    from pipelines.dataset.parquet_store import ParquetEventReader

    config = DatasetConfig()
    config.data.parquet_store_dir = parquet_store
    config.data.augment_octants   = True

    builder = TabularSplitBuilder(config, BaselineFeatureConfig(), quiet_logger)
    splits  = builder.run()

    train_features, train_targets = splits["train"]
    val_features, val_targets     = splits["val"]

    octant_frame = ParquetEventReader(parquet_store).load_store().octant_frame
    train_octants = int(octant_frame["base_event_id"].isin(set(builder.split_base_ids["train"])).sum())

    assert train_features.shape[1] == len(builder.feature_names)
    assert train_features.shape[0] == train_octants
    assert train_features.shape[0] > len(builder.split_base_ids["train"])
    assert val_features.shape[0]   == len(builder.split_base_ids["val"])
    assert train_targets.shape[1]  == 3
    assert np.isfinite(train_features).all() and np.isfinite(val_features).all()


@pytest.mark.parametrize("family_key", sorted(BASELINE_MODEL_FAMILIES))
def test_booster_families_fit_and_predict(family_key):
    booster = BaselineBoosterConfig(max_estimators=40, early_stopping_rounds=10, thread_count=2)
    splits  = _synthetic_splits()
    family  = BASELINE_MODEL_FAMILIES[family_key]

    model = family(booster, seed=0).fit(splits["train"][0], splits["train"][1], splits["val"][0], splits["val"][1], {"learning_rate": 0.1})

    predictions = model.predict(splits["test"][0])
    assert predictions.shape == (60, 3)
    assert len(model.best_iterations()) == 3

    importances = model.importances([f"feature_{index}" for index in range(splits["train"][0].shape[1])])
    assert len(importances) == splits["train"][0].shape[1]
    assert abs(sum(importances.values()) - 1.0) < 1e-6


def test_booster_tuner_returns_finished_study(quiet_logger):
    booster = BaselineBoosterConfig(max_estimators=25, early_stopping_rounds=5, thread_count=2)
    search  = BaselineSearchConfig(n_trials=2, startup_trials=1, random_state=0)
    splits  = _synthetic_splits(seed=1)

    study = BoosterTuner(BASELINE_MODEL_FAMILIES["lightgbm"], booster, search, 0, splits, quiet_logger).optimize()
    assert len(study.trials) == 2
    assert np.isfinite(study.best_value)


def test_baseline_entry_defaults_enable_octants():
    entry = BaselineEntryConfig()
    assert entry.dataset.data.augment_octants is True
    assert set(entry.models) == set(BASELINE_MODEL_FAMILIES)
