import numpy as np
import pandas as pd

from configuration import CrossValidationConfig, CrossValidationEntryConfig
from pipelines.cross_validation import CrossValidationPipeline, CrossValidationReport
from pipelines.dataset import CrossValidationSplitter, StratificationLabeller


COORDINATE_COLUMNS = ("event_x", "event_y", "event_z")


def _targets(count, seed=0):
    generator = np.random.default_rng(seed)
    return pd.DataFrame(generator.normal(size=(count, 3)).astype(np.float32), columns=list(COORDINATE_COLUMNS))


def test_stratification_labeller_bin_counts():
    labeller = StratificationLabeller()
    targets  = _targets(300).to_numpy(dtype=np.float32)

    single_group = labeller.build_labels(targets, number_of_bins=1)
    assert np.unique(single_group).tolist() == [0]

    multi_group = labeller.build_labels(targets, number_of_bins=3)
    assert len(np.unique(multi_group)) > 1
    assert len(multi_group) == len(targets)


def test_stratified_folds_partition_and_disjoint(quiet_logger):
    count  = 400
    config = CrossValidationConfig(n_folds=5, validation_fraction=0.1)
    folds  = CrossValidationSplitter(quiet_logger, COORDINATE_COLUMNS, config).split(_targets(count))

    assert len(folds) == config.n_folds

    all_test = np.concatenate([test_indices for _, _, test_indices in folds])
    assert len(all_test) == count
    assert len(np.unique(all_test)) == count

    for train_indices, validation_indices, test_indices in folds:
        partition = set(train_indices) | set(validation_indices) | set(test_indices)
        assert partition == set(range(count))
        assert not (set(train_indices) & set(validation_indices))
        assert not (set(train_indices) & set(test_indices))
        assert not (set(validation_indices) & set(test_indices))
        assert len(validation_indices) >= 1


def test_unstratified_folds_cover_all(quiet_logger):
    count  = 120
    config = CrossValidationConfig(n_folds=3, stratified=False)
    folds  = CrossValidationSplitter(quiet_logger, COORDINATE_COLUMNS, config).split(_targets(count, seed=1))

    all_test = np.concatenate([test_indices for _, _, test_indices in folds])
    assert sorted(all_test.tolist()) == list(range(count))


def test_report_aggregates_across_folds(quiet_logger, tmp_path):
    fold_records = [
        {"fold": 0, "run_directory": "a", "metrics": {"euclidean_mean": 10.0, "mae": 4.0, "count": 50}},
        {"fold": 1, "run_directory": "b", "metrics": {"euclidean_mean": 12.0, "mae": 6.0, "count": 50}},
    ]
    report  = CrossValidationReport(tmp_path, quiet_logger, CrossValidationConfig(n_folds=2))
    summary = report.build(fold_records)

    assert summary["aggregate"]["euclidean_mean"]["mean"] == 11.0
    assert summary["aggregate"]["mae"]["min"] == 4.0
    assert summary["aggregate"]["mae"]["max"] == 6.0
    assert (tmp_path / "cross_validation_metrics.json").exists()
    assert (tmp_path / "cross_validation_report.md").exists()


def test_cv_folds_are_group_aware_on_base_event(dataset_config, quiet_logger, tmp_path):
    entry                              = CrossValidationEntryConfig(model_name="graphsage")
    entry.dataset                      = dataset_config
    entry.dataset.data.augment_octants = True
    entry.cross_validation             = CrossValidationConfig(n_folds=3, validation_fraction=0.2)
    entry.training.io.log_base_dir     = tmp_path

    pipeline = CrossValidationPipeline(entry, logger=quiet_logger)
    pipeline._prepare_run()
    pipeline._prepare_folds()

    base_ids         = pipeline.samples[:, 7].astype(int)
    covered_test_ids = set()

    for train_indices, validation_indices, test_indices in pipeline.folds:
        train_ids      = set(base_ids[train_indices].tolist())
        validation_ids = set(base_ids[validation_indices].tolist())
        test_ids       = set(base_ids[test_indices].tolist())

        assert train_ids.isdisjoint(validation_ids)
        assert train_ids.isdisjoint(test_ids)
        assert validation_ids.isdisjoint(test_ids)
        covered_test_ids |= test_ids

    total_test_samples = sum(len(test_indices) for _, _, test_indices in pipeline.folds)
    assert total_test_samples == len(covered_test_ids)


def test_pipeline_runs_two_folds(dataset_config, quiet_logger):
    entry                  = CrossValidationEntryConfig(model_name="graphsage")
    entry.dataset          = dataset_config
    entry.dataset.data.subset_fraction = 0.5
    entry.dataset.data.stats_sample_size = 16

    entry.cross_validation = CrossValidationConfig(n_folds=2, validation_fraction=0.2)

    entry.training.loop.epochs               = 1
    entry.training.loop.batch_size           = 8
    entry.training.loop.num_workers          = 0
    entry.training.loop.use_amp              = False
    entry.training.loop.pin_memory           = False
    entry.training.loop.persistent_workers   = False
    entry.training.loop.device               = "cpu"
    entry.training.loop.validation_frequency = 1
    entry.training.io.log_base_dir           = quiet_logger.log_dir or "."

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temporary_directory:
        entry.training.io.log_base_dir = Path(temporary_directory)
        summary = CrossValidationPipeline(entry, logger=quiet_logger).run()

        assert len(summary["folds"]) == 2
        assert "euclidean_mean" in summary["aggregate"]
        assert Path(summary["report_path"]).exists()
        assert Path(summary["metrics_path"]).exists()


def test_pna_pipeline_injects_degree_histogram_per_fold(dataset_config, quiet_logger):
    entry                  = CrossValidationEntryConfig(model_name="pna")
    entry.dataset          = dataset_config
    entry.dataset.data.subset_fraction = 0.5
    entry.dataset.data.stats_sample_size = 16

    entry.cross_validation = CrossValidationConfig(n_folds=2, validation_fraction=0.2)

    entry.training.loop.epochs               = 1
    entry.training.loop.batch_size           = 8
    entry.training.loop.num_workers          = 0
    entry.training.loop.use_amp              = False
    entry.training.loop.pin_memory           = False
    entry.training.loop.persistent_workers   = False
    entry.training.loop.device               = "cpu"
    entry.training.loop.validation_frequency = 1

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temporary_directory:
        entry.training.io.log_base_dir = Path(temporary_directory)
        summary = CrossValidationPipeline(entry, logger=quiet_logger).run()

        assert len(summary["folds"]) == 2
        assert entry.model_overrides.get("degree_histogram")
