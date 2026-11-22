import numpy as np

from configuration import CrossValidationConfig, CrossValidationEntryConfig
from pipelines.cross_validation import CrossValidationPipeline, CrossValidationReport
from pipelines.dataset import LatticePlanes, PlaneSplitter


def _lattice(count_per_axis=12, spacing=0.5):
    axis  = np.arange(count_per_axis, dtype=np.float32) * spacing
    grids = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([grid.ravel() for grid in grids], axis=1)


def test_lattice_planes_hold_every_stride_plane():
    targets = _lattice()
    held    = LatticePlanes.held(targets, stride=6, offset=3)

    for axis in range(3):
        planes = np.unique(targets[held[:, axis], axis])
        assert planes.tolist() == [1.5, 4.5]


def test_scored_flags_cover_every_axis_of_train_events_and_only_held_axes_otherwise():
    targets = _lattice()
    held    = LatticePlanes.held(targets, stride=6, offset=3)
    scored  = LatticePlanes.scored(targets, stride=6, offset=3)

    train = ~held.any(axis=1)
    assert scored[train].all()
    assert (scored[~train] == held[~train]).all()


def test_plane_split_keeps_held_planes_out_of_train(quiet_logger):
    targets          = _lattice()
    train, val, test = PlaneSplitter(quiet_logger, 6, 3, 0.5, 42).split(targets)
    held             = LatticePlanes.held(targets, stride=6, offset=3)

    assert set(train) | set(val) | set(test) == set(range(len(targets)))
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert not held[train].any()
    assert held[val].any(axis=1).all() and held[test].any(axis=1).all()


def test_plane_folds_hold_every_coordinate_once():
    targets = _lattice()
    counts  = sum(LatticePlanes.held(targets, stride=3, offset=offset).astype(int) for offset in range(3))

    assert (counts == 1).all()


def test_report_aggregates_across_folds(quiet_logger, tmp_path):
    fold_records = [
        {"fold": 0, "run_directory": "a", "metrics": {"rms_3d": 10.0, "mae": 4.0, "count": 50}},
        {"fold": 1, "run_directory": "b", "metrics": {"rms_3d": 12.0, "mae": 6.0, "count": 50}},
    ]
    report  = CrossValidationReport(tmp_path, quiet_logger, CrossValidationConfig(n_folds=2))
    summary = report.build(fold_records)

    assert summary["aggregate"]["rms_3d"]["mean"] == 11.0
    assert summary["aggregate"]["mae"]["min"] == 4.0
    assert summary["aggregate"]["mae"]["max"] == 6.0
    assert (tmp_path / "cross_validation_metrics.json").exists()
    assert (tmp_path / "cross_validation_report.md").exists()


def test_cv_folds_are_group_aware_on_base_event(dataset_config, quiet_logger, tmp_path):
    entry                              = CrossValidationEntryConfig(model_name="graphsage")
    entry.dataset                      = dataset_config
    entry.dataset.data.augment_octants = True
    entry.cross_validation             = CrossValidationConfig(n_folds=3, val_share=0.5)
    entry.training.io.log_base_dir     = tmp_path

    pipeline = CrossValidationPipeline(entry, logger=quiet_logger)
    pipeline._prepare_run()
    pipeline._prepare_folds()

    base_ids = pipeline.samples[:, 7].astype(int)
    targets  = pipeline.dataset_pipeline.event_targets

    for fold_index, (train_indices, validation_indices, test_indices) in enumerate(pipeline.folds):
        train_ids      = set(base_ids[train_indices].tolist())
        validation_ids = set(base_ids[validation_indices].tolist())
        test_ids       = set(base_ids[test_indices].tolist())
        held           = LatticePlanes.held(targets, stride=3, offset=fold_index)

        assert train_ids.isdisjoint(validation_ids)
        assert train_ids.isdisjoint(test_ids)
        assert validation_ids.isdisjoint(test_ids)
        assert not held[sorted(train_ids)].any()


def test_pipeline_runs_two_folds(dataset_config, quiet_logger):
    entry                  = CrossValidationEntryConfig(model_name="graphsage")
    entry.dataset          = dataset_config
    entry.dataset.data.subset_fraction = 0.5
    entry.dataset.data.stats_sample_size = 16

    entry.cross_validation = CrossValidationConfig(n_folds=3, val_share=0.5)

    entry.training.loop.epochs               = 1
    entry.training.loop.batch_size           = 8
    entry.training.loop.num_workers          = 0
    entry.training.loop.use_amp              = False
    entry.training.loop.pin_memory           = False
    entry.training.loop.persistent_workers   = False
    entry.training.loop.device               = "cpu"
    entry.training.io.log_base_dir           = quiet_logger.log_dir or "."

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temporary_directory:
        entry.training.io.log_base_dir = Path(temporary_directory)
        summary = CrossValidationPipeline(entry, logger=quiet_logger).run()

        assert len(summary["folds"]) == 3
        assert "rms_3d" in summary["aggregate"]
        assert Path(summary["report_path"]).exists()
        assert Path(summary["metrics_path"]).exists()


def test_pna_pipeline_injects_degree_histogram_per_fold(dataset_config, quiet_logger):
    entry                  = CrossValidationEntryConfig(model_name="pna")
    entry.dataset          = dataset_config
    entry.dataset.data.subset_fraction = 0.5
    entry.dataset.data.stats_sample_size = 16

    entry.cross_validation = CrossValidationConfig(n_folds=3, val_share=0.5)

    entry.training.loop.epochs               = 1
    entry.training.loop.batch_size           = 8
    entry.training.loop.num_workers          = 0
    entry.training.loop.use_amp              = False
    entry.training.loop.pin_memory           = False
    entry.training.loop.persistent_workers   = False
    entry.training.loop.device               = "cpu"

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temporary_directory:
        entry.training.io.log_base_dir = Path(temporary_directory)
        summary = CrossValidationPipeline(entry, logger=quiet_logger).run()

        assert len(summary["folds"]) == 3
        assert entry.model_overrides.get("degree_histogram")
