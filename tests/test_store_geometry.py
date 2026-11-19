import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webui"))

from configuration                               import TrainEntryConfig
from configuration.benchmark.general             import BenchmarkConfig
from configuration.explainability                import FeatureImportanceConfig
from configuration.sweep                         import EnergyEfficiencySweepConfig
from tools                                       import Logger
from pipelines.benchmark.pipeline                import BenchmarkPipeline
from pipelines.dataset                           import ParquetEventReader
from pipelines.explainability                    import FeatureImportancePipeline
from pipelines.inference                         import InferencePipeline
from pipelines.sweep                             import EnergyEfficiencySweepPipeline
from pipelines.inference.dataset_explorer_export import DatasetExplorerExportPipeline
from pipelines.shared                            import CacheGeometryStamp, StoreGeometryRecord, TrainingRunMetadata
from pipelines.training                          import TrainingPipeline

from event_explorer import EventExplorer
from project_paths  import ProjectPaths


class StubServerLogger:
    def ok(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


STORE_GEOMETRY = {"geometry_fingerprint": "a" * 64, "channel_count": 6909}
OTHER_GEOMETRY = {"geometry_fingerprint": "b" * 64, "channel_count": 6909}


@pytest.fixture(scope="module")
def trained_run(parquet_store, tmp_path_factory):
    entry                                = TrainEntryConfig()
    entry.model_name                     = "graphsage"
    entry.dataset.data.parquet_store_dir = parquet_store
    entry.dataset.data.stats_sample_size = 16
    entry.training.io.log_base_dir       = tmp_path_factory.mktemp("geometry_runs")
    entry.training.loop.device           = "cpu"
    entry.training.loop.epochs           = 1
    entry.training.loop.batch_size       = 8
    entry.training.loop.num_workers      = 0
    entry.training.warmup.warmup_steps   = 2

    summary = TrainingPipeline(entry, logger=Logger(log_dir="", name="train", level="ERROR")).run()
    return Path(summary["run_directory"]), entry


def test_record_round_trips_and_refuses_another_geometry(tmp_path):
    record = StoreGeometryRecord(tmp_path)
    record.save(STORE_GEOMETRY)

    assert record.recorded() == STORE_GEOMETRY
    record.require(STORE_GEOMETRY)
    with pytest.raises(ValueError, match="store geometry"):
        record.require(OTHER_GEOMETRY)


def test_missing_record_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="before the documented channel mapping"):
        StoreGeometryRecord(tmp_path).recorded()


def test_missing_directory_is_not_reported_as_an_old_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        StoreGeometryRecord(tmp_path / "absent" / "metadata").recorded()


def test_old_runs_are_refused_before_their_logs_are_truncated(tmp_path):
    old_run = tmp_path / "edgeconv_old"
    (old_run / "metadata").mkdir(parents=True)
    (old_run / "logs").mkdir()
    for name in ("inference", "explain", "sweep"):
        (old_run / "logs" / f"{name}.log").write_text("earlier analysis", encoding="utf-8")

    explain_config               = FeatureImportanceConfig()
    explain_config.run_directory = old_run
    sweep_config                 = EnergyEfficiencySweepConfig()
    sweep_config.run_directory   = old_run

    builders = (lambda: InferencePipeline(old_run), lambda: FeatureImportancePipeline(explain_config), lambda: EnergyEfficiencySweepPipeline(sweep_config))
    for build in builders:
        with pytest.raises(FileNotFoundError, match="before the documented channel mapping"):
            build()

    assert sorted(path.name for path in (old_run / "logs").iterdir()) == ["explain.log", "inference.log", "sweep.log"]
    assert all((old_run / "logs" / f"{name}.log").read_text(encoding="utf-8") == "earlier analysis" for name in ("inference", "explain", "sweep"))


def test_cache_stamp_refuses_unstamped_and_foreign_caches(tmp_path):
    stamped  = tmp_path / "stamped.npz"
    old      = tmp_path / "old.npz"
    np.savez_compressed(stamped, gt=np.zeros((2, 3)), **CacheGeometryStamp.payload(STORE_GEOMETRY["geometry_fingerprint"]))
    np.savez_compressed(old, gt=np.zeros((2, 3)))

    CacheGeometryStamp.require(np.load(stamped), stamped, STORE_GEOMETRY["geometry_fingerprint"])
    with pytest.raises(ValueError, match="store geometry"):
        CacheGeometryStamp.require(np.load(stamped), stamped, OTHER_GEOMETRY["geometry_fingerprint"])
    with pytest.raises(ValueError, match="before the documented channel mapping"):
        CacheGeometryStamp.require(np.load(old), old, STORE_GEOMETRY["geometry_fingerprint"])


def test_training_records_the_store_geometry(trained_run, parquet_store):
    run_directory, _ = trained_run
    recorded         = StoreGeometryRecord(run_directory / "metadata").recorded()

    assert recorded == {"geometry_fingerprint": ParquetEventReader(parquet_store).load_geometry().geometry_fingerprint, "channel_count": 6909}


def test_inference_refuses_a_run_without_record(trained_run, tmp_path):
    run_directory, _ = trained_run
    old_run          = tmp_path / run_directory.name
    for path in run_directory.rglob("*"):
        if path.is_file() and path.name != StoreGeometryRecord.FILENAME:
            target = old_run / path.relative_to(run_directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())

    with pytest.raises(FileNotFoundError, match="before the documented channel mapping"):
        InferencePipeline(old_run, logger=Logger(log_dir="", name="infer", level="ERROR"), device="cpu", splits=("test",), batch_size=8).run()


def test_inference_refuses_a_run_of_another_geometry(trained_run):
    run_directory, _ = trained_run
    record           = StoreGeometryRecord(run_directory / "metadata")
    original         = record.recorded()

    record.save(OTHER_GEOMETRY)
    try:
        with pytest.raises(ValueError, match="store geometry"):
            InferencePipeline(run_directory, logger=Logger(log_dir="", name="infer", level="ERROR"), device="cpu", splits=("test",), batch_size=8).run()
    finally:
        record.save(original)


def test_resumed_training_refuses_a_run_without_record_before_writing(trained_run):
    run_directory, entry = trained_run
    run_name             = run_directory.name.removeprefix(f"{entry.model_name}_")
    (run_directory / "metadata" / StoreGeometryRecord.FILENAME).rename(run_directory / "metadata" / "held.json")
    before = sorted(path for path in run_directory.rglob("*"))

    try:
        entry.training.loop.resume = True
        with pytest.raises(FileNotFoundError, match="before the documented channel mapping"):
            TrainingRunMetadata(entry.training, entry.model_name, run_directory.parent, run_name=run_name, logger=Logger(log_dir="", name="resume", level="ERROR"))
        assert sorted(path for path in run_directory.rglob("*")) == before
    finally:
        entry.training.loop.resume = False
        (run_directory / "metadata" / "held.json").rename(run_directory / "metadata" / StoreGeometryRecord.FILENAME)


def test_resumed_training_refuses_a_run_of_another_geometry(trained_run):
    run_directory, entry = trained_run
    run_name             = run_directory.name.removeprefix(f"{entry.model_name}_")

    try:
        entry.training.loop.resume = True
        metadata                   = TrainingRunMetadata(entry.training, entry.model_name, run_directory.parent, run_name=run_name, logger=Logger(log_dir="", name="resume", level="ERROR"))
        assert metadata.resuming
        with pytest.raises(ValueError, match="store geometry"):
            metadata.save_store_geometry(OTHER_GEOMETRY)
        metadata.close()
    finally:
        entry.training.loop.resume = False


def test_benchmark_resume_refuses_old_stage_results_before_writing(tmp_path, quiet_logger):
    config                          = BenchmarkConfig()
    config.run_name                 = "old"
    config.training.io.log_base_dir = tmp_path
    pipeline_directory              = tmp_path / "benchmark_old" / "pipeline"
    pipeline_directory.mkdir(parents=True)
    (pipeline_directory / "training_results.json").write_text(json.dumps({"graphsage": {"status": "DONE", "metrics": {"euclidean_mean": 0.1}}}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="before the documented channel mapping"):
        BenchmarkPipeline(config, logger=quiet_logger).run()
    assert sorted(path.name for path in pipeline_directory.iterdir()) == ["training_results.json"]


def test_benchmark_resume_refuses_stage_results_of_another_geometry(tmp_path, quiet_logger):
    pipeline                    = BenchmarkPipeline(BenchmarkConfig(), logger=quiet_logger)
    pipeline.pipeline_directory = tmp_path
    pipeline.store_geometry     = STORE_GEOMETRY
    (tmp_path / "training_results.json").write_text(json.dumps({"graphsage": {"status": "DONE", "metrics": {"euclidean_mean": 0.1}}}), encoding="utf-8")

    StoreGeometryRecord(tmp_path).save(OTHER_GEOMETRY)
    with pytest.raises(ValueError, match="store geometry"):
        pipeline._record_store_geometry()

    StoreGeometryRecord(tmp_path).save(STORE_GEOMETRY)
    pipeline._record_store_geometry()


def test_dataset_explorer_cache_is_stamped_and_old_caches_are_refused(dataset_config, quiet_logger, tmp_path):
    path        = DatasetExplorerExportPipeline("raw", tmp_path, logger=quiet_logger, dataset_config=dataset_config).run()
    fingerprint = ParquetEventReader(dataset_config.data.parquet_store_dir).load_geometry().geometry_fingerprint
    explorer    = EventExplorer(ProjectPaths(), StubServerLogger())

    data, meta = explorer._read_dataset_cache(path, "raw", fingerprint)
    assert meta["count"] == data["gt"].shape[0]

    archive = dict(np.load(path))
    archive.pop(CacheGeometryStamp.KEY)
    old = tmp_path / "old_raw.npz"
    np.savez_compressed(old, **archive)

    with pytest.raises(ValueError, match="before the documented channel mapping"):
        explorer._read_dataset_cache(old, "raw", fingerprint)
