import json
from pathlib import Path

from configuration.architectures import MODEL_CONFIG_REGISTRY
from configuration.benchmark     import BenchmarkConfig
from pipelines.benchmark         import BenchmarkPipeline, ComparisonReport, ModelSizer, TrialCollector


def test_model_sizer_matches_reference(quiet_logger):
    config = BenchmarkConfig()
    config.size_match.reference_model = "gps"

    sizer  = ModelSizer(config, quiet_logger)
    target = sizer.reference_count()
    result = sizer.match("graphsage", target)

    assert result.parameters > 0
    assert abs(result.deviation_pct) < 5.0
    assert result.overrides["hidden_dim"] == result.width


def test_model_sizer_produces_head_divisible_width(quiet_logger):
    config = BenchmarkConfig()
    config.size_match.reference_model = "gps"

    sizer  = ModelSizer(config, quiet_logger)
    target = sizer.reference_count()
    result = sizer.match("gps_lite", target)

    heads = MODEL_CONFIG_REGISTRY["gps_lite"]().heads

    assert result.parameters > 0
    assert result.width % heads == 0
    assert abs(result.deviation_pct) < 5.0
    assert result.overrides["hidden_dim"] == result.width


def test_comparison_report_ranks_and_writes(quiet_logger, tmp_path):
    records = [
        {"model": "a", "parameters": 100, "width": 64, "size_deviation_pct": 1.0, "size_flags": [], "overfit_status": "PASS", "overfit_loss": 0.01, "max_batch": 64, "max_batch_peak_gb": 2.0, "max_batch_status": "PASS", "train_status": "DONE", "best_val_loss": 0.5, "duration_s": 10.0, "run_directory": "x", "metrics": {"euclidean_mean": 7.0, "mae": 3.0, "rmse": 4.0, "r2": 0.7, "median_error": 2.0, "euclidean_median": 6.0}},
        {"model": "b", "parameters": 110, "width": 70, "size_deviation_pct": -1.0, "size_flags": [], "overfit_status": "PASS", "overfit_loss": 0.02, "max_batch": 32, "max_batch_peak_gb": 2.5, "max_batch_status": "PASS", "train_status": "DONE", "best_val_loss": 0.4, "duration_s": 11.0, "run_directory": "y", "metrics": {"euclidean_mean": 5.0, "mae": 2.0, "rmse": 3.0, "r2": 0.8, "median_error": 1.0, "euclidean_median": 4.0}},
    ]

    config  = BenchmarkConfig()
    written = ComparisonReport(records, tmp_path, config, quiet_logger).write_all()

    assert [record["model"] for record in written["ranked"]] == ["b", "a"]
    assert Path(written["report_path"]).exists()
    assert Path(written["summary_path"]).exists()


def test_trial_collector_merges_stage_files(quiet_logger, tmp_path):
    (tmp_path / "size_match.json").write_text(json.dumps({"a": {"model": "a", "parameters": 100, "width": 64, "deviation_pct": 0.5, "flags": []}}), encoding="utf-8")
    (tmp_path / "training_results.json").write_text(json.dumps({"a": {"model": "a", "status": "DONE", "best_val_loss": 0.3, "run_directory": "r"}}), encoding="utf-8")
    (tmp_path / "evaluation_results.json").write_text(json.dumps({"a": {"euclidean_mean": 4.2, "mae": 1.1}}), encoding="utf-8")

    records = TrialCollector(tmp_path, ["a"], quiet_logger).collect()

    assert len(records) == 1
    assert records[0]["parameters"] == 100
    assert records[0]["train_status"] == "DONE"
    assert records[0]["metrics"]["euclidean_mean"] == 4.2


def test_benchmark_pipeline_runs(dataset_config, quiet_logger, tmp_path):
    keep   = {"graphsage", "gine"}
    config = BenchmarkConfig()

    config.skip_models                = tuple(model_name for model_name in MODEL_CONFIG_REGISTRY if model_name not in keep)
    config.size_match.reference_model = "graphsage"
    config.size_match.tolerance       = 0.20

    config.overfit.sample_count        = 8
    config.overfit.epochs              = 2
    config.overfit.batch_size          = 4
    config.overfit.require_convergence = False
    config.overfit.abort_on_fail       = False

    config.benchmark_training.epochs = 1

    config.dataset                     = dataset_config
    config.dataset.data.subset_fraction = 0.6

    config.training.loop.device               = "cpu"
    config.training.loop.batch_size           = 8
    config.training.loop.num_workers          = 0
    config.training.loop.use_amp              = False
    config.training.loop.pin_memory           = False
    config.training.loop.persistent_workers   = False
    config.training.loop.validation_frequency = 1
    config.training.io.log_base_dir           = tmp_path

    summary = BenchmarkPipeline(config, logger=quiet_logger).run()

    assert Path(summary["report_path"]).exists()
    assert Path(summary["summary_path"]).exists()
    assert {record["model"] for record in summary["ranked"]} == keep

    payload = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert len(payload["records"]) == 2
    for record in payload["records"]:
        assert record["train_status"] == "DONE"
        assert record["parameters"] is not None
        assert "euclidean_mean" in record["metrics"]
