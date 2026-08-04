import json
from pathlib import Path

import pytest
import torch

from configuration import TrainEntryConfig
from tools import Logger
from tools.runtime.completion import CompletionMarker
from pipelines.training import TrainingPipeline


def _entry(parquet_store, tmp_path, model_name="graphsage"):
    entry = TrainEntryConfig()
    entry.model_name                     = model_name
    entry.dataset.data.parquet_store_dir = parquet_store
    entry.dataset.data.stats_sample_size = 16
    entry.training.io.log_base_dir       = tmp_path / "runs"
    entry.training.loop.device           = "cpu"
    entry.training.loop.epochs           = 1
    entry.training.loop.batch_size       = 8
    entry.training.loop.num_workers      = 0
    entry.training.warmup.warmup_steps   = 2
    return entry


def test_training_pipeline_one_epoch(parquet_store, tmp_path):
    entry = TrainEntryConfig()
    entry.model_name                          = "graphsage"
    entry.dataset.data.parquet_store_dir      = parquet_store
    entry.dataset.data.stats_sample_size      = 32
    entry.training.io.log_base_dir            = tmp_path / "runs"
    entry.training.loop.device                = "cpu"
    entry.training.loop.epochs                = 1
    entry.training.loop.batch_size            = 8
    entry.training.loop.num_workers           = 0
    entry.training.warmup.warmup_steps        = 2
    entry.training.loss.euclidean_weight      = 0.2
    entry.training.loss.containment_weight    = 0.1
    entry.training.loss.light_falloff_weight  = 0.5

    logger  = Logger(log_dir="", name="train_test", level="ERROR")
    summary = TrainingPipeline(entry, logger=logger).run()

    assert torch.isfinite(torch.tensor(summary["best_val_loss"]))
    assert len(list((tmp_path / "runs").rglob("best_model.pt"))) == 1


def test_training_tracks_per_coordinate_errors(parquet_store, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    entry = _entry(parquet_store, tmp_path)

    logger  = Logger(log_dir="", name="train_coord_errors", level="ERROR")
    summary = TrainingPipeline(entry, logger=logger).run()

    accumulator = EventAccumulator(str(Path(summary["run_directory"]) / "tensorboard"))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])

    assert {"val_mae/x", "val_mae/y", "val_mae/z"} <= tags
    assert {"val_rmse/x", "val_rmse/y", "val_rmse/z"} <= tags
    assert all(event.value >= 0.0 for event in accumulator.Scalars("val_mae/x"))


def test_training_pipeline_pna_injects_degree_histogram(parquet_store, tmp_path):
    entry = TrainEntryConfig()
    entry.model_name                     = "pna"
    entry.dataset.data.parquet_store_dir = parquet_store
    entry.dataset.data.stats_sample_size = 16
    entry.training.io.log_base_dir       = tmp_path / "runs"
    entry.training.loop.device           = "cpu"
    entry.training.loop.epochs           = 1
    entry.training.loop.batch_size       = 8
    entry.training.loop.num_workers      = 0
    entry.training.warmup.warmup_steps   = 2

    logger = Logger(log_dir="", name="train_pna", level="ERROR")
    TrainingPipeline(entry, logger=logger).run()

    assert "degree_histogram" in entry.model_overrides
    assert sum(entry.model_overrides["degree_histogram"]) > 0


def test_training_stamps_a_completion_marker(parquet_store, tmp_path):
    entry = _entry(parquet_store, tmp_path)

    logger  = Logger(log_dir="", name="train_marker", level="ERROR")
    summary = TrainingPipeline(entry, logger=logger).run()

    marker = CompletionMarker.read(summary["run_directory"])
    assert marker["stage"] == "training"
    assert marker["epochs_completed"] == 1


def test_overfit_gate_writes_a_passing_report(parquet_store, tmp_path):
    entry = _entry(parquet_store, tmp_path)
    entry.training.overfit_check.enabled         = True
    entry.training.overfit_check.n_examples      = 2
    entry.training.overfit_check.max_steps       = 40
    entry.training.overfit_check.steps_per_epoch = 10
    entry.training.overfit_check.pass_loss_ratio = 1.0

    logger  = Logger(log_dir="", name="train_overfit", level="ERROR")
    summary = TrainingPipeline(entry, logger=logger).run()

    report = json.loads((tmp_path / "runs").rglob("overfit_report.json").__next__().read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["n_examples"] == 2
    assert report["steps_run"] == 40
    assert report["sanitized_overrides"]["scheduler.type"] == "constant"
    assert report["best_loss"] <= report["initial_loss"]
    assert not (tmp_path / "runs" / summary["run_directory"] / "overfit_check").exists()


def test_overfit_gate_aborts_training_when_the_ratio_is_unreachable(parquet_store, tmp_path):
    entry = _entry(parquet_store, tmp_path)
    entry.training.overfit_check.enabled         = True
    entry.training.overfit_check.n_examples      = 2
    entry.training.overfit_check.max_steps       = 4
    entry.training.overfit_check.steps_per_epoch = 2
    entry.training.overfit_check.pass_loss_ratio = 1e-9
    entry.training.overfit_check.stop_threshold  = 1e-12

    logger = Logger(log_dir="", name="train_overfit_fail", level="ERROR")

    with pytest.raises(RuntimeError, match="Overfit check failed"):
        TrainingPipeline(entry, logger=logger).run()

    assert not list((tmp_path / "runs").rglob("checkpoints/best_model.pt"))


def test_validation_frequency_skips_intermediate_epochs(parquet_store, tmp_path):
    entry = _entry(parquet_store, tmp_path)
    entry.training.loop.epochs               = 3
    entry.training.loop.validation_frequency = 3

    logger  = Logger(log_dir="", name="train_valfreq", level="ERROR")
    summary = TrainingPipeline(entry, logger=logger).run()

    assert len(summary["val_losses"]) == 3
    assert all(torch.isnan(torch.tensor(loss)) for loss in summary["val_losses"][:2])
    assert torch.isfinite(torch.tensor(summary["val_losses"][2]))


def test_ema_produces_a_finite_validation_loss(parquet_store, tmp_path):
    entry = _entry(parquet_store, tmp_path)
    entry.training.loop.use_ema   = True
    entry.training.loop.ema_decay = 0.9

    logger  = Logger(log_dir="", name="train_ema", level="ERROR")
    summary = TrainingPipeline(entry, logger=logger).run()

    assert torch.isfinite(torch.tensor(summary["best_val_loss"]))


def test_ema_checkpoint_stores_the_evaluated_shadow_weights(parquet_store, tmp_path):
    entry = _entry(parquet_store, tmp_path)
    entry.training.loop.use_ema   = True
    entry.training.loop.ema_decay = 0.9

    logger = Logger(log_dir="", name="train_ema_ckpt", level="ERROR")
    TrainingPipeline(entry, logger=logger).run()

    best_path = next((tmp_path / "runs").rglob("best_model.pt"))
    last_path = next((tmp_path / "runs").rglob("last.pt"))
    best      = torch.load(best_path, map_location="cpu", weights_only=False)
    state     = torch.load(last_path, map_location="cpu", weights_only=False)
    shadow    = state["ema"]["shadow"]

    assert all(torch.equal(best["params"][name], tensor) for name, tensor in shadow.items())
    assert any(not torch.equal(state["model"][name], tensor) for name, tensor in shadow.items())


def test_resume_continues_from_the_saved_trainer_state(parquet_store, tmp_path):
    entry = _entry(parquet_store, tmp_path)
    entry.run_name = "resume_case"

    logger  = Logger(log_dir="", name="train_resume_first", level="ERROR")
    summary = TrainingPipeline(entry, logger=logger).run()

    assert (tmp_path / "runs" / "graphsage_resume_case" / "last.pt").is_file()

    resumed = _entry(parquet_store, tmp_path)
    resumed.run_name             = "resume_case"
    resumed.training.loop.epochs = 3
    resumed.training.loop.resume = True

    logger  = Logger(log_dir="", name="train_resume_second", level="ERROR")
    summary = TrainingPipeline(resumed, logger=logger).run()

    assert len(summary["train_losses"]) == 3
