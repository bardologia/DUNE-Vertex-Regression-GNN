from configuration import TuneEntryConfig
from tools import Logger
from pipelines.tuning import Tuner


def test_tuner_runs_two_trials(parquet_store, tmp_path):
    entry = TuneEntryConfig()
    entry.model_name                     = "graphsage"
    entry.dataset.data.parquet_store_dir = parquet_store
    entry.dataset.data.stats_sample_size = 32
    entry.training.io.tuner_base_dir     = tmp_path / "tuning"
    entry.training.loop.device           = "cpu"
    entry.training.loop.num_workers      = 0
    entry.training.warmup.warmup_steps   = 2
    entry.tuning.n_trials                = 2
    entry.tuning.startup_trials          = 1
    entry.tuning.epochs                  = 1
    entry.tuning.batch_size              = 8
    entry.tuning.warmup_trials           = 0
    entry.tuning.warmup_steps            = 0

    tuner = Tuner(entry, Logger(log_dir="", name="tune", level="ERROR"))
    study = tuner.optimize()

    assert len(study.trials) == 2

    results_path = tmp_path / "tuning" / entry.model_name / "best_trial.json"
    assert results_path.exists()

    model_overrides, optimizer_overrides = tuner.build_best_overrides()
    assert isinstance(model_overrides, dict)
    assert "learning_rate_encoder" in optimizer_overrides
