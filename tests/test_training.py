import torch

from configuration import TrainEntryConfig
from tools import Logger
from pipelines.training import TrainingPipeline


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
    entry.training.loop.verbose               = False
    entry.training.warmup.warmup_steps        = 2
    entry.training.loss.euclidean_weight      = 0.2
    entry.training.loss.containment_weight    = 0.1
    entry.training.loss.light_falloff_weight  = 0.5

    logger  = Logger(log_dir="", name="train_test", level="ERROR")
    summary = TrainingPipeline(entry, logger=logger).run()

    assert torch.isfinite(torch.tensor(summary["final_test_loss"]))
    assert torch.isfinite(torch.tensor(summary["final_validation_loss"]))
    assert len(list((tmp_path / "runs").rglob("best_model.pt"))) == 1


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
    entry.training.loop.verbose          = False
    entry.training.warmup.warmup_steps   = 2

    logger = Logger(log_dir="", name="train_pna", level="ERROR")
    TrainingPipeline(entry, logger=logger).run()

    assert "degree_histogram" in entry.model_overrides
    assert sum(entry.model_overrides["degree_histogram"]) > 0
