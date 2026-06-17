from pathlib import Path

from configuration import TrainEntryConfig
from tools import Logger
from pipelines.training import TrainingPipeline
from pipelines.inference import InferencePipeline


def test_inference_pipeline_outputs(tiny_dataset, tmp_path):
    entry = TrainEntryConfig()
    entry.model_name                         = "graphsage"
    entry.dataset_dir                        = tiny_dataset
    entry.training.io.log_base_dir           = tmp_path / "runs"
    entry.training.loop.device               = "cpu"
    entry.training.loop.epochs               = 1
    entry.training.loop.batch_size           = 8
    entry.training.loop.num_workers          = 0
    entry.training.loop.verbose              = False
    entry.training.warmup.warmup_steps       = 2

    summary = TrainingPipeline(entry, logger=Logger(log_dir="", name="train", level="ERROR")).run()
    run_directory = summary["run_directory"]

    results = InferencePipeline(run_directory, logger=Logger(log_dir="", name="infer", level="ERROR"), device="cpu", splits=("test",), batch_size=8).run()

    assert "test" in results
    assert "euclidean_mean" in results["test"]["metrics"]
    assert "mae_x" in results["test"]["metrics"]

    analysis_directory = Path(run_directory) / "analysis" / "test"
    assert any(analysis_directory.glob("*.png"))
    assert any(analysis_directory.glob("*.gif"))
    assert (Path(run_directory) / "metadata" / "metrics.json").exists()
