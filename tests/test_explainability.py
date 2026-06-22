import json
from pathlib import Path

from configuration                 import DatasetConfig, TrainEntryConfig
from configuration.explainability  import FeatureImportanceConfig
from tools                         import Logger
from pipelines.dataset             import FeatureLayout
from pipelines.training            import TrainingPipeline
from pipelines.explainability      import FeatureImportancePipeline


def test_feature_layout_matches_default_dimensions():
    layout = FeatureLayout(DatasetConfig())

    assert len(layout.node_features()) == 17
    assert len(layout.edge_features()) == 23

    node_groups = FeatureLayout.group_index(layout.node_features())
    assert node_groups["position"] == [0, 1, 2]
    assert node_groups["direction_to_centroid"]  == [7, 8, 9]
    assert node_groups["direction_to_brightest"] == [10, 11, 12]


def test_feature_layout_respects_toggles():
    config                                = DatasetConfig()
    config.graph.inertia_features         = False
    config.graph.rank_features            = False
    config.graph.edge_rbf_count           = 0

    layout = FeatureLayout(config)

    assert len(layout.node_features()) == 13
    assert len(layout.edge_features()) == 7
    assert all(name != "intensity_rank" for name, _ in layout.node_features())


def test_feature_importance_pipeline_outputs(parquet_store, tmp_path):
    entry = TrainEntryConfig()
    entry.model_name                     = "graphsage"
    entry.dataset.data.parquet_store_dir = parquet_store
    entry.dataset.data.stats_sample_size = 32
    entry.training.io.log_base_dir       = tmp_path / "runs"
    entry.training.loop.device           = "cpu"
    entry.training.loop.epochs           = 1
    entry.training.loop.batch_size       = 8
    entry.training.loop.num_workers      = 0
    entry.training.loop.verbose          = False
    entry.training.warmup.warmup_steps   = 2

    summary       = TrainingPipeline(entry, logger=Logger(log_dir="", name="train", level="ERROR")).run()
    run_directory = summary["run_directory"]

    config                     = FeatureImportanceConfig()
    config.run_directory       = Path(run_directory)
    config.device              = "cpu"
    config.split               = "test"
    config.batch_size          = 8
    config.max_events          = 12
    config.permutation_repeats = 2
    config.eg_samples          = 4
    config.eg_events           = 8
    config.shap_nsamples       = 32
    config.shap_events         = 4
    config.top_k               = 5

    output_directory = FeatureImportancePipeline(config, logger=Logger(log_dir="", name="explain", level="ERROR")).run()

    assert (output_directory / "report.md").exists()
    assert (output_directory / "results.json").exists()
    assert (output_directory / "node_importance.csv").exists()
    assert (output_directory / "edge_importance.csv").exists()
    assert any((output_directory / "plots").glob("permutation_*.png"))
    assert any((output_directory / "plots").glob("gradient_*.png"))
    assert any((output_directory / "plots").glob("expected_gradients_*.png"))
    assert any((output_directory / "plots").glob("kernel_shap_*.png"))

    payload = json.loads((output_directory / "results.json").read_text(encoding="utf-8"))
    assert "expected_gradients" in payload["analysis"]
    assert set(payload["analysis"]["expected_gradients"]["completeness"]) == {"x", "y", "z"}

    kernel_shap = payload["analysis"]["kernel_shap"]
    assert len(kernel_shap["node"]) == 17
    assert all(abs(kernel_shap["completeness"][axis] - 1.0) < 0.1 for axis in ("x", "y", "z"))
