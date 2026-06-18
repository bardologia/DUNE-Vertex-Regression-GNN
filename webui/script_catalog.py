from __future__ import annotations

from project_paths import ProjectPaths


class ScriptCatalog:

    METADATA = {
        "train": {
            "label"    : "Train",
            "category" : "Training",
            "purpose"  : "Train one GNN model end to end, logging checkpoints and metrics to runs/.",
        },
        "tune": {
            "label"    : "Tune",
            "category" : "Training",
            "purpose"  : "Run the Optuna hyperparameter search over the model and training configuration.",
        },
        "cross_validate": {
            "label"    : "Cross-Validate",
            "category" : "Training",
            "purpose"  : "Run stratified K-fold cross-validation, aggregating held-out test metrics across folds.",
        },
        "benchmark": {
            "label"    : "Benchmark",
            "category" : "Training",
            "purpose"  : "Benchmark the whole model zoo: capacity matching, overfit gate, max-batch probe, training, and a ranked comparison report.",
        },
        "correct_coordinates": {
            "label"    : "Correct Coordinates",
            "category" : "Data",
            "purpose"  : "Correct ground-truth vertex coordinates extracted from raw event files.",
        },
        "augment_octants": {
            "label"    : "Augment Octants",
            "category" : "Data",
            "purpose"  : "Generate octant-reflected augmentations of the corrected event frames.",
        },
        "build_parquet_store": {
            "label"    : "Build Parquet Store",
            "category" : "Data",
            "purpose"  : "Materialise the event frames into a partitioned Parquet store.",
        },
        "infer": {
            "label"    : "Infer",
            "category" : "Analysis",
            "purpose"  : "Run the full inference analysis (metrics, plots, animations, report) on a training run.",
        },
    }

    ORDER = [
        "correct_coordinates",
        "augment_octants",
        "build_parquet_store",
        "train",
        "infer",
        "tune",
        "cross_validate",
        "benchmark",
    ]

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def get(self, key: str) -> dict | None:
        entry = self.paths.script_entry(key)
        if entry is None or not entry["path"].exists():
            return None

        metadata = self.METADATA.get(key, {"label": key, "category": "Other", "purpose": ""})

        return {
            "key"          : key,
            "file"         : entry["relative"],
            "label"        : metadata["label"],
            "category"     : metadata["category"],
            "purpose"      : metadata["purpose"],
            "has_config"   : entry["has_config"],
            "config_class" : entry["entry_config_class"],
        }

    def list(self) -> list[dict]:
        scripts = []
        for key in self.ORDER:
            description = self.get(key)
            if description is not None:
                scripts.append(description)
        return scripts
