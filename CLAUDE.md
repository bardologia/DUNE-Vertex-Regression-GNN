# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Graph Neural Network pipeline for 3D vertex reconstruction of neutrino interaction events in Liquid Argon Time Projection Chambers (LArTPCs), applied to the DUNE photon detection system. Each detector event is a sparse graph over optical sensors; the model predicts the $(x, y, z)$ vertex coordinate.

The codebase is an installable package (`pyproject.toml`, package name `dune-gnn`) organised after the DLR-TomoSAR layout: `configuration/`, `models/`, `pipelines/`, `tools/`, `webui/`, `main/`, `tests/`.

## Commands

```bash
pip install -e .                                          # editable install (run once, re-run after adding a new top-level package)

python main/correct_coordinates.py                        # raw CSVs -> coordinate-corrected CSVs (sensor frame)
python main/build_parquet_store.py                        # corrected CSVs -> Parquet store (geometry/events/octants)
python main/create_dataset.py                             # Parquet store -> chunked graph dataset
python main/train.py  --model_name gps                    # train a model (loss-only loop; checkpoints best on val loss)
python main/infer.py  --run_directory runs/gps_<stamp>    # full analysis: metrics, plots, GIFs, report on a run
python main/tune.py   --model_name gps                    # Optuna search over a model's hyperparameters
python webui/serve.py                                     # web control panel + server monitoring
```

Training computes only the loss per epoch for speed; the full evaluation (per-coordinate metrics, publication plots, rotating 3D GIFs, markdown report, `metadata/metrics.json`) is produced by `main/infer.py` (or automatically when `--infer_after true` is passed to `train.py`). The model zoo includes hierarchical cascade variants (`gps_cascade`, `gatv2_cascade`, `gine_cascade`) whose head predicts each coordinate conditioned on the previously predicted ones.

All runtime configuration is passed through `ConfigCli` overrides of the form `--dotted.path value` (e.g. `--training.loop.epochs 50 --training.optimizer.learning_rate_encoder 1e-3 --model_overrides "{'hidden_dim': 192}"`). Run any entry script with `--help-config` to list its editable fields. Defaults live in the dataclasses, never as module-level values.

## Architecture

Pipeline: raw CSVs -> coordinate correction -> Parquet store (+ octant augmentation) -> preprocessing -> KNN graph construction -> GNN encoder -> pooling -> regression head -> $(x, y, z)$.

### configuration/
Dataclass config tree introspected by `ConfigCli`:
- `data/` — `DataConfig` (source `parquet`|`csv`, parquet store path, octant augmentation), `GraphConfig`, `PreprocessingConfig`, `SplitConfig`, `RayConfig`, aggregated `DatasetConfig`.
- `training/general.py` — `OptimizerConfig` (three learning-rate groups), `SchedulerConfig`, `WarmupConfig`, `EarlyStoppingConfig`, `EMAConfig`, `GradientClipperConfig`, `OverfitConfig`, `LossConfig`, `TrainingLoopConfig`, aggregated `TrainingConfig`. Field names match the `tools/training` components.
- `architectures/zoo.py` — `BaseGNNConfig` plus one config dataclass per model and `MODEL_CONFIG_REGISTRY`.
- `tuning/` — `TuningConfig`.
- `entry/` — `DatasetEntryConfig`, `TrainEntryConfig`, `TuneEntryConfig` (what the entry scripts and the webui edit).

### models/
`blocks.py` holds the shared parts (DropPath, FeedForward, GPS layer/encoder, generic `MessagePassingEncoder`, mean-max / multiscale-SAGPool / Set2Set pooling, FiLM hierarchical and MLP heads) and the `GraphRegressor` base, which wraps `encoder -> pool -> norm -> regression_head` and exposes those three submodules for the optimizer parameter groups. Each architecture is one file (`gps`, `gps_lite`, `gatv2`, `gine`, `graphsage`, `pna`, `gcn`, `transformer_conv`, `edgeconv`, `general_conv`, `res_gated`, `supergat`). `models/__init__.py` defines `MODEL_REGISTRY` and `get_model(name, config=None, **overrides)`.

### pipelines/
- `dataset/` — `coordinate_correction.py` (`CoordinateTransform`, `DatasetCorrector`, `DatasetAugmentor`), `parquet_store.py` (`ParquetDatasetWriter`, `ParquetEventReader`), `parquet_source.py` (`ParquetGraphSource`), `loading.py` (CSV `DataLoader`), `preprocessing.py` (`DataProcessor`), `graph.py` (`Graph`), `ray_executor.py` (shared `RayExecutor`), `builder.py` (`DatasetBuilder`), `normalization.py` (`Normalizer`, `NormalizationStats`), `splitting.py` (`TargetBalancer`), `pipeline.py` (`DatasetPipeline`).
- `shared/run_metadata.py` — `TrainingRunMetadata` (run directories, tensorboard writer, tracker).
- `training/` — `Loss`, `Metrics`, `Trainer`, `TrainingPipeline`.
- `tuning/` — `Tuner`.

### tools/
Self-contained, copy-adapted from DLR-TomoSAR. `monitoring/` (`Logger`, `Tracker`, `ResourceMonitor`, `ShapeLogger`, `ModelSummary`), `training/` (`Warmup`, `Scheduler`, `EarlyStopping`, `GradientClipper`, `Checkpoint`, `MetricAggregator`, `ExponentialMovingAverage`), `runtime/` (`ConfigCli`, `Reproducibility`), `reporting/` (markdown, plotting). Always log through `tools.monitoring.Logger`.

### webui/
Stdlib `ThreadingHTTPServer` + `RequestRouter` over component libraries (scripts catalog, config registry, process manager, system + GPU monitor, tensorboard manager, results browser, model library) with a hash-routed vanilla-JS SPA. Control panel for launching the entry scripts and monitoring the server. No animations or documentation pages.

## Data substrate

The Parquet store (`pipelines/dataset/parquet_store.py`) holds canonical sensor geometry once, per-event light vectors, and an octant index for full-volume reflection augmentation. `ParquetEventReader.iterate_corrected()` / `iterate_augmented()` yield event frames (`bin, x, y, z, ly_amount`) that the `Graph` builder consumes. Coordinate correction maps filename targets into the sensor frame.

## Node and edge features

8-D node features: sensor position $(x, y, z)$, light intensity, hit flag, light fraction, distance to the light-weighted centroid, local light density. 7-D edge features on the KNN graph: Euclidean distance, normalised 3-D displacement, inverse-square distance, light gradient, Sorensen-Dice light similarity. Targets are z-scored with training-split statistics; metrics are reported in physical units (MAE, RMSE, R^2, median AE, Euclidean error).

## Code conventions (hard rules)

- Class-based design only; no core logic in free functions; an orchestrator method sequences the steps and is the last method in its class.
- No comments of any kind; no docstrings.
- No abbreviations in variable names; names are fully spelled and descriptive.
- All logging through `tools.monitoring.Logger`; never `print` or ad-hoc loggers.
- Academic, formal, impersonal tone in all textual output; no emojis.
- No backward compatibility or silent fallbacks; readers and writers move to the new form only and break loudly. Regenerate old artifacts rather than supporting them.
- Vertically align runs of `=` and `:`; separate logical blocks with blank lines; one statement per line.
- Defaults live in config dataclasses; entry points apply `ConfigCli` overrides for single runs and override functions for batch loops. Entry-point bootstrap is the only place `sys.path`/environment pinning is allowed.
