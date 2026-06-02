# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Graph Neural Network pipeline for 3D vertex reconstruction of neutrino interaction events in Liquid Argon Time Projection Chambers (LArTPCs), applied to the DUNE photon detection system. Each detector event is a sparse graph over optical sensors; the model predicts the $(x, y, z)$ vertex coordinate.

## Commands

```bash
# Build the preprocessed graph dataset from raw CSVs
python main/create_dataset.py

# Train the model (logs and checkpoints written to runs/)
python main/train.py

# Hyperparameter search via Optuna
python main/tune.py

# Monitor training
tensorboard --logdir=runs/
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA GPU (recommended).

## Configuration

All hyperparameters live in `core/config.py` as nested `dataclass` objects under `GlobalConfig`. Nothing is passed via CLI flags — edit the dataclasses directly before running. Key groups:

| Group | Notable fields |
|---|---|
| `DataConfig` | `input_dir`, `max_files` |
| `GraphConfig` | `k_neighbors` (4), `bidirectional` (False) |
| `PreprocessingConfig` | `zscore_threshold` (9.0), `scale_factor` (0.1), `detection_efficiency` (0.02) |
| `ModelConfig` | `gps_hidden_dim` (128), `gps_num_layers` (4), `gps_heads` (4) |
| `TrainingConfig` | `epochs` (10), `batch_size` (4), `use_amp` (True) |
| `EarlyStoppingConfig` | `patience` (15) |
| `TuningConfig` | `n_trials` (5) |

## Architecture

The pipeline is: raw CSVs → preprocessing → KNN graph construction → GPS backbone → multi-scale pooling → hierarchical regression head → $(x, y, z)$.

**Data pipeline** (`core/data.py`, `core/dataset.py`):
- `DataLoader` discovers CSVs under `data/`, extracts ground-truth coordinates from filenames (`pmod3_{x}_{y}_{z}_...csv`), and parallelises file I/O via Ray.
- `DataProcessor` applies: spatial outlier detection (z-score + KD-tree), intensity scaling (×0.1), binomial detection efficiency simulation (2%), and log(1+x) transform.
- `Graph` builds a PyG `Data` object per event: 8-dimensional node features (sensor position, light intensity, hit flag, light fraction, distance to light-weighted centroid, local light density) and 7-dimensional edge features on a KNN graph (Euclidean distance, normalised displacement, inverse-square distance, light gradient, Sørensen-Dice similarity).
- `DatasetNormalizer` z-scores node features, edge attributes, and targets using training-split statistics only.

**Model** (`core/model.py`):
- `GraphBackbone`: linear input projection → stack of `GPSLayer`s (default 4). Each GPS layer runs GATv2 convolution (local) and multi-head self-attention (global) in parallel, fuses them with a learned sigmoid gate, and applies a FFN — all with DropPath residuals and LayerNorm.
- `MultiScalePool`: at each of 3 resolution levels, concatenates mean and max pool, then coarsens via SAGPool (ratio 0.5). The 3-level readouts are concatenated → 768-dim graph embedding.
- `HierarchicalRegressionHead`: predicts $\hat{x}$, then $\hat{y}$ conditioned on $\hat{x}$ via FiLM, then $\hat{z}$ conditioned on $(\hat{x}, \hat{y})$ via FiLM. Each coordinate uses a `CoordinateHead` MLP.

**Training** (`core/trainer.py`): AdamW with three parameter groups (head LR 1e-2, pooling LR 1e-3, backbone LR 1e-3), linear warmup (100 steps) → cosine annealing (250 epochs, η_min 1e-6), AMP, gradient clipping (norm 1.0), optional EMA (decay 0.9995). Loss: MSE in normalised space. Metrics in physical units: MAE, RMSE, R², median AE, Euclidean error.

**Tuning** (`core/tuner.py`): Optuna TPE with MedianPruner over ~72 hyperparameters spanning model, optimizer, and training config.

## Code conventions (enforced by `agents/general_rules.md`)

These are hard rules for this project — violations constitute a contract breach:

- **Class-based design only.** No core logic as free functions; no script-style pipelines without explicit class boundaries.
- **No comments of any kind.** All comments in any form are forbidden.
- **No abbreviations in variable names.** `lr`, `bs`, `cfg`, `tmp`, `opt`, `idx` and all similar shortenings are forbidden. Names must be fully spelled and semantically descriptive. Exception: standard mathematical symbols in self-contained formula contexts.
- **All logging through `core/logger.py`.** Never instantiate ad-hoc loggers. `rich` is mandatory for any long-running loop (dataset preprocessing, training epoch loop, evaluation, tuning).
- **Academic, formal, impersonal tone** in all textual outputs and markdown artifacts. No emojis, no slang, no conversational fillers.
- Markdown artifacts produced by agents go in a single shared directory — agents must not create new subdirectories for them.
