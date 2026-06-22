<div align="center">

# DUNE-GNN

### Graph Neural Networks for 3D Interaction-Vertex Reconstruction in Liquid Argon Time Projection Chambers

*A geometric deep-learning pipeline for reconstructing the spatial origin of neutrino interactions from the photon detection system of the Deep Underground Neutrino Experiment (DUNE).*

<br>

![Python](https://img.shields.io/badge/python-≥3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-graph_learning-3C2179)
![Optuna](https://img.shields.io/badge/Optuna-hyperparameter_search-0E4C92)
![Domain](https://img.shields.io/badge/domain-neutrino_physics-6A1B9A)

</div>

---

## Abstract

The reconstruction of the interaction vertex — the three-dimensional point $(x, y, z)$ at which a neutrino interacts within a detector — is a foundational task in accelerator and astroparticle neutrino physics. In a Liquid Argon Time Projection Chamber (LArTPC), the prompt scintillation light produced by a charged-particle cascade is observed by a sparse array of optical sensors distributed over the detector volume. Recovering the vertex from this sparse, irregular, and noisy optical signal is an ill-posed inverse problem that does not admit a fixed-grid, image-based representation.

**DUNE-GNN** formulates vertex reconstruction as a *graph-level regression* problem. Each detector event is encoded as a sparse geometric graph over the optical sensors, in which nodes carry per-sensor photometric and positional features and edges encode local spatial and photometric relationships. A hybrid local–global graph transformer backbone, a multi-scale hierarchical pooling stage, and a conditionally-factorised regression head jointly map the event graph to a continuous coordinate prediction. The pipeline is fully configuration-driven, reproducible, and instrumented for large-scale hyperparameter optimisation.

---

## 1. Scientific Motivation

The Deep Underground Neutrino Experiment (DUNE) is a next-generation long-baseline neutrino oscillation experiment whose far detector comprises multiple large LArTPC modules instrumented with a **Photon Detection System (PDS)**. When a neutrino interacts in the liquid argon, the resulting particles deposit energy and produce a prompt flash of vacuum-ultraviolet scintillation light. This light propagates through the volume and is registered by optical sensors with characteristic arrival intensities that depend on the geometry between the interaction point and each sensor.

Because the optical response is governed by solid-angle, attenuation, and Rayleigh-scattering effects, the *pattern* of light across the sensor array encodes the spatial position of the interaction. The reconstruction problem is therefore naturally posed on the **geometry of the detector itself**, rather than on a rasterised image: sensors are spatially irregular, the active set per event is sparse, and the signal is permutation-invariant with respect to sensor indexing. These properties make graph neural networks a principled inductive bias for the task.

---

## 2. Problem Formulation

Let an event be observed as a set of optical sensors $\{s_i\}_{i=1}^{n}$, each with a fixed spatial position $\mathbf{r}_i \in \mathbb{R}^3$ and a measured light response $\ell_i \in \mathbb{R}_{\geq 0}$. We construct an undirected geometric graph $G = (V, E)$ and learn a function

$$
f_\theta : G \longmapsto \hat{\mathbf{v}} = (\hat{x}, \hat{y}, \hat{z}) \in \mathbb{R}^3
$$

that regresses the interaction vertex. The model is optimised by minimising the mean squared error between predictions and ground-truth vertices in a normalised coordinate space, while all reported performance metrics are expressed in physical units.

Ground-truth coordinates are derived from the simulation file naming convention `pmod3_{x}_{y}_{z}_..._N.csv`, allowing the supervised target to be recovered directly from the dataset without an auxiliary label store. The parsed vertex is then mapped from the simulation frame into the sensor frame via a fixed `CoordinateTransform`, so that targets and sensor positions share a single coordinate system.

---

## 3. Method

The end-to-end pipeline is summarised as:

```
   raw CSV events
        │
        ▼
  preprocessing            outlier rejection · intensity scaling · detection-efficiency simulation · log(1+x)
        │
        ▼
  graph construction       active-sensor pruning · k-nearest-neighbour graph · 17-D node features · 23-D edge features
        │
        ▼
  GPS backbone             stacked local (GATv2) + global (self-attention) layers with gated fusion
        │
        ▼
  multi-scale pooling      3-level mean/max readout + SAGPool coarsening → graph embedding
        │
        ▼
  hierarchical head        x → (y | x) → (z | x, y)  via FiLM conditioning
        │
        ▼
   vertex (x, y, z)
```

### 3.1 Graph Representation

Each event is materialised as a PyTorch Geometric `Data` object on a $k$-nearest-neighbour topology over sensor positions. Only active sensors (positive realized light) enter the graph, with an optional cap on the brightest sensors, so a full 6912-channel detector collapses to roughly 250 nodes per event and a usable batch size becomes possible:

- **Node features (17-dimensional at default settings):** sensor position, measured light intensity, the per-sensor light fraction, the distance to the light-weighted centroid of the event, a measure of local light density, the normalised direction to the centroid, and per-node inertia and light-rank descriptors. The exact dimensionality follows `BaseGNNConfig.input_dim` and the `GraphConfig` feature toggles (`direction_features`, `inertia_features`, `rank_features`).
- **Edge features (23-dimensional at default settings):** Euclidean separation, normalised displacement vector, inverse-square distance, light gradient along the edge, a Sørensen–Dice photometric similarity, and a 16-bin radial-basis expansion of the edge distance (`GraphConfig.edge_rbf_count`). The exact dimensionality follows `BaseGNNConfig.edge_dim`.

This explicitly injects the detector geometry and the photometric structure of the flash into the learned representation.

### 3.2 Local–Global Backbone (GPS)

The backbone is a stack of **GPS layers**, each of which processes the graph through two complementary channels evaluated in parallel:

- a **local** channel based on **GATv2** message passing, which aggregates information from a node's immediate geometric neighbourhood; and
- a **global** channel based on **multi-head self-attention**, which captures long-range dependencies across the entire sensor array.

The two channels are combined through a **learned sigmoid gate**, followed by a position-wise feed-forward network. Each sub-block uses residual connections with stochastic depth (DropPath) and layer normalisation, yielding a stable, expressive representation that balances local detector geometry against global event structure.

### 3.3 Multi-Scale Hierarchical Pooling

To produce a fixed-dimensional graph embedding that is robust to event size, the model applies **three levels** of hierarchical pooling. At each level, mean- and max-pooled readouts are concatenated and the graph is then coarsened with **SAGPool** (Self-Attention Graph Pooling). The per-level readouts are concatenated to form the final graph embedding, summarising the event at progressively coarser spatial resolutions.

### 3.4 Conditionally-Factorised Regression Head

Rather than predicting the three coordinates independently, the head exploits their conditional structure. It predicts $\hat{x}$ first, then $\hat{y}$ conditioned on $\hat{x}$, and finally $\hat{z}$ conditioned on $(\hat{x}, \hat{y})$, where conditioning is implemented through **FiLM** (Feature-wise Linear Modulation). Each coordinate is produced by a dedicated regression MLP, allowing the network to model dependencies between spatial axes.

### 3.5 Optimisation

Training uses **AdamW** with three discriminative parameter groups (regression head, pooling stage, and backbone) under a **linear-warmup → cosine-annealing** learning-rate schedule. The procedure further supports automatic mixed precision, gradient-norm clipping, an optional exponential moving average of the weights, and early stopping. Performance is reported in physical units via mean absolute error, root-mean-square error, the coefficient of determination $R^2$, median absolute error, and the mean Euclidean vertex error.

---

## 4. Repository Structure

```
DUNE-GNN/
├── main/                     # thin entry points (config → pipeline); only place that touches sys.path
│   ├── build_parquet_store.py  # build the geometry-once parquet store from raw CSVs
│   ├── train.py                # training entry point (logs + checkpoints → runs/)
│   ├── infer.py                # inference / evaluation on a trained checkpoint
│   ├── explain.py              # graph feature-importance attribution on a trained checkpoint
│   ├── cross_validate.py       # k-fold cross-validation
│   ├── tune.py                 # Optuna hyperparameter search
│   ├── sweep.py                # energy/efficiency robustness sweep on a trained run
│   ├── benchmark.py            # throughput / batch-size benchmarking
│   ├── export_events.py        # export raw/preprocessed events
│   ├── export_dataset_events.py
│   └── _bootstrap.py           # environment pinning (GPU, sys.path)
├── configuration/            # nested dataclass configuration, composed into per-entry configs
│   ├── data/                   # DataConfig, GraphConfig, PhysicsConfig, SplitConfig, HotChannelConfig
│   ├── architectures/          # per-model config dataclasses + MODEL_CONFIG_REGISTRY (zoo)
│   ├── training/               # TrainingLoopConfig, OptimizerConfig, SchedulerConfig, LossConfig, ...
│   ├── tuning/                 # TuningConfig (Optuna search budget)
│   ├── cross_validation/       # cross-validation config
│   ├── benchmark/              # benchmark config
│   └── entry/                  # TrainEntryConfig, InferEntryConfig, TuneEntryConfig, ...
├── models/                   # GraphRegressor, shared blocks, GPS backbone + the conv zoo
├── pipelines/                # the actual logic
│   ├── dataset/                # parquet store, PyG dataset, normalisation (train-split stats only), graph build, augmentation
│   ├── training/               # training loop, loss, pipeline orchestration
│   ├── inference/              # predictor, metrics, plots, animations, reporting
│   ├── cross_validation/       # k-fold orchestration
│   ├── tuning/                 # Optuna TPE + MedianPruner search
│   ├── benchmark/              # benchmark stages, sizing, batch probe
│   └── shared/                 # run metadata / run paths
├── tools/                    # reusable components (Logger, monitors, training utilities, runtime/config CLI)
│   ├── monitoring/             # logger, resource monitor, tracker, inspection
│   ├── training/               # scheduling, checkpoint, gradients, early stopping
│   ├── reporting/              # markdown reporting
│   └── runtime/                # ConfigCli, reproducibility
├── webui/                    # local web UI for launching/monitoring runs and browsing results
├── tests/                    # pytest suite (driven by the Dune conda env)
├── data/                     # raw simulated events (pmod3_*.csv)
├── data_frames/              # generated parquet store (gitignored)
├── runs/                     # training/inference run outputs (gitignored)
└── pyproject.toml
```

---

## 5. Installation

```bash
pip install -r requirements.txt
```

**Requirements.** Python ≥ 3.11 and a CUDA-capable GPU (recommended). Core dependencies (pinned in `pyproject.toml`) include PyTorch, PyTorch Geometric, scikit-learn, NumPy/SciPy/pandas, PyArrow, Optuna (hyperparameter optimisation), TensorBoard, and `rich`. Raw-CSV ingestion is parallelised with the standard-library `ProcessPoolExecutor` (worker count via `DataConfig.store_worker_count`).

---

## 6. Usage

```bash
# 1. Build the geometry-once parquet store from the raw CSV events
python main/build_parquet_store.py

# 2. Train the model (checkpoints and logs are written under runs/)
python main/train.py

# 3. Run inference / evaluation on a trained checkpoint
python main/infer.py

# 4. Run k-fold cross-validation
python main/cross_validate.py

# 5. Run a hyperparameter search with Optuna
python main/tune.py

# 6. Sweep light scale factor and detection efficiency on a trained run
python main/sweep.py --run_directory runs/<model>_<stamp> \
    --scale.minimum 0.05 --scale.maximum 0.5 --scale.step 0.05 \
    --efficiency.minimum 0.01 --efficiency.maximum 0.1 --efficiency.step 0.01

# 7. Attribute graph node/edge feature importance on a trained run
python main/explain.py --run_directory runs/<model>_<stamp>

# 8. Monitor training in real time
tensorboard --logdir=runs/
```

The **energy/efficiency sweep** (`main/sweep.py`) re-evaluates a trained checkpoint on its own held-out split while regenerating the optical signal under a Cartesian grid of light scale factors (`PhysicsConfig.scale_factor`) and binomial detection efficiencies (`PhysicsConfig.detection_efficiency`). The split membership and the training-split normalisation statistics are held fixed, so the sweep isolates the model's robustness to a shift in the photon-counting regime. Each axis is given as `minimum`, `maximum`, and `step`; every grid cell runs a full inference pass. Outputs are written under `runs/<run>/sweeps/energy_efficiency_<stamp>/`: a `report.md`, machine-readable `results.json` / `results.csv`, and per-metric heatmaps plus marginal line plots in `plots/`.

The **feature-importance attribution** (`main/explain.py`) explains which components of the event-graph representation a trained checkpoint relies on, evaluated on its own held-out split with the training-split normalisation statistics held fixed. It reports four complementary attributions over the node and edge feature columns (individually and grouped by semantic block): **permutation importance** — the held-out degradation when a feature is shuffled across every node or edge, averaged over `--permutation_repeats` seeds with a standard deviation; **occlusion (mean-ablation) importance** — the degradation when a feature is replaced by its dataset mean; **gradient saliency** — the mean magnitude of the gradient of the squared prediction error with respect to each normalised input feature, alongside its gradient×input variant; and **expected gradients** — a SHAP-consistent, axiomatic attribution that integrates the gradient of each predicted coordinate along paths from data-distribution baselines, reported per coordinate `(x, y, z)` with a completeness check (`--eg_samples`, `--eg_events`). Outputs are written under `runs/<run>/explainability/feature_importance_<stamp>/`: a `report.md` with ranked tables and a cross-method consensus ranking, machine-readable `results.json`, per-feature `node_importance.csv` / `edge_importance.csv`, and one publication-quality horizontal bar chart per method in `plots/`. Use `--max_events` to cap the evaluation set for a fast pass and `--split` to target `val`/`train`/`test`.

Every entry point wraps its configuration in a `ConfigCli`, so any nested config field can be overridden from the command line without editing source — for example:

```bash
python main/train.py --model_name gps --training.loop.epochs 50 --training.loop.batch_size 32
python main/tune.py   --model_name pna --tuning.n_trials 50
```

---

## 7. Configuration

The project is **configuration-driven**: every hyperparameter is declared as a field on a nested `dataclass` under `configuration/`, composed into a per-entry config (`TrainEntryConfig`, `InferenceEntryConfig`, `TuneEntryConfig`, `CrossValidationEntryConfig`, ...) in `configuration/entry/general.py`. Defaults can be edited in source, but each entry point also exposes the full nested tree through a `ConfigCli`, so any leaf can be overridden from the command line as `--<dotted.path> <value>` (run any entry with `--help-config` to list every overridable leaf). The principal groups are:

| Group (module) | Representative fields |
|---|---|
| `DataConfig` (`configuration/data`) | `raw_input_dir`, `parquet_store_dir`, `augment_octants`, `subset_fraction`, `stats_sample_size` |
| `GraphConfig` (`configuration/data`) | `k_neighbors`, `bidirectional`, `edge_rbf_count`, `direction_features` |
| `PhysicsConfig` (`configuration/data`) | `scale_factor`, `detection_efficiency`, `efficiency_seed` |
| `HotChannelConfig` (`configuration/data`) | `active_fraction`, `median_factor`, `neighbor_count` |
| Model config (`configuration/architectures`, per model in `MODEL_CONFIG_REGISTRY`) | `hidden_dim`, `num_layers`, `heads`, `pooling`, `head_type` |
| `TrainingLoopConfig` (`configuration/training`) | `epochs`, `batch_size`, `use_amp`, `gradient_accumulation_steps` |
| `OptimizerConfig` / `SchedulerConfig` / `WarmupConfig` | discriminative learning rates, `eta_min`, warmup schedule |
| `EarlyStoppingConfig` | `patience`, `min_delta`, `restore_best` |
| `TuningConfig` (`configuration/tuning`) | `n_trials`, `startup_trials`, `warmup_trials` |

Hyperparameter search (`pipelines/tuning/tuner.py`) employs the Optuna **Tree-structured Parzen Estimator** with a **median pruner**, exploring a joint space spanning model, optimiser, and training configuration. The best trial is written to `<tuner_dir>/<model>/best_trial.json`.

---

## 8. Engineering Principles

This repository adheres to a strict engineering standard, reflecting the reproducibility requirements of scientific software:

- **Class-based, single-responsibility design** throughout the library, with each process orchestrated by a scheduler method that sequences its steps.
- **Fully spelled, semantically descriptive identifiers** — abbreviations are prohibited except for standard mathematical symbols in self-contained formulae.
- **Centralised structured logging** via `tools/monitoring/logger.py`, with `rich` instrumentation mandatory for every long-running loop.
- **Train-split-only normalisation statistics**, eliminating information leakage between data splits.
- An **academic, formal, and impersonal** tone across all code artefacts and documentation.

---

## 9. Citation

If this work informs your research, please cite it as:

```bibtex
@software{dune_gnn,
  title  = {DUNE-GNN: Graph Neural Networks for 3D Interaction-Vertex
            Reconstruction in Liquid Argon Time Projection Chambers},
  author = {{DUNE-GNN contributors}},
  year   = {2026},
  note   = {Geometric deep-learning pipeline for the DUNE photon detection system}
}
```

---

## 10. Selected References

1. DUNE Collaboration. *Deep Underground Neutrino Experiment (DUNE), Far Detector Technical Design Report.* JINST (2020).
2. Rampášek, L. et al. *Recipe for a General, Powerful, Scalable Graph Transformer (GPS).* NeurIPS (2022).
3. Brody, S., Alon, U. & Yahav, E. *How Attentive are Graph Attention Networks? (GATv2).* ICLR (2022).
4. Lee, J., Lee, I. & Kang, J. *Self-Attention Graph Pooling (SAGPool).* ICML (2019).
5. Perez, E. et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI (2018).

---

<div align="center">
<sub>Built for reproducible neutrino-physics research · configuration-driven · fully instrumented</sub>
</div>
