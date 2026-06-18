<div align="center">

# DUNE-GNN

### Graph Neural Networks for 3D Interaction-Vertex Reconstruction in Liquid Argon Time Projection Chambers

*A geometric deep-learning pipeline for reconstructing the spatial origin of neutrino interactions from the photon detection system of the Deep Underground Neutrino Experiment (DUNE).*

<br>

![Python](https://img.shields.io/badge/python-≥3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-graph_learning-3C2179)
![Optuna](https://img.shields.io/badge/Optuna-hyperparameter_search-0E4C92)
![Ray](https://img.shields.io/badge/Ray-parallel_I%2FO-028CF0)
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

Ground-truth coordinates are derived from the simulation file naming convention `pmod3_{x}_{y}_{z}_..._N.csv`, allowing the supervised target to be recovered directly from the dataset without an auxiliary label store.

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
  graph construction       active-sensor pruning · k-nearest-neighbour graph · 7-D node features · 7-D edge features
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

- **Node features (7-dimensional):** sensor position, measured light intensity, the per-sensor light fraction, the distance to the light-weighted centroid of the event, and a measure of local light density.
- **Edge features (7-dimensional):** Euclidean separation, normalised displacement vector, inverse-square distance, light gradient along the edge, and a Sørensen–Dice photometric similarity.

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
├── main/
│   ├── create_dataset.py     # build the preprocessed graph dataset from raw CSVs
│   ├── train.py              # training entry point (logs + checkpoints → runs/)
│   └── tune.py               # Optuna hyperparameter search
├── core/
│   ├── config.py             # nested dataclass configuration (GlobalConfig)
│   ├── data.py               # CSV discovery, label extraction, Ray-parallel I/O
│   ├── dataset.py            # PyG dataset, normalisation (train-split statistics only)
│   ├── model.py              # GPS backbone, multi-scale pooling, hierarchical head
│   ├── trainer.py            # training loop, scheduling, AMP, EMA, metrics
│   ├── tuner.py              # Optuna TPE + MedianPruner search space
│   ├── runtime.py            # runtime / device orchestration
│   └── logger.py             # centralised rich-based structured logging
├── notebooks/                # pipeline inspection and sensitivity analyses
├── agents/                   # project engineering conventions and contracts
├── data/                     # raw simulated events (pmod3_*.csv)
└── requirements.txt
```

---

## 5. Installation

```bash
pip install -r requirements.txt
```

**Requirements.** Python ≥ 3.10 and a CUDA-capable GPU (recommended). Core dependencies include PyTorch, PyTorch Geometric, scikit-learn, NumPy/SciPy/pandas, Ray (parallel file ingestion), Optuna (hyperparameter optimisation), TensorBoard, and `rich`.

---

## 6. Usage

```bash
# 1. Build the preprocessed graph dataset from the raw CSV events
python main/create_dataset.py

# 2. Train the model (checkpoints and logs are written under runs/)
python main/train.py

# 3. Run a hyperparameter search with Optuna
python main/tune.py

# 4. Monitor training in real time
tensorboard --logdir=runs/
```

---

## 7. Configuration

The project is **entirely configuration-driven**: there are no command-line flags. All hyperparameters are declared as nested `dataclass` objects under `GlobalConfig` in `core/config.py` and are edited directly prior to a run. The principal groups are:

| Group | Representative fields |
|---|---|
| `DataConfig` | `input_dir`, `max_files` |
| `GraphConfig` | `k_neighbors`, `bidirectional` |
| `PreprocessingConfig` | `zscore_threshold`, `scale_factor`, `detection_efficiency` |
| `ModelConfig` | `gps_hidden_dim`, `gps_num_layers`, `gps_heads` |
| `TrainingConfig` | `epochs`, `batch_size`, `use_amp` |
| `EarlyStoppingConfig` | `patience` |
| `TuningConfig` | `n_trials` |

Hyperparameter search (`core/tuner.py`) employs the Optuna **Tree-structured Parzen Estimator** with a **median pruner**, exploring a large joint space spanning model, optimiser, and training configuration.

---

## 8. Engineering Principles

This repository adheres to a strict, contractually-enforced engineering standard (see `agents/general_rules.md`), reflecting the reproducibility requirements of scientific software:

- **Class-based, single-responsibility design** throughout the core library.
- **Fully spelled, semantically descriptive identifiers** — abbreviations are prohibited except for standard mathematical symbols in self-contained formulae.
- **Centralised structured logging** via `core/logger.py`, with `rich` instrumentation mandatory for every long-running loop.
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
