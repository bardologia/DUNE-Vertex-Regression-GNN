# Event Vertex Reconstruction in Liquid Argon Time Projection Chambers via Graph Neural Networks

## Abstract

This repository implements a **Graph Neural Network (GNN)** pipeline for three-dimensional vertex reconstruction of particle interaction events in Liquid Argon Time Projection Chambers (LArTPCs), with specific application to the **Deep Underground Neutrino Experiment (DUNE)** photon detection system. The framework represents each detector event as a sparse graph over optical sensor positions, where node features encode photon hit information and edge features capture spatial and calorimetric relationships between sensors. A **General, Powerful, Scalable (GPS) Graph Transformer** backbone with multi-scale hierarchical pooling extracts graph-level representations, which are subsequently decoded into $(x, y, z)$ event vertex coordinates through a **hierarchical regression head** employing Feature-wise Linear Modulation (FiLM) conditioning. The complete pipeline encompasses data ingestion and preprocessing, graph construction, model training with modern regularization techniques, and hyperparameter optimization via Bayesian search.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
   - 2.1 [Raw Data Format](#21-raw-data-format)
   - 2.2 [Data Loading](#22-data-loading)
   - 2.3 [Preprocessing Pipeline](#23-preprocessing-pipeline)
   - 2.4 [Graph Construction](#24-graph-construction)
   - 2.5 [Normalization and Splitting](#25-normalization-and-splitting)
3. [Model Architecture](#3-model-architecture)
   - 3.1 [Overview](#31-overview)
   - 3.2 [Input Projection](#32-input-projection)
   - 3.3 [GPS Layer](#33-gps-layer)
   - 3.4 [Multi-Scale Pooling](#34-multi-scale-pooling)
   - 3.5 [Hierarchical Regression Head](#35-hierarchical-regression-head)
   - 3.6 [FiLM Conditioning](#36-film-conditioning)
4. [Training Procedure](#4-training-procedure)
   - 4.1 [Optimizer Configuration](#41-optimizer-configuration)
   - 4.2 [Learning Rate Schedule](#42-learning-rate-schedule)
   - 4.3 [Regularization](#43-regularization)
   - 4.4 [Loss Function](#44-loss-function)
   - 4.5 [Evaluation Metrics](#45-evaluation-metrics)
   - 4.6 [Early Stopping and Checkpointing](#46-early-stopping-and-checkpointing)
   - 4.7 [Hyperparameter Optimization](#47-hyperparameter-optimization)
5. [Project Structure](#5-project-structure)
6. [Installation and Usage](#6-installation-and-usage)
7. [Configuration Reference](#7-configuration-reference)

---

## 1. Introduction

Neutrino interaction vertex reconstruction is a critical task in the analysis pipeline of LArTPC-based experiments such as DUNE. Precise determination of the three-dimensional coordinates $(x, y, z)$ at which a neutrino interacts within the detector volume is essential for event classification, energy estimation, and oscillation parameter extraction. Traditional reconstruction methods rely on handcrafted algorithms operating on charge readout signals; however, the photon detection system (PDS) offers complementary timing and calorimetric information that can be exploited for fast, independent vertex estimation.

This work formulates vertex reconstruction as a **graph-level regression problem**. Each event is represented as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where nodes $v_i \in \mathcal{V}$ correspond to optical sensors (e.g., SiPMs or PMTs) and edges $e_{ij} \in \mathcal{E}$ connect spatially proximate sensors determined via $k$-nearest neighbor (KNN) search. The model ingests per-sensor photon detection data and outputs a predicted vertex coordinate vector $\hat{\mathbf{y}} = (\hat{x}, \hat{y}, \hat{z}) \in \mathbb{R}^3$.

---

## 2. Dataset

### 2.1 Raw Data Format

Each event is stored as an individual **CSV file** in the `data/` directory. The filename encodes the ground truth event vertex coordinates according to the following schema:

```
pmod3_{x}_{y}_{z}_{...}.csv
```

where the second, third, and fourth underscore-delimited fields correspond to the true $x$, $y$, and $z$ coordinates, respectively. Each CSV file contains a tabular listing of sensor responses for that event with the following columns:

| Column          | Type    | Description                                           |
|-----------------|---------|-------------------------------------------------------|
| `bin`           | float   | Sensor bin identifier                                 |
| `x`             | float   | Sensor $x$-position (cm)                              |
| `y`             | float   | Sensor $y$-position (cm)                              |
| `z`             | float   | Sensor $z$-position (cm)                              |
| `light_amount`  | float   | Raw photon count detected by the sensor               |

A typical event contains on the order of several thousand sensors (rows), the majority of which may register zero photon counts. The sparse nature of the light detection pattern motivates the use of graph-based representations that naturally accommodate irregular sensor geometries.

### 2.2 Data Loading

Data ingestion is handled by the `DataLoader` class (in `core/data.py`), which performs the following operations:

1. **File discovery**: All CSV files matching `*.csv` in the configured input directory are sorted and truncated to `max_files`.
2. **Parallel parsing**: File I/O is parallelized over all available cores using [Ray](https://www.ray.io/). Each file is parsed independently, extracting both the sensor response DataFrame and the ground truth coordinates from the filename.
3. **Target extraction**: Ground truth vertex coordinates are assembled into a target DataFrame with columns `(event_x, event_y, event_z)`.

### 2.3 Preprocessing Pipeline

The `DataProcessor` class applies a sequence of physically motivated transformations to each event DataFrame prior to graph construction:

1. **Spatial outlier detection and removal**: For each sensor, the light intensity is compared to its spatial neighborhood (defined by a Euclidean radius $r$) using a $z$-score criterion. Sensors whose $z$-score exceeds a configurable threshold $\tau_z$ are flagged as outliers and their light values are replaced by the neighborhood mean. This mitigates the effect of noisy or malfunctioning channels. The spatial neighborhood queries are accelerated using `scipy.spatial.cKDTree`.

   $$z_i = \frac{|l_i - \mu_{\mathcal{N}(i)}|}{\sigma_{\mathcal{N}(i)}}$$

   where $l_i$ is the light intensity of sensor $i$, and $\mu_{\mathcal{N}(i)}$, $\sigma_{\mathcal{N}(i)}$ are the mean and standard deviation of light intensities in its spatial neighborhood $\mathcal{N}(i)$.

   Default parameters: $\tau_z = 9.0$, $r = 1.0$ cm.

2. **Light intensity scaling**: All photon counts are multiplied by a global scale factor $\alpha$ to map raw counts into a more suitable dynamic range for subsequent processing.

   $$l_i' = \alpha \cdot l_i$$

   Default: $\alpha = 0.1$.

3. **Detection efficiency simulation**: To simulate realistic photon detection efficiency $\epsilon$, each sensor's scaled photon count is resampled via a binomial process:

   $$l_i'' \sim \mathrm{Binomial}\left(\lfloor l_i' \rceil,\; \epsilon\right)$$

   Default: $\epsilon = 0.02$ (2% detection efficiency).

4. **Logarithmic transform**: A $\log(1+x)$ transform is applied to compress the dynamic range and stabilize variance:

   $$l_i^{*} = \log(1 + \max(l_i'', 0))$$

### 2.4 Graph Construction

Graph construction is performed by the `Graph` class (`core/dataset.py`), which converts each preprocessed event DataFrame into a PyTorch Geometric `Data` object. This involves two stages: **node feature engineering** and **edge construction with feature extraction**.

#### 2.4.1 Node Features

Each sensor $i$ in the event is represented by an 8-dimensional node feature vector:

| Index | Feature                  | Description                                                                                      |
|-------|--------------------------|--------------------------------------------------------------------------------------------------|
| 0     | $x_i$                    | Sensor $x$-coordinate                                                                           |
| 1     | $y_i$                    | Sensor $y$-coordinate                                                                           |
| 2     | $z_i$                    | Sensor $z$-coordinate                                                                           |
| 3     | $l_i$                    | Preprocessed light intensity                                                                    |
| 4     | $h_i$                    | Binary hit flag: $h_i = \mathbb{1}[l_i > 0]$                                                    |
| 5     | $f_i$                    | Light fraction: $f_i = l_i / \sum_j l_j$                                                        |
| 6     | $d_i^{c}$                | Euclidean distance to the light-weighted centroid                                                |
| 7     | $\rho_i$                 | Local light density: sum of light intensities over the $k$-nearest neighbors                     |

The **light-weighted centroid** is computed as:

$$\mathbf{c} = \frac{\sum_i l_i \cdot \mathbf{p}_i}{\sum_i l_i + \varepsilon}$$

where $\mathbf{p}_i = (x_i, y_i, z_i)$ and $\varepsilon = 10^{-8}$ prevents division by zero. The distance from each sensor to this centroid provides a measure of proximity to the region of highest scintillation activity, serving as a physics-informed feature.

The **local light density** for sensor $i$ is defined as:

$$\rho_i = \sum_{j \in \mathrm{KNN}_k(i) \setminus \{i\}} l_j$$

where $\mathrm{KNN}_k(i)$ denotes the $k$-nearest neighbors of sensor $i$ in Euclidean space (default $k = 4$), computed using a KD-tree algorithm.

#### 2.4.2 Edge Construction and Features

Edges are established via a $k$-nearest neighbor (KNN) graph over sensor positions. For each sensor $i$, directed edges are created to its $k$ nearest spatial neighbors. If bidirectional mode is enabled, reverse edges are also added with appropriately negated directional features.

Each edge $(i, j)$ carries a 7-dimensional feature vector:

| Index | Feature                     | Description                                                                                   |
|-------|-----------------------------|-----------------------------------------------------------------------------------------------|
| 0     | $d_{ij}$                    | Euclidean distance between sensors $i$ and $j$                                                 |
| 1     | $\Delta x_{ij} / d_{ij}$   | Normalized displacement along $x$                                                              |
| 2     | $\Delta y_{ij} / d_{ij}$   | Normalized displacement along $y$                                                              |
| 3     | $\Delta z_{ij} / d_{ij}$   | Normalized displacement along $z$                                                              |
| 4     | $1 / (d_{ij}^2 + \varepsilon)$ | Inverse square distance (analogous to geometric light fall-off)                             |
| 5     | $l_j - l_i$                | Light gradient from source to destination                                                      |
| 6     | $\frac{2 \min(l_i, l_j)}{l_i + l_j + \varepsilon}$ | Light similarity (Sørensen-Dice coefficient analog)                  |

The inverse square distance feature is motivated by the physical inverse-square law governing photon flux from a point source, while the light gradient and similarity features encode calorimetric relationships between neighboring sensors.

### 2.5 Normalization and Splitting

#### 2.5.1 Stratified Splitting

The dataset is partitioned into training, validation, and test splits using **stratified shuffle splitting** (`sklearn.model_selection.StratifiedShuffleSplit`). Stratification is performed over a composite binning of the three target coordinates: each coordinate is discretized into $n$ percentile-based bins, and the Cartesian product of bin indices forms the stratification label. This ensures that the marginal and joint distributions of target coordinates are preserved across splits.

Default proportions: 80% train, 10% validation, 10% test.

#### 2.5.2 Z-Score Normalization

After splitting, **z-score normalization** is applied to node features, edge attributes, and target variables. Statistics (mean, standard deviation) are computed exclusively on the **training split** to prevent information leakage:

$$\tilde{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \varepsilon}$$

Normalization metrics are persisted and used during inference to denormalize model predictions back to physical units.

---

## 3. Model Architecture

### 3.1 Overview

The model follows an **encoder-decoder** paradigm:

1. **Encoder (GraphBackbone)**: A stack of GPS (General, Powerful, Scalable) Transformer layers that jointly leverage local message passing (GATv2) and global self-attention to enrich node representations, followed by multi-scale hierarchical pooling to produce a fixed-width graph-level embedding.
2. **Decoder (HierarchicalRegressionHead)**: A coordinate-wise regression head that predicts $(\hat{x}, \hat{y}, \hat{z})$ sequentially, with each subsequent coordinate conditioned on prior predictions via FiLM modulation.

The overall forward pass of the `Model` class can be summarized as:

$$\mathbf{h}_{\mathcal{G}} = \text{LayerNorm}\!\left(\text{GraphBackbone}(\mathcal{G})\right)$$

$$(\hat{x}, \hat{y}, \hat{z}) = \text{HierarchicalRegressionHead}(\mathbf{h}_{\mathcal{G}})$$

### 3.2 Input Projection

The raw 8-dimensional node features are projected into the hidden dimension $d$ of the GPS layers via a linear layer followed by layer normalization:

$$\mathbf{h}_i^{(0)} = \text{LayerNorm}\!\left(\mathbf{W}_{\text{proj}} \mathbf{x}_i + \mathbf{b}_{\text{proj}}\right)$$

Default hidden dimension: $d = 128$.

### 3.3 GPS Layer

Each GPS layer (class `GPSLayer`) implements a **dual-track architecture** that combines local graph convolution with global self-attention, inspired by Rampasek et al. (2022). The layer consists of three residual sub-blocks:

#### 3.3.1 Local Track — GATv2 Convolution

The local track applies Graph Attention Network v2 (GATv2Conv, Brody et al. 2022) to the pre-normalized node features:

$$\mathbf{h}_i^{\text{local}} = \text{GATv2Conv}\!\left(\text{LayerNorm}(\mathbf{h}_i),\; \mathcal{E},\; \mathbf{e}_{ij}\right)$$

GATv2 uses dynamic attention coefficients that depend on both source and destination node features, with edge features incorporated as additive bias terms. Multi-head attention with $H$ heads (default $H = 4$) is employed, with outputs averaged (non-concatenated) to preserve the hidden dimension. Self-loops are added to ensure each node attends to itself.

#### 3.3.2 Global Track — Multi-Head Self-Attention

The global track applies standard multi-head self-attention over all nodes within each graph in the batch, enabling long-range information flow regardless of graph topology:

$$\mathbf{h}_i^{\text{global}} = \text{MHA}\!\left(\text{LayerNorm}(\mathbf{h}_i)\right)$$

Dense batching (`to_dense_batch`) is used to pad variable-size graphs to a uniform length within each batch, with attention masks preventing cross-graph attention. This operation has $O(N^2)$ complexity per graph and constitutes the primary computational bottleneck.

#### 3.3.3 Gated Fusion

The local and global representations are fused via a learned sigmoid gate:

$$\mathbf{g} = \sigma\!\left(\mathbf{W}_g [\mathbf{h}^{\text{local}} \| \mathbf{h}^{\text{global}}] + \mathbf{b}_g\right)$$

$$\mathbf{h}^{\text{fused}} = \mathbf{g} \odot \mathbf{h}^{\text{local}} + (1 - \mathbf{g}) \odot \mathbf{h}^{\text{global}}$$

The fused representation is added to the input via a residual connection with **stochastic depth** (DropPath):

$$\mathbf{h}_i \leftarrow \mathbf{h}_i + \text{DropPath}(\mathbf{h}^{\text{fused}})$$

#### 3.3.4 Feed-Forward Network

A two-layer feed-forward network with GELU activation and residual connection completes the GPS layer:

$$\mathbf{h}_i \leftarrow \mathbf{h}_i + \text{DropPath}\!\left(\text{FFN}\!\left(\text{LayerNorm}(\mathbf{h}_i)\right)\right)$$

$$\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \,\text{Dropout}\!\left(\text{GELU}\!\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right)\right) + \mathbf{b}_2$$

The FFN expansion ratio is configurable (default: $4\times$). DropPath rates are linearly increased from 0 to `drop_path_rate` across the depth of the GPS stack following the stochastic depth schedule of Huang et al. (2016).

By default, the backbone consists of $L = 4$ GPS layers.

### 3.4 Multi-Scale Pooling

Graph-level representations are obtained through a `MultiScalePool` module that constructs a hierarchical coarsening of the graph and aggregates information at multiple resolutions.

At each of $M$ resolution levels (default $M = 3$):
1. **Mean and max pooling** are applied to the current-resolution node features and concatenated.
2. **Self-Attention Graph Pooling (SAGPool)** with GATv2-based scoring (Lee et al., 2019) selects a fraction $r$ (default $r = 0.5$) of the most salient nodes, producing a coarsened graph for the next level.

The readouts from all $M$ levels are concatenated to form the final graph-level representation:

$$\mathbf{h}_{\mathcal{G}} = \bigoplus_{m=1}^{M} \left[\text{MeanPool}(\mathbf{H}^{(m)}) \| \text{MaxPool}(\mathbf{H}^{(m)})\right] \in \mathbb{R}^{2Md}$$

where $\bigoplus$ denotes concatenation and $\mathbf{H}^{(m)}$ are the node features at resolution level $m$. This yields a representation with dimensionality $2 \times 3 \times 128 = 768$ under default settings.

### 3.5 Hierarchical Regression Head

The `HierarchicalRegressionHead` predicts vertex coordinates **sequentially** — $\hat{x}$ first, then $\hat{y}$ conditioned on $\hat{x}$, then $\hat{z}$ conditioned on $(\hat{x}, \hat{y})$ — to capture inter-coordinate dependencies:

1. **Feature projection**: The graph embedding is projected to an intermediate dimension via a linear layer, layer norm, GELU, and dropout.

   $$\mathbf{f} = \text{Dropout}\!\left(\text{GELU}\!\left(\text{LayerNorm}\!\left(\mathbf{W}_f \mathbf{h}_{\mathcal{G}} + \mathbf{b}_f\right)\right)\right)$$

2. **$x$-coordinate prediction**:

   $$\hat{x} = \text{CoordinateHead}_x(\mathbf{f})$$

3. **$y$-coordinate prediction** (conditioned on $\hat{x}$):

   $$\mathbf{f}_y = \text{FiLM}_y(\hat{x},\; \mathbf{f})$$
   $$\hat{y} = \text{CoordinateHead}_y(\mathbf{f}_y)$$

4. **$z$-coordinate prediction** (conditioned on $\hat{x}$ and $\hat{y}$):

   $$\mathbf{f}_z = \text{FiLM}_z([\hat{x}, \hat{y}],\; \mathbf{f})$$
   $$\hat{z} = \text{CoordinateHead}_z(\mathbf{f}_z)$$

Each `CoordinateHead` is a multi-layer perceptron with LayerNorm, GELU activation, and dropout between hidden layers, terminating in a single linear output. Default hidden dimensions: $(128, 64)$.

### 3.6 FiLM Conditioning

**Feature-wise Linear Modulation (FiLM)** (Perez et al., 2018) is used to condition intermediate features on previously predicted coordinates. A small conditioning network maps the condition vector $\mathbf{c}$ to scale ($\gamma$) and shift ($\beta$) parameters:

$$(\gamma, \beta) = \text{split}\!\left(\mathbf{W}_2 \,\text{GELU}\!\left(\mathbf{W}_1 \mathbf{c} + \mathbf{b}_1\right) + \mathbf{b}_2\right)$$

$$\text{FiLM}(\mathbf{c}, \mathbf{f}) = \gamma \odot \mathbf{f} + \beta$$

The last layer of the conditioning network is initialized to zero weights with bias $\gamma_0 = 1$, $\beta_0 = 0$, ensuring that conditioning behaves as an identity at initialization and gradually learns to modulate features during training.

---

## 4. Training Procedure

### 4.1 Optimizer Configuration

Training employs the **AdamW** optimizer (Loshchilov & Hutter, 2019) with decoupled weight decay. Three distinct parameter groups are defined with independent learning rates and regularization strengths:

| Parameter Group        | Default LR   | Default Weight Decay |
|------------------------|-------------|----------------------|
| Regression Head        | $10^{-2}$   | $10^{-5}$           |
| Multi-Scale Pooling    | $10^{-3}$   | $10^{-5}$           |
| GPS Backbone (GCN)     | $10^{-3}$   | $10^{-4}$           |

This differential learning rate scheme assigns a higher learning rate to the regression head (which must adapt more aggressively to the coordinate space) while keeping the backbone learning rate lower to promote stable feature extraction. Adam hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

### 4.2 Learning Rate Schedule

Training uses a two-phase learning rate schedule:

1. **Linear warmup** (default 100 steps): The learning rate is linearly increased from a fraction $\lambda_0 = 0.1$ of the base rate to the full base rate. This stabilizes early training when gradients may be large and noisy.

   $$\text{lr}(t) = \text{lr}_{\text{base}} \cdot \left[\lambda_0 + (1 - \lambda_0) \cdot \frac{t}{T_{\text{warmup}}}\right], \quad t \leq T_{\text{warmup}}$$

2. **Cosine annealing** (after warmup): The learning rate follows a cosine decay schedule from the base rate to a minimum value $\eta_{\min}$:

   $$\text{lr}(t) = \eta_{\min} + \frac{1}{2}(\text{lr}_{\text{base}} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi \, t}{T_{\max}}\right)\right)$$

   Default: $T_{\max} = 250$ epochs, $\eta_{\min} = 10^{-6}$.

### 4.3 Regularization

Multiple regularization mechanisms are employed to prevent overfitting:

- **Dropout** ($p = 0.1$): Applied within feed-forward networks and attention layers.
- **Attention dropout** ($p = 0.1$): Applied to attention weights in both GATv2 and global self-attention.
- **Stochastic depth / DropPath**: Linearly increasing drop probability from 0 to 0.1 across GPS layers. At each layer, the entire residual branch is stochastically dropped during training.
- **Weight decay**: Decoupled $L_2$ regularization applied via AdamW (see Section 4.1).
- **Gradient clipping**: Global gradient norm is clipped to a maximum of 1.0 to stabilize training.
- **Mixed-precision training (AMP)**: When CUDA is available, automatic mixed precision is used with `torch.amp.GradScaler` for memory efficiency and training speed, without compromising convergence.
- **Exponential Moving Average (EMA)** (optional): An exponentially weighted moving average of model parameters is maintained ($\beta = 0.9995$), and the EMA model is used for evaluation. This acts as a form of temporal ensembling.

### 4.4 Loss Function

The training objective is the **Mean Squared Error (MSE)** between predicted and ground truth (normalized) vertex coordinates:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|^2 = \frac{1}{N} \sum_{i=1}^{N} \left[(\hat{x}_i - x_i)^2 + (\hat{y}_i - y_i)^2 + (\hat{z}_i - z_i)^2\right]$$

where predictions and targets are in the normalized (z-scored) coordinate space. Denormalization is applied post-hoc for metric computation.

### 4.5 Evaluation Metrics

Model performance is assessed using the following per-coordinate and aggregate metrics, all computed in the **original (denormalized) physical coordinate space**:

| Metric                         | Definition                                                                                              |
|--------------------------------|---------------------------------------------------------------------------------------------------------|
| **MAE** (per-coord & overall)  | $\text{MAE}_c = \frac{1}{N}\sum_i |\hat{c}_i - c_i|$                                                   |
| **RMSE** (per-coord & overall) | $\text{RMSE}_c = \sqrt{\frac{1}{N}\sum_i (\hat{c}_i - c_i)^2}$                                         |
| **$R^2$** (per-coord & overall)| $R^2_c = 1 - \frac{\sum_i (\hat{c}_i - c_i)^2}{\sum_i (c_i - \bar{c})^2}$                              |
| **Median absolute error**      | Per-coordinate median of $|\hat{c}_i - c_i|$                                                            |
| **Euclidean error**            | $\epsilon_i = \|\hat{\mathbf{y}}_i - \mathbf{y}_i\|_2$ — mean, std, median, min, and max are reported   |

All metrics are logged to **TensorBoard** at every validation epoch for per-coordinate tracking and cross-coordinate comparison.

### 4.6 Early Stopping and Checkpointing

- **Early stopping**: Training halts when the validation loss fails to improve by at least $\delta_{\min} = 0.0$ for 15 consecutive epochs. When triggered, the model parameters are restored to the best observed state.
- **Checkpointing**: The best model (by validation loss) is saved as `best_model.pt`, including model parameters, optimizer state, scheduler state, EMA shadow parameters, early stopping state, and warmup progress. Training can be resumed from any checkpoint.

### 4.7 Hyperparameter Optimization

Automated hyperparameter search is conducted using **Optuna** (Akiba et al., 2019) with Tree-structured Parzen Estimator (TPE) sampling. The search space encompasses:

| Hyperparameter                 | Search Space                          |
|--------------------------------|---------------------------------------|
| GPS hidden dimension           | $\{64, 128, 192, 256\}$              |
| Number of GPS layers           | $[2, 6]$                             |
| Number of attention heads      | $\{2, 4, 8\}$                        |
| GPS / attention dropout        | $[0.0, 0.2]$, step $0.05$            |
| DropPath rate                  | $[0.0, 0.2]$, step $0.05$            |
| Pooling levels                 | $[2, 4]$                             |
| SAGPool ratio                  | $[0.3, 0.7]$, step $0.1$             |
| Hierarchical feature dim       | $\{128, 256, 384\}$                  |
| Coordinate embedding dim       | $\{32, 64, 128\}$                    |
| Regression layers              | $[2, 4]$ with per-layer dim in $\{64, 128, 256\}$ |
| Per-group learning rates       | log-uniform $[5 \times 10^{-5}, 2 \times 10^{-3}]$ |
| Per-group weight decays        | log-uniform $[10^{-6}, 5 \times 10^{-4}]$          |

Optuna's **MedianPruner** is employed to terminate unpromising trials early, with configurable warmup trials and warmup steps prior to pruning activation.

---

## 5. Project Structure

```
dune-gnn/
├── core/
│   ├── config.py        # Dataclass-based configuration (GlobalConfig)
│   ├── data.py          # DataLoader and DataProcessor classes
│   ├── dataset.py       # Graph construction, DatasetBuilder, DatasetNormalizer, DatasetLoader
│   ├── model.py         # GPS backbone, multi-scale pooling, FiLM, hierarchical regression head
│   ├── trainer.py       # Training loop, optimizer, scheduler, EMA, early stopping, metrics
│   ├── tuner.py         # Optuna-based hyperparameter optimization
│   └── logger.py        # Logging, TensorBoard tracking, diagnostic plots, model summaries
├── agents/              # AI agent specifications for code generation
├── data/                # Raw event CSV files
├── datasets/            # Preprocessed graph datasets (chunked .pt files)
├── main/                # Entry-point scripts (e.g., create_dataset.py)
├── notebooks/           # Jupyter notebooks for analysis
├── runs/                # Training run logs and checkpoints
├── tuning/              # Hyperparameter search artifacts
└── requirements.txt     # Python dependencies
```

---

## 6. Installation and Usage

### 6.1 Requirements

- Python $\geq$ 3.10
- CUDA-compatible GPU (recommended)

### 6.2 Installation

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package              | Purpose                                      |
|----------------------|----------------------------------------------|
| `torch`              | Deep learning framework                      |
| `torch-geometric`    | Graph neural network operators and utilities  |
| `scikit-learn`       | KNN, stratified splitting, preprocessing     |
| `numpy` / `pandas`   | Numerical computation and data manipulation  |
| `ray`                | Distributed parallel data loading and processing |
| `optuna`             | Bayesian hyperparameter optimization          |
| `tensorboard`        | Training metric visualization                |
| `matplotlib`         | Diagnostic plotting                          |
| `tqdm`               | Progress bars                                |

### 6.3 Dataset Construction

```bash
python main/create_dataset.py
```

This will load raw CSV files from `data/`, apply the preprocessing pipeline, construct KNN graphs, and save the resulting PyTorch Geometric dataset in chunked format to `datasets/`.

### 6.4 Training

Training is initiated through the `Trainer` class, which accepts a `Model` instance, normalization metrics, and configuration. Logs and checkpoints are written to the `runs/` directory and can be monitored with TensorBoard:

```bash
tensorboard --logdir=runs/
```

---

## 7. Configuration Reference

All hyperparameters are centralized in `core/config.py` via nested Python `dataclass` objects under the `GlobalConfig` root. The following table summarizes the principal configuration groups:

| Config Group        | Key Parameters                                                                                   |
|---------------------|--------------------------------------------------------------------------------------------------|
| `DataConfig`        | `input_dir`, `max_files`, `coordinate_columns`, `position_columns`, `light_column`                |
| `GraphConfig`       | `k_neighbors` (4), `knn_algorithm` ("kd_tree"), `bidirectional` (False)                           |
| `PreprocessingConfig` | `zscore_threshold` (9.0), `radius` (1.0), `scale_factor` (0.1), `detection_efficiency` (0.02)  |
| `ModelConfig`       | `input_dim` (8), `gps_hidden_dim` (128), `gps_num_layers` (4), `gps_heads` (4), `edge_dim` (7)   |
| `TrainingConfig`    | `epochs` (10), `batch_size` (4), `use_amp` (True), `max_grad_norm` (1.0)                         |
| `OptimizerConfig`   | Per-group `lr` and `weight_decay`, `betas` (0.9, 0.999)                                          |
| `SchedulerConfig`   | `epochs` (250), `eta_min` ($10^{-6}$)                                                            |
| `EarlyStoppingConfig` | `patience` (15), `min_delta` (0.0), `restore_best` (True)                                     |
| `EMAConfig`         | `use_ema` (False), `ema_decay` (0.9995)                                                          |
| `WarmupConfig`      | `warmup_steps` (100), `warmup_start_factor` (0.1)                                                |
| `SplitConfig`       | `test_size` (0.1), `val_ratio` (0.1), `random_state` (42)                                        |
| `TuningConfig`      | `n_trials` (5), `warmup_trials` (1), `warmup_steps` (1)                                          |

---

## References

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.
- Brody, S., Alon, U., & Yahav, E. (2022). How attentive are Graph Attention Networks? *Proceedings of the International Conference on Learning Representations (ICLR)*.
- Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep networks with stochastic depth. *Proceedings of the European Conference on Computer Vision (ECCV)*, 646–661.
- Lee, J., Lee, I., & Kang, J. (2019). Self-Attention Graph Pooling. *Proceedings of the International Conference on Machine Learning (ICML)*, 3734–3743.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *Proceedings of the International Conference on Learning Representations (ICLR)*.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).
- Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D. (2022). Recipe for a General, Powerful, Scalable Graph Transformer. *Advances in Neural Information Processing Systems (NeurIPS)*, 35.
