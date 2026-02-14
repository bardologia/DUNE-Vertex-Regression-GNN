# DUNE-GNN: Graph Neural Networks for Neutrino Event Reconstruction

## Abstract

This repository implements a Graph Neural Network (GNN) framework for spatial event reconstruction in the Deep Underground Neutrino Experiment (DUNE) detector. The model processes light sensor readings from neutrino interaction events and reconstructs the three-dimensional coordinates of the interaction vertex using Graph Attention Networks (GAT) combined with Self-Attention Graph (SAG) pooling mechanisms.

## Overview

Neutrino detectors like DUNE utilize arrays of photon sensors to detect scintillation light produced by particle interactions. The spatial distribution and intensity of detected photons provide information about the location and properties of the original interaction. This project formulates the event reconstruction problem as a graph-based regression task, where:

- **Nodes** represent individual photon detector modules with features including spatial position $(x, y, z)$ and detected light intensity
- **Edges** connect spatially proximate sensors using k-nearest neighbors (k-NN) graph construction
- **Task** is to regress the 3D coordinates $(x_{\text{event}}, y_{\text{event}}, z_{\text{event}})$ of the neutrino interaction vertex

## Architecture

### Model Components

#### 1. Graph Construction
The input data undergoes the following preprocessing pipeline:

```
Raw CSV Files → Spatial Feature Extraction → k-NN Graph Construction → PyTorch Geometric Data Objects
```

- **Spatial outlier detection**: Z-score based anomaly detection with configurable neighborhood radius
- **k-NN graph**: Bidirectional edges constructed using KD-tree algorithm
- **Edge attributes**: Euclidean distances between connected sensors

#### 2. Graph Neural Network Backbone

The backbone architecture consists of stacked **GAT (Graph Attention Network) v2** layers:

```
Input Features (x, y, z, light_intensity) 
    ↓
GAT Block 1 (in_dim → 192, heads=6)
    ↓ [LayerNorm + ReLU + Dropout]
GAT Block 2 (192 → 384, heads=6)
    ↓ [LayerNorm + ReLU + Dropout]
GAT Block 3 (384 → 256, heads=6)
    ↓
SAG Pooling (ratio=0.5, layers=2)
    ↓ [Mean + Max aggregation]
Regression Head (512 → 256 → 128 → 3)
    ↓
Output: (x_pred, y_pred, z_pred)
```

**Key architectural features:**
- **Attention mechanism**: Multi-head attention with edge features for adaptive neighborhood aggregation
- **Residual connections**: Skip connections in each GAT block for gradient flow
- **Hierarchical pooling**: SAG pooling for graph-level representation learning
- **Dual aggregation**: Concatenation of mean and max pooling for robust feature extraction

#### 3. Regression Head

A multi-layer perceptron (MLP) with LayerNorm and dropout for final coordinate prediction:

```python
MLP(512, 256, 128, 3) with:
- LayerNorm after each hidden layer
- ReLU activation
- Dropout (p=0.1)
- Xavier initialization (gain=0.01) for output layer
```

## Technical Details

### Training Pipeline

- **Optimizer**: Adam with per-parameter-group learning rates and weight decay
  - GCN layers: $lr \in [10^{-5}, 10^{-3}]$, $\lambda \in [10^{-6}, 5 \times 10^{-4}]$
  - Pooling: $lr \in [10^{-5}, 10^{-3}]$, $\lambda \in [10^{-6}, 10^{-4}]$
  - Regression head: $lr \in [5 \times 10^{-5}, 2 \times 10^{-3}]$, $\lambda \in [10^{-6}, 10^{-4}]$

- **Learning rate schedule**: Cosine annealing with linear warmup
  - Warmup steps: 100
  - Warmup start factor: 0.1
  - $\eta_{\min} = 10^{-6}$

- **Loss function**: Mean Squared Error (MSE) for coordinate regression

- **Regularization**:
  - Dropout in GAT layers and regression head
  - L2 weight decay with per-parameter-group rates
  - Gradient clipping (max_norm=1.0)
  - Optional: Exponential Moving Average (EMA) of model weights ($\beta = 0.9995$)

- **Training optimizations**:
  - Mixed precision training (AMP)
  - Gradient accumulation (steps=4)
  - Early stopping with patience=15

### Hyperparameter Optimization

The framework includes Optuna-based hyperparameter tuning with the following search space:

| Parameter | Range | Distribution |
|-----------|-------|--------------|
| GCN hidden dims | [128, 192, 256, 384] | Categorical |
| Regression hidden dims | [128, 256, 512] | Categorical |
| GAT dropout | [0.0, 0.2] | Uniform (step=0.05) |
| Regression dropout | [0.0, 0.2] | Uniform (step=0.05) |
| GAT heads | [4, 6, 8] | Categorical |
| Learning rates | $[10^{-5}, 10^{-3}]$ | Log-uniform |
| Weight decay | $[10^{-6}, 10^{-4}]$ | Log-uniform |

### Data Processing

**Preprocessing pipeline**:
1. **Outlier detection**: Spatial z-score filtering with neighborhood radius
2. **Light intensity scaling**: Optional multiplicative factor for energy normalization
3. **Detection efficiency**: Binomial sampling to simulate photon detection efficiency
4. **Log transformation**: $\log(1 + x)$ applied to light intensity for dynamic range compression
5. **Normalization**: Feature-wise standardization $(x - \mu) / \sigma$
6. **Dataset balancing**: Stratified train/val/test split with target distribution preservation

**Distributed processing**: Ray framework for parallel data loading and preprocessing

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/dune-gnn.git
cd dune-gnn

# Create a virtual environment (recommended)
conda create -n dune-gnn python=3.9
conda activate dune-gnn

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric torch-scatter

# Install remaining dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch                    # Deep learning framework
torch-geometric          # Graph neural network library
torch-scatter            # Scatter operations for graph pooling
scikit-learn             # Machine learning utilities
numpy                    # Numerical computing
pandas                   # Data manipulation
matplotlib               # Visualization
tqdm                     # Progress bars
tensorboard              # Experiment tracking
optuna                   # Hyperparameter optimization
ray[tune,data]           # Distributed computing
```

## Usage

### 1. Data Preparation

Prepare your DUNE detector data in CSV format with the following structure:

```csv
bin,x,y,z,light_amount
0,1.23,4.56,7.89,42.5
...
```

Filenames should encode the event coordinates: `event_x_y_z.csv`

### 2. Dataset Building

```python
from core.dataset import DatasetBuilder
from core.logger import Logger

logger = Logger(log_dir="./logs", name="DatasetBuilder")
builder = DatasetBuilder(output_dir="./datasets/processed", logger=logger)

# Build and preprocess dataset
builder.build_dataset()
```

Configuration options in `core/config.py`:
- Input directory path
- k-NN graph parameters (k, algorithm)
- Preprocessing thresholds (outlier detection, scaling)
- Feature and coordinate column mappings

### 3. Model Training

```python
from main.train_pipeline import main

# Train with default configuration
train_losses, val_losses, test_results = main()
```

Or via command line:

```bash
cd main
python train_pipeline.py
```

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir runs/ --port 6006
```

### 4. Hyperparameter Tuning

```python
from core.tuner import GNNTuner
from core.dataset import DatasetLoader

loader = DatasetLoader(preprocessed_dir="./datasets/processed")
train_dataset, val_dataset, test_dataset = loader.load_split_datasets()

tuner = GNNTuner(norm_metrics=loader.norm_metrics, dataset_config=loader.config)
best_params = tuner.tune(train_dataset, val_dataset, n_trials=100)
```

Or use the Jupyter notebook:

```bash
jupyter notebook main/tunner_pipeline.ipynb
```

## Project Structure

```
dune-gnn/
├── core/               # Core modules
│   ├── config.py      # Configuration dataclasses
│   ├── data.py        # Data loading and preprocessing
│   ├── dataset.py     # Graph dataset construction
│   ├── logger.py      # Logging utilities
│   ├── model.py       # GNN architecture
│   ├── trainer.py     # Training loop and utilities
│   └── tuner.py       # Hyperparameter optimization
├── main/              # Execution pipelines
│   ├── dataset_pipeline.py     # Dataset preparation script
│   ├── train_pipeline.py       # Training script
│   └── tunner_pipeline.ipynb   # Hyperparameter tuning notebook
├── data/              # Raw data directory
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Configuration

The `core/config.py` file provides comprehensive configuration through dataclasses:

- **DataConfig**: File paths, column mappings, input limits
- **GraphConfig**: k-NN parameters, edge construction settings
- **ModelConfig**: Architecture dimensions, dropout rates, attention heads
- **TrainingConfig**: Batch size, learning rate, epochs, device settings
- **EarlyStoppingConfig**: Patience, minimum delta, model restoration
- **SchedulerConfig**: Cosine annealing parameters
- **WarmupConfig**: Linear warmup settings
- **EMAConfig**: Exponential moving average parameters



