# AI Agent Specification — Generic Modular Model Builder (Template-Aligned)

## 1) Objective

Generate a **generic, modular deep-learning model module** that stays **as close as possible** to the given template’s structure and philosophy, while remaining **architecture-agnostic** (GNN/CNN/Transformer/MLP/etc.).

Core requirements:

- Highly modular: small reusable blocks + clear separation of responsibilities.
- A single **orchestrator class** that wires everything: `Model`.
- Explicit **layer initialization** inside modules.
- Config-driven construction (`config.model.*`) with minimal hardcoding.
- Maintain consistent naming, forward signatures, and extensibility.

---

## 2) Mandatory Design Principles (Non-Negotiable)

### 2.1 Modularity
The module must be decomposed into:
- **Blocks**: repeated building units (e.g., ConvBlock, AttnBlock, GNNBlock, MLPBlock).
- **Backbone**: stack of blocks that produces a feature embedding.
- **Pooling/Aggregation** (optional): converts sequence/graph/grid features to a fixed-size vector.
- **Head(s)**: task-specific output modules (regression/classification/multi-task).
- **Orchestrator**: `Model` that combines backbone + pooling + head(s).

### 2.2 Orchestrator Pattern
A class named `Model(nn.Module)` must exist and must:
- Instantiate components in `__init__`
- Own the end-to-end forward path in `forward()`
- Optionally return intermediate features when `return_features=True`

### 2.3 Explicit Initialization
Each module that contains learnable layers must implement explicit initialization logic:
- Linear/Conv weights: Xavier or Kaiming
- Bias: zeros
- Norm layers: weights=1, bias=0
Initialization must not be left entirely to PyTorch defaults.

### 2.4 Config-Driven Construction
No magic constants inside modules. Hyperparameters are read from `config.model.*` and passed into constructors.

---

## 3) Required Class Skeleton (Generic)

The generated module must define classes that follow this hierarchy:

1) `Block(nn.Module)` (one or more, specialized by modality)
- Example names: `CoreBlock`, `FeatureBlock`, `EncoderBlock`
- Must support:
  - main transform path (e.g., conv/attn/message-passing)
  - optional residual/skip connection
  - optional normalization
  - optional activation
  - optional dropout
- Must include `_initialize_weights(...)` or similar.

2) `Pooling/Aggregation` modules (optional but supported)
- Must provide a consistent interface:
  - `forward(features, context) -> pooled_features`
- Examples:
  - `MeanPooling`, `MaxPooling`, `AttentionPooling`, `GlobalPooling`
- If pooling is not required, implement `IdentityPooling`.

3) `Backbone(nn.Module)`
- Builds a list/stack of blocks (e.g., ModuleList)
- Provides a property:
  - `out_dim` (final feature dimension)
- Forward returns either:
  - per-element features (sequence/graph/grid)
  - or pooled features (if pooling is internal)

4) `Head(nn.Module)` (one or more)
- Example names:
  - `RegressionHead`
  - `ClassificationHead`
  - `TaskHead`
- Must be configurable:
  - hidden dims, dropout, output dim
- Must include explicit initialization (especially last layer).

5) `Model(nn.Module)` (Orchestrator)
- Must instantiate:
  - `self.backbone`
  - `self.pool` (optional; can be identity)
  - `self.head` or `self.<task_head_name>`
  - `self.norm` (feature normalization after backbone, optional but template-aligned)
  - `self.feature_dim`
- Must implement:
  - `forward(inputs, return_features: bool = False)`

---

## 4) Forward Interface Contract

### 4.1 Inputs
The agent must generate code that accepts a generic `inputs` object. It must not hardcode fields like `.x`, `.edge_index`, `.batch`.

Instead, it must support one of these generic patterns:

- Pattern A (dict-like):
  - `inputs["features"]`, optional `inputs["mask"]`, `inputs["context"]`
- Pattern B (attribute-like):
  - `inputs.features`, optional `inputs.mask`, `inputs.context`

The generator must include a small adapter that extracts:
- `features` (required)
- `context` (optional)
- `mask` (optional)

### 4.2 Outputs
`Model.forward()` must return:
- predictions (tensor)
- and if `return_features=True`, return:
  - `(predictions, features_dict)`

`features_dict` must include at least:
- `"backbone"`: backbone-level representation
- `"predictions"`: final output

---

## 5) Config Schema (Generic, Minimal)

The generated model must rely on a minimal config contract:

Required:
- `config.model.input_dim`
- `config.model.hidden_dims` (list[int])
- `config.model.output_dim`
- `config.model.dropout`

Optional:
- `config.model.use_norm` (bool)
- `config.model.norm_type` (string: "layernorm" | "batchnorm" | "none")
- `config.model.activation` (string: "relu" | "gelu" | "silu" | "none")
- `config.model.pooling` (string: "mean" | "max" | "attn" | "none")
- `config.model.pooling_params` (dict-like)
- `config.model.head_hidden_dims` (list[int])  (if different from backbone)
- `config.model.head_dropout`

The generator must:
- Provide robust defaults when optional fields are missing.
- Never assume modality-specific config keys unless explicitly provided.

---

## 6) Initialization Rules (Standardized)

The generator must apply these initialization policies:

- `nn.Linear` / `nn.Conv*`:
  - Kaiming uniform for ReLU/SILU
  - Xavier uniform for GELU or when unspecified
- Bias: constant 0.0
- `nn.LayerNorm` / `nn.BatchNorm*`:
  - weight = 1.0
  - bias = 0.0

The last layer of the head should be initialized with a smaller gain (template-aligned), e.g.:
- Xavier uniform with gain ~0.01, bias zero.

---

## 7) Extensibility Hooks (Must Include)

The agent must include extension points that do not break the skeleton:

- Multiple heads:
  - allow `self.heads = nn.ModuleDict({...})` if `config.model.multi_head=True`
- Multiple pooling strategies:
  - implement pooling factory `build_pooling(config, feature_dim)`
- Optional skip connections per block:
  - skip projection if dims differ

---

## 8) Forbidden Specificity (Important)

The generator must NOT:
- Assume graph-only inputs (no `.edge_index`, `.batch` as mandatory).
- Assume PyTorch Geometric is required.
- Hardcode a specific block type (GAT/CNN/Transformer) unless the user asks.
- Require a specific dataset object structure beyond generic feature extraction.

---

## 9) Output Deliverable

The agent must output **a single Python module** implementing the above structure, including:

- Generic blocks
- Backbone
- Pooling factory
- Head
- Orchestrator Model
- Explicit initialization

The final code must be readable, modular, and directly pluggable into the training template.

