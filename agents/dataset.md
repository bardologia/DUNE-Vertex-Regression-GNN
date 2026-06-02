# AI Agent Specification — Generic Dataset Pipeline Generator (Template-Constrained)

## 1) Objective

Generate a **dataset pipeline module** that follows the template’s architecture and philosophy:

- Modular pipeline with clear responsibilities.
- Multiple dedicated classes:
  1) Processing classes (preprocessing / cleaning / filtering)
  2) Feature creation classes (feature engineering / graph building / tensorization)
  3) Normalization class (fit on train split, apply to all splits)
  4) DataLoader factory class (creates train/val/test loaders consistently)
- Optional artifact caching (save/load) but kept modular and replaceable.
- Config-driven behavior (`config.data.*`, `config.preprocessing.*`, `config.split.*`, `config.dataloader.*`), without hardcoding dataset-specific assumptions.

The output must be **dataset-agnostic** and **architecture-agnostic** (graph/sequence/image/tabular), while keeping the same modular breakdown.

---

## 2) Required Classes (Must Exist)

The generated module must define these classes (names may vary, but these roles must exist as distinct classes):

1) **Processor**
   - Example names: `DatasetProcessor`, `Preprocessor`, `DataProcessor`
   - Responsibility: clean/filter/transform raw samples.
   - Must be independent of model/training.

2) **Feature Builder**
   - Example names: `FeatureBuilder`, `GraphBuilder`, `Tensorizer`, `Featurizer`
   - Responsibility: convert processed samples into model-ready tensors/objects.
   - Must not perform train/val/test splitting.

3) **Normalizer**
   - Example names: `DatasetNormalizer`, `Normalizer`
   - Responsibility:
     - `compute_stats(splits, source_split="train")`
     - `normalize(splits)`
   - Must compute statistics only on a designated source split (default train).
   - Must return normalization metrics (mean/std/etc.) needed by training for de-normalization.

4) **Split / Loader Orchestrator**
   - Example names: `DatasetLoader`, `SplitBuilder`, `DataModule`, `DataPipeline`
   - Responsibility:
     - load raw or cached artifacts
     - build splits (train/val/test)
     - attach targets/labels
     - normalize
     - create DataLoaders

Additionally (optional but template-aligned):

5) **Artifact Builder**
   - Example names: `DatasetBuilder`, `ArtifactBuilder`
   - Responsibility:
     - build dataset artifacts once and save (chunked saving allowed)
     - store metadata + config snapshot + target table

---

## 3) Role Separation (Strict)

The generator must enforce these separations:

- Processing must not depend on feature building internals.
- Feature building must not handle split logic.
- Normalization must not handle split logic or feature construction.
- DataLoader creation must be its own class or clearly separated factory method.

No single class should do everything.

---

## 4) Standard Pipeline Stages (Template-Aligned)

The orchestration class (e.g., `DatasetLoader` or `DataPipeline`) must implement a `run()` method that follows this high-level order:

1) Load artifacts or raw data:
   - `raw_samples, targets = load()`
2) Process raw samples:
   - `processed_samples = processor.run(raw_samples)` (or equivalent)
3) Build features:
   - `dataset_objects = feature_builder.build(processed_samples)`
4) Split indices (train/val/test):
   - stratified if configured, otherwise random
5) Attach labels/targets to dataset objects:
   - consistent interface across splits
6) Normalize splits:
   - fit stats on train only
   - apply to all splits
   - return `norm_metrics`
7) Create DataLoaders:
   - `train_loader, val_loader, test_loader`
8) Return:
   - `splits`, `norm_metrics`, `saved_config` (optional)

This ordering must be explicit and readable.

---

## 5) Processing Classes (Generic Requirements)

Processing classes must:

- Be config-driven (thresholds, transforms, filters).
- Operate on collections of samples.
- Support reporting/logging of:
  - how many samples removed/changed
  - summary statistics pre/post

Examples of processing steps (generic placeholders, not hardcoded):
- outlier detection
- clipping / sanitization
- scaling transforms
- log transforms
- stochastic effects (seed-controlled)

The generator must not embed domain-specific operations unless provided.

---

## 6) Feature Builder (Generic Requirements)

The feature builder must:

- Convert each processed sample into a model-ready structure:
  - Tensor dict (`{"features": ..., "context": ...}`)
  - or framework-specific object (e.g., PyG `Data`) if configured.
- Support optional parallelization (Ray/multiprocessing) behind a flag:
  - must be optional and isolated, so disabling it does not change logic.
- Provide deterministic behavior controlled by `config.split.random_state` when randomness exists.

---

## 7) Normalization (Required Contract)

The normalizer must:

### 7.1 Compute Statistics
Implement:

- `compute_stats(splits, source_split="train")`

It must compute mean/std (or other) for relevant tensors, depending on dataset object type.

### 7.2 Apply Normalization
Implement:

- `normalize(splits) -> splits`

Must apply stats consistently to every split.

### 7.3 Return Metrics
The pipeline must return a `norm_metrics` dictionary with at least:

- `"feature_mean"`
- `"feature_std"`
- `"target_mean"` (if supervised and numeric)
- `"target_std"` (if supervised and numeric)

If the dataset has multiple feature groups (nodes/edges/sequence/etc.), metrics can be namespaced but must remain generic and consistent.

---

## 8) Splitting and Balancing (Generic Requirements)

The pipeline must support:

- Random split: train/val/test ratios
- Stratified split when:
  - classification labels exist, or
  - a custom stratification function is provided

The split logic must be configurable via:

- `config.split.test_size`
- `config.split.val_ratio`
- `config.split.random_state`
- `config.split.stratify` (bool)
- Optional: `config.split.n_bins` for continuous target stratification

The generator must not assume specific target dimensions.

---

## 9) DataLoader Factory (Required)

A dedicated loader creation layer must exist:

- Example: `DataLoaderFactory` or `DatasetLoader.create_dataloaders(...)`

Requirements:
- Creates `train_loader`, `val_loader`, `test_loader`
- Uses config-driven:
  - batch_size
  - shuffle
  - num_workers
  - pin_memory
  - drop_last
- Must support switching backend:
  - standard PyTorch DataLoader
  - graph/sequence specialized loader (only if configured)

---

## 10) Artifact Save/Load (Optional but Template-Compatible)

If artifact caching is enabled:

- Save dataset in chunks (for large datasets)
- Save metadata:
  - number of chunks
  - chunk size
  - dataset length
- Save targets
- Save a config snapshot

Load must reconstruct full dataset deterministically.

Artifact IO must be in its own class (`DatasetBuilder` / `ArtifactBuilder`) or isolated methods.

---

## 11) Logging Requirements

All main classes should accept a `logger` and emit:

- `logger.section("[...]")`
- `logger.subsection("...")`

At minimum, log:
- number of samples loaded
- number removed/modified by processing
- split sizes
- normalization stats before/after (train split)
- dataloader sizes/config

---

## 12) Output Contract (Pipeline Return)

The orchestrator `run()` must return:

- `splits`: dict with keys `"train"`, `"val"`, `"test"`
- `norm_metrics`: dict of normalization statistics
- `saved_config`: optional config snapshot loaded from artifacts (or None)

Optionally, it may return DataLoaders directly, but splits must remain available.

---

## 13) Forbidden Specificity

The generator must NOT:

- Hardcode graph fields (`x`, `edge_index`, `batch`) as mandatory.
- Hardcode any particular data format (CSV, parquet, dataframe) unless provided.
- Couple preprocessing to a specific domain transform.
- Mix normalization with splitting logic.
- Mix dataloader creation with feature building.

Keep the pipeline generic and modular.
