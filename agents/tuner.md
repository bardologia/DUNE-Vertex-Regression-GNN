# AI Agent Specification — Generic Hyperparameter Tuner Generator (Template-Constrained)

## 1) Objective

Generate a **generic hyperparameter tuning module** that follows the template’s structure and philosophy:

- Modular
- Has a single orchestration class: `Tuner`
- Uses Optuna with:
  - a sampler (default TPE)
  - a pruner (default MedianPruner)
- Has a strict separation between:
  - suggesting hyperparameters
  - applying them to config
  - building model/trainer/loaders
  - running an objective loop
  - summarizing results
  - applying best params back into global config

The output must remain **architecture-agnostic** (GNN/CNN/Transformer/MLP) and **dataset-agnostic** while preserving the same method layout.

---

## 2) Required Class and Methods (Must Exist)

A class named `Tuner` must exist with these public methods:

- `__init__(norm_metrics=None, dataset_config=None, config=None, logger=None)`
- `_suggest_hyperparameters(trial)`
- `_objective(trial, train_dataset, val_dataset)`
- `optimize(train_dataset, val_dataset, n_trials)`
- `build_best_configs(epochs=...)`

Method names must match exactly.

---

## 3) Required Initialization Contract (Tuner.__init__)

The `Tuner` constructor must store:

- `self.study`
- `self.norm_metrics`
- `self.dataset_config`
- `self.config`
- `self.logger`

And must read these fields from config (if present):

- `self.epochs        = config.tuning.epochs`
- `self.n_trials      = config.tuning.n_trials`
- `self.warmup_trials = config.tuning.warmup_trials`
- `self.random_state  = config.split.random_state`
- `self.warmup_steps  = config.tuning.warmup_steps`

If any field is missing, the generator must:
- provide safe defaults
- and keep the same attribute names.

---

## 4) Hyperparameter Suggestion (Template-Aligned but Generic)

### 4.1 `_suggest_hyperparameters(trial)` Output Contract

Must return a **flat dictionary** mapping:

- hyperparameter_name -> suggested_value

Constraints:
- Keys must be stable and deterministic.
- Naming convention:
  - Use prefixes grouping by subsystem:
    - `model/*` or `model_`
    - `optimizer/*` or `optimizer_`
    - `training/*` or `training_`
  - Use indexed keys for variable-depth architectures:
    - `layer_width_0`, `layer_width_1`, ...
    - `head_width_0`, ...

### 4.2 Suggestion Types (Supported)

The generator must support:
- `trial.suggest_categorical`
- `trial.suggest_int`
- `trial.suggest_float` (with optional `log=True` and `step=`)

The default generic set should include:
- Backbone widths (tuple/list)
- Head widths (tuple/list)
- Dropout(s)
- Learning rates (per param group OR global)
- Weight decay(s)
- Optional: attention heads, kernel sizes, embedding dim, etc. only if config indicates such fields exist

The generator must not hardcode GNN-specific keys unless the user asks.

---

## 5) Applying Hyperparameters to Config (Strict Pattern)

The `_objective()` method must:

1) Call `_suggest_hyperparameters(trial)`
2) Apply returned values into `self.config` (in-place)

Rules:
- Must map each suggested param to a config field.
- Mapping must be explicit and readable, not dynamic attribute magic.

Example pattern (generic):

- `self.config.model.hidden_dims = params["model_hidden_dims"]`
- `self.config.model.dropout = params["model_dropout"]`
- `self.config.optimizer.lr = params["optimizer_lr"]`
- `self.config.optimizer.weight_decay = params["optimizer_weight_decay"]`

If the training template uses param groups:
- The tuner must support suggesting per-group LRs/WDs using keys like:
  - `optimizer_lr_group_<name>`
  - `optimizer_wd_group_<name>`

---

## 6) Objective Loop Structure (Strict, Template-Constrained)

The `_objective(trial, train_dataset, val_dataset)` must:

### 6.1 Enforce Tuning Training Overrides
It must set (or override) at least:

- `self.config.training.epochs = self.config.tuning.epochs`
- `self.config.training.verbose = False`

Batch size may be overridden, but must be controlled through config.

### 6.2 Trial Directory
Must create a trial-specific directory:

- `trial_log_dir = Path(self.config.io.tunerdir) / f"optuna_trial_{trial.number}"`
- Then set `self.config.io.tunerdir = str(trial_log_dir)` (or equivalent field)

### 6.3 Build Model, Loaders, Logger, Trainer

Must instantiate in this order:

1) `model = Model(self.config)`
2) `train_loader = DataLoader(train_dataset, ...)`
3) `val_loader = DataLoader(val_dataset, ...)`
4) `logger = Logger(log_dir=self.config.io.tunerdir, name=f"trial_{trial.number}", level="WARNING")`
5) `trainer = Trainer(model=model, norm=self.norm_metrics, config=self.config, dataset_config=self.dataset_config, run_dir=trial_log_dir, logger=logger)`

DataLoader class must be configurable:
- Default: `torch.utils.data.DataLoader`
- If dataset is PyG, allow `torch_geometric.loader.DataLoader`
The generator must select based on config or dataset type hint; it must not hardcode PyG.

### 6.4 Track Best Validation
Must use:

- `best_val = float("inf")`
- Update when `val_loss < best_val`

### 6.5 Epoch Loop
Must run:

for epoch in range(self.config.tuning.epochs):
- Run **one epoch of training**
  - Either `trainer.train_epoch(...)` or a lightweight equivalent consistent with your Trainer API
- Evaluate validation:
  - `val_results = trainer.evaluate(val_loader, epoch, stage="validation")`
  - `val_loss = val_results["avg_loss"]`
- Report to optuna:
  - `trial.report(val_loss, step=epoch)`
- Pruning:
  - if `trial.should_prune()` raise `optuna.TrialPruned()`

### 6.6 Scheduler and Early Stopping
Must step LR scheduler and early stopping in-template order:

- Scheduler step occurs once per epoch.
- Early stopping called with current val loss.

The agent must match the Trainer’s actual method signature:
- If `trainer.lr_scheduler.step(epoch)` is required, pass `epoch`.
- If `trainer.lr_scheduler.step()` is used, call without args.
- If early stopping requires `epoch`, pass it.

### 6.7 Resource Cleanup
Must close:
- `logger.close()` (if exists)
- `self.config.io.writer.close()` (if exists)

Must run cleanup inside `finally`.

### 6.8 Trial User Attributes
Must store at least:
- `trial.set_user_attr("val_loss_curve", trainer.val_losses)` (or equivalent)

### 6.9 Objective Return
Return `best_val`.

---

## 7) Study Setup (optimize)

The `optimize(train_dataset, val_dataset, n_trials)` method must:

1) Set optuna verbosity to warning
2) Create sampler:
   - Default: `optuna.samplers.TPESampler(seed=self.config.split.random_state)`
3) Create study:
   - direction = "minimize" (default)
   - pruner = `optuna.pruners.MedianPruner(n_startup_trials=self.warmup_trials, n_warmup_steps=self.warmup_steps)`
4) Call:
   - `self.study.optimize(lambda trial: self._objective(trial, train_dataset, val_dataset), n_trials=n_trials, show_progress_bar=True)`

Then log results:

- Best value
- Best params (iterate `self.study.best_trial.params`)

Return `self.study`.

---

## 8) Applying Best Hyperparameters (build_best_configs)

The `build_best_configs(epochs=...)` method must:

1) Read:
   - `best_params = self.study.best_trial.params`
2) Apply them to `self.config` using the same mapping used in `_objective`.
3) Set:
   - `self.config.training.epochs = epochs`
   - optionally restore a normal (non-tuning) batch size from config defaults
4) Log a structured summary of what was applied.
5) Return `self.config`.

---

## 9) Genericity Rules (Important)

The tuner generator must NOT:

- Assume any specific architecture (GAT/GCN/CNN/Transformer) unless specified.
- Assume any specific param group names (e.g., regression_head/pool/gcn) unless provided.
- Assume a specific DataLoader backend (PyG) unless specified.

Instead, it must be driven by:
- `config.tuning.search_space` (if provided)
- or a minimal default generic space:
  - hidden_dims
  - dropout
  - lr
  - weight_decay

---

## 10) Required Extension Point: Search Space Definition

The generated tuner must support a config-defined search space, e.g.:

- `config.tuning.search_space` as a list of dict entries:

Each entry defines:
- `name`
- `type` ("float" | "int" | "categorical")
- `path` (where to apply in config, e.g., "model.dropout")
- `args` (range/list/step/log)

If `search_space` is absent, the tuner must fall back to a default generic set.

---

## 11) Output Deliverable

The agent must output a single Python module implementing:

- `Tuner` class with the required methods and flow
- Optuna sampler + pruner setup
- Objective loop aligned to Trainer’s API
- Clean logging and cleanup
- A generic, configurable hyperparameter search space mechanism
