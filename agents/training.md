# AI Agent Specification — Deep Learning Training Code Generator (Template-Constrained)

## 1. Purpose

This agent generates a PyTorch training module that strictly follows a predefined structural template.

The generated code must:

- Preserve the exact architectural structure:
  - EarlyStopping
  - Warmup
  - Scheduler
  - EMA
  - Checkpoint
  - Loss
  - Metrics
  - Trainer
- Preserve execution order and training logic.
- Adapt only problem-specific components:
  - Model
  - Dataset interface
  - Task type (regression/classification/custom)
  - Loss definition
  - Metric computation
  - Parameter groups

The framework must not be simplified or restructured.

---

## 2. Structural Invariants (MANDATORY)

### 2.1 Required Classes

The generated module must define the following classes (names must match exactly):

- EarlyStopping
- Warmup
- Scheduler
- EMA
- Checkpoint
- Loss
- Metrics
- Trainer

All must exist in the same module unless explicitly instructed otherwise.

---

### 2.2 Execution Flow (STRICT ORDER)

#### 2.2.1 Initialization Order

The Trainer.__init__ must initialize components in this order:

1. logger
2. config
3. dataset_config
4. tracker
5. device
6. model moved to device
7. AMP/scaler selection (torch.amp.autocast, torch.amp.GradScaler)
8. param_groups = make_param_groups()
9. optimizer = AdamW(param_groups, betas=..., eps=...)
10. warmup = Warmup(optimizer, config, logger, tracker)
11. ema = EMA(model, config, logger, tracker)
12. early_stopping = EarlyStopping(config, logger, tracker)
13. lr_scheduler = Scheduler(optimizer, warmup, config, logger, tracker)
14. criterion = Loss(logger, tracker)
15. metrics = Metrics(verbose, tracker, logger)
16. checkpoint = Checkpoint(logger, tracker)
17. shape_logger = ShapeLogger(model=model, logger=logger).attach()
18. summary = ModelSummary(logger=logger, model=model)
19. summary.run()
20. summary.save_markdown(...)

The agent must not reorder these steps.

---

### 2.2.2 Training Loop Order

The Trainer.train() method must follow:

for epoch in range(epochs):
    train_epoch(...)
    val_results = evaluate(val_loader, epoch, stage="validation")
    train_results = evaluate(train_loader_or_single_batch, epoch, stage="train")
    log loss_comparison

    if val improves:
        update best trackers
        checkpoint.save(...)

    lr_scheduler.step(epoch)

    if early_stopping(val_loss, model, epoch):
        break

load best checkpoint weights into model

final evaluate on:
    train
    validation
    test

return final results

---

## 3. Required Features (Always Present)

### 3.1 AMP (Mixed Precision)

- Controlled by config.training.use_amp and CUDA availability.
- Must use:
  - torch.amp.autocast("cuda", enabled=...)
  - torch.amp.GradScaler("cuda")

Must support both AMP and non-AMP branches.

---

### 3.2 Gradient Accumulation

- Controlled by config.training.gradient_accumulation_steps
- Must:
  - Divide loss by accumulation steps before backward
  - Only call optimizer step when accumulation condition is met

---

### 3.3 Gradient Clipping

Must apply:

torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

---

### 3.4 Parameter Groups

The agent must generate Trainer.make_param_groups() that:

- Creates named groups
- Assigns LR and weight decay per group
- Logs parameter counts

Template structure:

param_groups.append({
    "params": ...,
    "lr": ...,
    "weight_decay": ...,
    "name": ...
})

If the model does not contain the template submodules, the agent must map groups to equivalent logical submodules while preserving this structure.

---

### 3.5 Warmup Scheduler

Must:

- Be step-based
- Modify optimizer LRs directly
- Ramp from warmup_start_factor to 1.0 over warmup_steps
- Log warmup/lr_factor

---

### 3.6 Main Scheduler (Cosine Annealing)

Must use torch.optim.lr_scheduler.CosineAnnealingLR

- Must not step during warmup
- Must log per-group learning rate scalars named scheduler/lr_<group_name>

---

### 3.7 EMA (Exponential Moving Average)

Must:

- Maintain shadow weights for trainable parameters
- Update every optimizer step:
  shadow = decay * shadow + (1 - decay) * param
- Support:
  - update(model, step)
  - apply_to(model)
  - restore(model)
  - state_dict()
  - load_state_dict(state)

Must log:
- ema/param_divergence
- ema/shadow_norm
- ema/model_norm
- ema/norm_ratio

---

### 3.8 Early Stopping

Must support:

- patience
- min_delta
- restore_best

Must track:
- best_loss
- counter
- best_model_state (CPU-cloned tensors)

Must log:
- early_stopping/counter
- early_stopping/best_val_loss

---

### 3.9 Checkpointing

Checkpoint must store at minimum:

- epoch
- global_step
- best_val_loss
- best_epoch
- best_metrics
- train_losses
- val_losses
- model_state_dict
- optimizer_state_dict
- lr_scheduler_state_dict (if present)
- ema_state_dict
- early_stopping_state
- warmup_state
- scaler_state_dict (if AMP)
- config
- dataset_config
- norm (if used)

Must implement:
- Checkpoint.save(trainer, path, epoch)
- Checkpoint.load(trainer, path) -> epoch

---

## 4. Adaptable Components (Allowed to Change)

### 4.1 Loss

The Loss class must preserve interface:

__call__(pred, data, epoch) -> dict

Must return a dictionary containing:
- total_loss (required)

The agent may adapt:
- Regression loss
- Classification loss
- Multi-task loss
- Auxiliary losses

Loss must log through tracker.log_dict("loss", ...)

---

### 4.2 Metrics

The Metrics class must:

- Provide calculate(epoch, preds, targets, stage)
- Provide track_results(results, epoch, stage)
- Return a metrics dictionary

The agent may adapt metrics for:
- Regression
- Classification
- Structured outputs

Metrics must log scalar dictionaries to tracker.

---

### 4.3 Target Handling

Default:
- targets = data.y

If different target field is specified, adapt extraction while preserving Trainer.forward() signature.

If normalization exists:
- Implement denormalize_targets()

If classification:
- No denormalization

---

## 5. Logging Requirements

Must log:

- Device information
- CUDA version
- Memory
- Log directory
- Optimizer parameter groups

Tracker must log:

- Scalars
- Dictionaries
- Gradients every N steps
- Optimizer stats every N steps
- Optional activations every N epochs
- Optional weight histograms every N epochs

---

## 6. Trainer Required Methods

Trainer must implement:

- _clear_memory()
- make_param_groups()
- denormalize_targets() (if regression normalization is used)
- forward(data, epoch)
- backward(loss, step)
- train_epoch(train_loader, epoch, loader_len=None)
- evaluate(loader, epoch, stage="validation")
- train(train_loader, val_loader, test_loader)

All method names must match exactly.

---

## 7. Overfit Mode Support

If config.overfit.enabled is true:

- Use a single batch repeated for training
- Evaluate training using that single batch
- Emit warning log

---

## 8. Forbidden Changes

The agent must NOT:

- Remove or merge classes
- Remove EMA, Warmup, EarlyStopping, Checkpoint
- Remove AMP branch
- Remove gradient accumulation logic
- Replace CosineAnnealingLR unless explicitly requested
- Rename core methods/classes
- Reorder training loop steps

Structural fidelity is the highest priority.

---

## 9. Agent Inputs

The generator must accept:

- task_type
- target_field
- output_dim
- loss_spec
- metrics_spec
- model_param_groups_spec
- normalization_spec (optional)
- device_spec

---

## 10. Agent Output

The agent must output a single Python module that:

- Preserves full structural template
- Only adapts task-specific components
- Maintains restartability and checkpoint compatibility
- Is production-ready
- Preserves architectural philosophy of the original template

Structural fidelity over brevity.
