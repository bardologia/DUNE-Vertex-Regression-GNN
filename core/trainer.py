import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from core.logger import ShapeLogger, ModelSummary, Tracker
from tqdm import tqdm
import gc
from pathlib import Path


class EarlyStopping:
    def __init__(self, config, logger, tracker):
        self.config           = config
        self.logger           = logger
        self.tracker          = tracker
        self.patience         = self.config.early_stopping.patience
        self.min_delta        = self.config.early_stopping.min_delta
        self.restore_best     = self.config.early_stopping.restore_best
        
        self.logger.section("[Early Stopping]")
        self.logger.subsection(f"Patience       : {self.patience}")
        self.logger.subsection(f"Min Delta      : {self.min_delta}")
        self.logger.subsection(f"Restore Best   : {self.restore_best} \n")

        self.best_loss        = None
        self.counter          = 0
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_state(model)
            self.counter = 0
            stop = False

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_state(model)
            self.tracker.log_scalar("early_stopping/best_val_loss", self.best_loss, epoch)
            stop = False

        else:
            self.counter += 1
            stop = (self.counter >= self.patience)

        self.tracker.log_scalar("early_stopping/counter", self.counter, epoch)

        if stop and self.restore_best:
            self.restore_model(model)

        return stop

    def _save_state(self, model):
        if self.restore_best:
            self.best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def restore_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state, strict=True)


class Warmup:
    def __init__(self, optimizer, config, logger, tracker):
        self.config              = config
        self.logger              = logger
        self.tracker             = tracker
        self.optimizer           = optimizer
        self.warmup_steps        = self.config.warmup.warmup_steps
        self.warmup_start_factor = self.config.warmup.warmup_start_factor
        self.enabled             = self.config.warmup.warmup_enabled
        self.base_lrs            = [group['lr'] for group in optimizer.param_groups]
        self.current_step        = 0
        self.warmup_finished     = False
        self._logged_completion  = False

        self.logger.section("[Warmup Scheduler]")
        self.logger.subsection(f"Warmup Enabled     : {self.enabled}")
        self.logger.subsection(f"Warmup Steps       : {self.warmup_steps}")
        self.logger.subsection(f"Warmup Start Factor: {self.warmup_start_factor} \n")

        if self.enabled and self.warmup_steps > 0:
            self._apply_warmup_factor(self.warmup_start_factor)
    
    def _apply_warmup_factor(self, factor: float) -> None:
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor
        
    def step(self) -> None:
        if not self.enabled or self.warmup_steps <= 0:
            return

        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            self._apply_warmup_factor(factor)
            self.tracker.log_scalar("warmup/lr_factor", factor, self.current_step)
        
        elif not self.warmup_finished:
            self._apply_warmup_factor(1.0)
            self.warmup_finished = True
            
            if not self._logged_completion:
                self.logger.info(f"Warmup completed at step {self.current_step}. Learning rates restored to base values.")
                self.tracker.log_scalar("warmup/lr_factor", 1.0, self.current_step)
                self._logged_completion = True
    
    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0


class Scheduler:
    def __init__(self, optimizer, warmup, config, logger, tracker):
        self.config    = config
        self.optimizer = optimizer
        self.warmup    = warmup
        self.logger    = logger
        self.tracker   = tracker
        self.scheduler = CosineAnnealingLR(optimizer, T_max=self.config.scheduler.epochs, eta_min=self.config.scheduler.eta_min)
     
        self.logger.section("[Learning Rate Scheduler]")
        self.logger.subsection(f"Scheduler Type    : CosineAnnealingLR")
        self.logger.subsection(f"Scheduler Params  : T_max={self.config.scheduler.epochs}")
        self.logger.subsection(f"Scheduler Eta Min : {self.config.scheduler.eta_min}")
        self.logger.subsection(f"Warmup Enabled    : {self.warmup.enabled} \n")
        
    def step(self, epoch: int) -> None:
        if self.warmup and not self.warmup.is_finished():
            return
        
        self.scheduler.step()
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            self.tracker.log_scalar(f"scheduler/lr_{group_name}", param_group['lr'], epoch)
    
    def state_dict(self) -> dict:
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)


class EMA:
    def __init__(self, model: nn.Module, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker
        self.enabled = self.config.ema.use_ema
        self.decay   = self.config.ema.ema_decay
        
        self.logger.section("[Exponential Moving Average (EMA)]")
        self.logger.subsection(f"EMA Enabled: {self.enabled}")
        self.logger.subsection(f"EMA Decay  : {self.decay} \n")

        self.shadow  = {}
        self.backup  = {}
        
        if self.enabled:
            self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int = None) -> None:
        if not self.enabled:
            return
        
        total_divergence = 0.0
        shadow_norm      = 0.0
        model_norm       = 0.0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                self.logger.warning(f"EMA: Initialized shadow for new parameter '{name}'")
                continue
            
            divergence        = (self.shadow[name] - param.detach()).norm().item()
            total_divergence += divergence
            
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            
            shadow_norm += self.shadow[name].norm().item()
            model_norm  += param.norm().item()
        
        if step is not None:
            self.tracker.log_scalar("ema/param_divergence", total_divergence, step)
            self.tracker.log_scalar("ema/shadow_norm",      shadow_norm, step)
            self.tracker.log_scalar("ema/model_norm",       model_norm, step)
            self.tracker.log_scalar("ema/norm_ratio",       shadow_norm / (model_norm + 1e-8), step)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        
        self.backup = {}
        
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "enabled" : self.enabled,
            "decay"   : self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}
        }

    def load_state_dict(self, state: dict) -> None:
        self.enabled = state['enabled']
        self.decay   = state['decay']
        self.shadow  = state['shadow']
        self.backup  = {}


class Checkpoint:
    def __init__(self, logger, tracker):
        self.logger = logger
        self.tracker = tracker
    
    def save(self, trainer, path: str, epoch: int) -> None:
        checkpoint = {
            "epoch"         : epoch,
            "global_step"   : trainer.global_step,
            "best_val_loss" : trainer.best_val_loss,
            "best_epoch"    : trainer.best_epoch,
            "best_metrics"  : trainer.best_metrics,
            "train_losses"  : trainer.train_losses,
            "val_losses"    : trainer.val_losses,
            
            "model_state_dict"     : trainer.model.state_dict(),
            "optimizer_state_dict" : trainer.optimizer.state_dict(),
            
            "config"         : getattr(trainer.config, "to_dict", lambda: None)() or str(trainer.config),
            "dataset_config" : getattr(trainer.dataset_config, "to_dict", lambda: None)() or str(trainer.dataset_config),
            "norm"           : trainer.norm, 

            "lr_scheduler_state_dict" : trainer.lr_scheduler.state_dict() if trainer.lr_scheduler else None,
            "ema_state_dict"          : trainer.ema.state_dict(),
            
            "early_stopping_state": {
                "best_loss"        : trainer.early_stopping.best_loss,
                "counter"          : trainer.early_stopping.counter,
                "best_model_state" : trainer.early_stopping.best_model_state,
            },
            
            "warmup_state": {
                "current_step"    : trainer.warmup.current_step,
                "warmup_finished" : trainer.warmup.warmup_finished,
            },
            
            "scaler_state_dict": trainer.scaler.state_dict() if trainer.scaler else None,
        }

        self.logger.info(f"Saving checkpoint at epoch {epoch} to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
    
    def load(self, trainer, path: str) -> int:
    
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
        
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        trainer.global_step   = checkpoint["global_step"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        trainer.best_epoch    = checkpoint["best_epoch"]
        trainer.best_metrics  = checkpoint["best_metrics"]
        trainer.train_losses  = checkpoint["train_losses"]
        trainer.val_losses    = checkpoint["val_losses"]
        
        if checkpoint["lr_scheduler_state_dict"] and trainer.lr_scheduler:
            trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        if checkpoint["ema_state_dict"]:
            trainer.ema.load_state_dict(checkpoint["ema_state_dict"])
        
        if "early_stopping_state" in checkpoint:
            es_state = checkpoint["early_stopping_state"]
            trainer.early_stopping.best_loss        = es_state["best_loss"]
            trainer.early_stopping.counter          = es_state["counter"]
            trainer.early_stopping.best_model_state = es_state["best_model_state"]
        
        if "warmup_state" in checkpoint:
            warmup_state = checkpoint["warmup_state"]
            trainer.warmup.current_step    = warmup_state["current_step"]
            trainer.warmup.warmup_finished = warmup_state["warmup_finished"]
        
        if checkpoint["scaler_state_dict"] and trainer.scaler:
            trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        epoch = checkpoint["epoch"]
        self.logger.info(f"Checkpoint loaded successfully. Resuming from epoch {epoch}")
        return epoch


class Loss:
    def __init__(self, logger, tracker):
        self.logger  = logger
        self.tracker = tracker

        self.logger.section("[Loss Function]")
        self.logger.subsection("Using Mean Squared Error (MSE) loss for regression \n")
    
    @staticmethod
    def mse_loss(pred, target):
        return F.mse_loss(pred, target)

    def __call__(self, pred, data, epoch):
        target     = data.y
        mse_loss   = self.mse_loss(pred, target)
        total_loss = mse_loss
        
        self.tracker.log_dict("loss", {
            "log_total_loss" : np.log1p(total_loss.item()),
            "log_mse_loss"   : np.log1p(mse_loss.item()),
        }, epoch)

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
        }


class Metrics:
    def __init__(self, verbose: bool, tracker, logger):
        self.verbose = verbose
        self.tracker = tracker
        self.logger  = logger
        
    def track_results(self, results: dict, epoch: int, stage: str = "validation"):
        self.tracker.log_dict(f"{stage}_mae", {
            "mae_x": float(results["mae_x"]),
            "mae_y": float(results["mae_y"]),
            "mae_z": float(results["mae_z"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_rmse", {
            "rmse_x": float(results["rmse_x"]),
            "rmse_y": float(results["rmse_y"]),
            "rmse_z": float(results["rmse_z"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_r2", {
            "r2_x": float(results["r2_x"]),
            "r2_y": float(results["r2_y"]),
            "r2_z": float(results["r2_z"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_median_error", {
            "median_error_x": float(results["median_error_x"]),
            "median_error_y": float(results["median_error_y"]),
            "median_error_z": float(results["median_error_z"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_std_error", {
            "std_error_x": float(results["std_error_x"]),
            "std_error_y": float(results["std_error_y"]),
            "std_error_z": float(results["std_error_z"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_mean_error", {
            "mean_error_x": float(results["mean_error_x"]),
            "mean_error_y": float(results["mean_error_y"]),
            "mean_error_z": float(results["mean_error_z"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_euclidian", {
            "euclidian_mean":   float(results["euclidian_mean"]),
            "euclidian_std":    float(results["euclidian_std"]),
            "euclidian_median": float(results["euclidian_median"]),
            "euclidian_max":    float(results["euclidian_max"]),
            "euclidian_min":    float(results["euclidian_min"]),
        }, epoch)

        self.tracker.log_dict(f"{stage}_overall_metrics", {
            "mae":          float(results["mae"]),
            "rmse":         float(results["rmse"]),
            "r2":           float(results["r2"]),
            "mean_error":   float(results["mean_error"]),
            "median_error": float(results["median_error"]),
            "std_error":    float(results["std_error"]),
        }, epoch)

    def calculate(self, epoch, preds, targets, stage="validation"):
        diff     = preds - targets
        abs_diff = torch.abs(diff)

        euclidian_mean   = torch.sqrt((diff ** 2).sum(dim=1)).mean().item()
        euclidian_std    = torch.sqrt((diff ** 2).sum(dim=1)).std().item()
        euclidian_median = torch.sqrt((diff ** 2).sum(dim=1)).median().item()
        euclidian_max    = torch.sqrt((diff ** 2).sum(dim=1)).max().item()
        euclidian_min    = torch.sqrt((diff ** 2).sum(dim=1)).min().item()

        mae  = abs_diff.mean().item()
        mse  = (diff ** 2).mean().item()
        rmse = float(np.sqrt(mse))

        ss_res = torch.sum(diff ** 2).item()
        ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        mae_per    = abs_diff.mean(dim=0)
        rmse_per   = torch.sqrt((diff ** 2).mean(dim=0))
        median_per = abs_diff.median(dim=0).values
        std_per    = abs_diff.std(dim=0)

        r2_per = []
        for i in range(preds.shape[1]):
            ss_res_i = torch.sum(diff[:, i] ** 2).item()
            ss_tot_i = torch.sum((targets[:, i] - targets[:, i].mean()) ** 2).item()
            r2_i     = 1 - (ss_res_i / ss_tot_i) if ss_tot_i > 0 else 0
            r2_per.append(r2_i)

        if self.verbose:
            self.logger.section(f"[Epoch {epoch}] Evaluation Metrics - {stage.capitalize()}")
            self.logger.subsection(f"x : MAE={mae_per[0]:.4f}, RMSE={rmse_per[0]:.4f}, R2={r2_per[0]:.4f}, Median Error={median_per[0]:.4f}, Std Error={std_per[0]:.4f}")
            self.logger.subsection(f"y : MAE={mae_per[1]:.4f}, RMSE={rmse_per[1]:.4f}, R2={r2_per[1]:.4f}, Median Error={median_per[1]:.4f}, Std Error={std_per[1]:.4f}")
            self.logger.subsection(f"z : MAE={mae_per[2]:.4f}, RMSE={rmse_per[2]:.4f}, R2={r2_per[2]:.4f}, Median Error={median_per[2]:.4f}, Std Error={std_per[2]:.4f} \n")

        results = {
            "mae"          : mae,
            "rmse"         : rmse,
            "r2"           : r2,
            "mean_error"   : mae,
            "median_error" : float(abs_diff.median().item()),
            "std_error"    : float(abs_diff.std().item()),

            "mean_all_mae"    : mae,
            "mean_all_rmse"   : rmse,
            "mean_coord_mae"  : float(abs_diff.mean(dim=1).mean().item()),
            "mean_coord_rmse" : float(torch.sqrt((diff ** 2).mean(dim=1)).mean().item()),
    
            "mae_x" : float(mae_per[0].item()),
            "mae_y" : float(mae_per[1].item()),
            "mae_z" : float(mae_per[2].item()),
            
            "rmse_x" : float(rmse_per[0].item()),
            "rmse_y" : float(rmse_per[1].item()),
            "rmse_z" : float(rmse_per[2].item()),
            
            "r2_x" : r2_per[0],
            "r2_y" : r2_per[1],
            "r2_z" : r2_per[2],
            
            "median_error_x" : float(median_per[0].item()),
            "median_error_y" : float(median_per[1].item()),
            "median_error_z" : float(median_per[2].item()),
            
            "std_error_x" : float(std_per[0].item()),
            "std_error_y" : float(std_per[1].item()),
            "std_error_z" : float(std_per[2].item()),
            
            "mean_error_x" : float(mae_per[0].item()),
            "mean_error_y" : float(mae_per[1].item()),
            "mean_error_z" : float(mae_per[2].item()),

            "euclidian_mean"   : euclidian_mean,
            "euclidian_std"    : euclidian_std,
            "euclidian_median" : euclidian_median,
            "euclidian_max"    : euclidian_max,
            "euclidian_min"    : euclidian_min,
        }

        self.track_results(results, epoch, stage)
        return results


class Trainer:
    def __init__(self, model, norm, config, dataset_config, run_dir, logger):
        self.logger          = logger
        self.config          = config
        self.dataset_config  = dataset_config
        
        self.logger.section("[Training Start]")
        self.logger.subsection(f"Device Name   : {torch.cuda.get_device_name(0)}")
        self.logger.subsection(f"Total Memory  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        self.logger.subsection(f"CUDA Version  : {torch.version.cuda}")
        self.logger.subsection(f"Log Directory : {self.config.io.logdir} \n")
        
        self.checkpoint_path = Path(run_dir) / "best_model.pt"
        self.tracker         = Tracker(writer=self.config.io.writer)
        
        self.device = self.config.training.device
        self.model  = model.to(self.device)
      
        self.norm = norm
        self.target_mean = torch.tensor(norm['target_mean'], dtype=torch.float32).to(self.device)
        self.target_std  = torch.tensor(norm['target_std'],  dtype=torch.float32).to(self.device)

        self.epochs               = self.config.training.epochs
        self.validation_frequency = self.config.training.validation_frequency

        self.use_amp = self.config.training.use_amp and torch.cuda.is_available()
        self.scaler  = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.accumulation_steps = self.config.training.gradient_accumulation_steps
        self.param_groups       = self.make_param_groups()
        self.optimizer          = torch.optim.AdamW(self.param_groups, betas = self.config.optimizer.betas, eps = self.config.optimizer.eps)

        self.warmup         = Warmup(self.optimizer, self.config, self.logger, self.tracker)
        self.ema            = EMA(self.model, self.config, self.logger, self.tracker)
        self.early_stopping = EarlyStopping(self.config, self.logger, self.tracker)
        self.lr_scheduler   = Scheduler(self.optimizer, self.warmup, self.config, self.logger, self.tracker)
        self.criterion      = Loss(self.logger, self.tracker)
        self.metrics        = Metrics(self.config.training.verbose, self.tracker, logger=self.logger)
        self.checkpoint     = Checkpoint(self.logger, self.tracker)
        self.shape_logger   = ShapeLogger(model = self.model, logger = self.logger).attach()
        self.summary        = ModelSummary(logger = self.logger, model = self.model)
        self.summary.run()
        self.summary.save_markdown(os.path.join(self.config.io.logdir, "model_summary.md"), title="PPO Model Summary")
        
        self.best_val_loss = float("inf")
        self.best_epoch    = -1
        self.best_metrics  = {}
        self.global_step   = 0
        self.train_losses  = []
        self.val_losses    = []
              
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def make_param_groups(self):
        param_groups = []

        rh_params   = list(self.model.regression_head.parameters())
        pool_params = list(self.model.backbone.pool.parameters())
        
        pool_ids   = set(id(p) for p in pool_params)
        gps_params = [p for p in self.model.backbone.parameters() if id(p) not in pool_ids]
        gps_params += list(self.model.norm.parameters())

        param_groups.append({
            'params'       : rh_params,
            'lr'           : self.config.optimizer.lr_regression_head,
            'weight_decay' : self.config.optimizer.weight_decay_regression_head,
            'name'         : 'regression_head'
        })

        param_groups.append({
            'params'       : pool_params,
            'lr'           : self.config.optimizer.lr_pool,
            'weight_decay' : self.config.optimizer.weight_decay_pool,
            'name'         : 'pool'
        })

        param_groups.append({
            'params'       : gps_params,
            'lr'           : self.config.optimizer.lr_gcn,
            'weight_decay' : self.config.optimizer.weight_decay_gcn,
            'name'         : 'gcn'
        })

        self.logger.section("[Optimizer Parameter Groups]")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            self.logger.subsection(
                f"{group['name']} - LR: {group['lr']}, Weight Decay: {group['weight_decay']}, Parameters: {num_params:,}"
            )

        return param_groups
    
    def denormalize_targets(self, target):
        std = self.target_std
        mean = self.target_mean
        denormalized_target = target * std + mean
        
        return denormalized_target

    def forward(self, data, epoch):
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred = self.model(data)
            loss_dict = self.criterion(pred, data, epoch)
            self.shape_logger.detach()
        
        return pred, loss_dict
    
    def backward(self, loss, step: bool):
        if self.scaler:
            self.scaler.scale(loss).backward()
            if step:
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                if self.global_step % 100 == 0:
                    self.tracker.log_gradients(self.model, self.global_step, max_grad_norm=self.config.training.max_grad_norm)
                    self.tracker.log_optimizer(self.optimizer, self.global_step)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.warmup.step()
                self.global_step += 1
                self.ema.update(self.model, step=self.global_step)
        else:
            loss.backward()
            if step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                
                if self.global_step % 100 == 0:
                    self.tracker.log_gradients(self.model, self.global_step, max_grad_norm=self.config.training.max_grad_norm)
                    self.tracker.log_optimizer(self.optimizer, self.global_step)
                
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.warmup.step()
                self.global_step += 1
                self.ema.update(self.model, step=self.global_step)

    def train_epoch(self, train_loader, epoch, loader_len=None):
        self.model.train()
        total_loss   = 0
        num_batches  = 0
        batch_losses = []
                
        activation_hooks = []
        if epoch > 0 and epoch % 10 == 0:
            activation_hooks = self.tracker.log_activations(self.model, epoch)
    
        try:
            for batch_idx, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{self.epochs}", leave=False, total=len(train_loader)):
                data = data.to(self.device, non_blocking=True)

                pred, loss_dict = self.forward(data, epoch)
                loss = loss_dict["total_loss"] / self.accumulation_steps

                should_step = (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) >= len(train_loader)
                self.backward(loss, step=should_step)
                
                loss_value        = loss.item()
                total_loss       += loss_value * self.accumulation_steps
                num_batches      += 1
                batch_losses.append(loss_value * self.accumulation_steps)
        finally:
            for hook in activation_hooks:
                hook.remove()

        if epoch > 0 and epoch % 10 == 0:
            self.tracker.log_weights(self.model, epoch)
                
    def evaluate(self, loader, epoch, stage="validation"):
        self.model.eval()
        self.ema.apply_to(self.model)

        try:
            total_loss  = 0
            all_preds   = []
            all_targets = []
            num_batches = 0
                        
            with torch.no_grad():
                for data in tqdm(loader, desc=f"Evaluating {stage} Epoch {epoch+1}/{self.epochs}", leave=False, total=len(loader)):
                    data = data.to(self.device, non_blocking=True)
                    pred, loss_dict = self.forward(data, epoch)
                    total_loss += loss_dict["total_loss"].item()
                    all_preds.append(pred.cpu())
                    all_targets.append(data.y.cpu())
                    num_batches += 1
                    
            avg_loss = total_loss / max(1, num_batches)

            preds   = torch.cat(all_preds) 
            targets = torch.cat(all_targets) 

            preds_den   = self.denormalize_targets(preds.to(self.device)).cpu()
            targets_den = self.denormalize_targets(targets.to(self.device)).cpu()
            metrics     = self.metrics.calculate(epoch, preds_den, targets_den, stage=stage)

            self.tracker.log_error_vs_coordinates(preds_den, targets_den, epoch, stage=stage)
            self.tracker.log_outlier_detection(preds_den.to(self.device), targets_den.to(self.device), epoch, stage=stage)
        finally:
            self.ema.restore(self.model)

        results = {
            "avg_loss"    : avg_loss,
            "preds_den"   : preds_den,
            "targets_den" : targets_den,
            "metrics"     : metrics,
        }

        return results

    def train(self, train_loader, val_loader, test_loader):    
        self.logger.section("[Training Loop]")
        self.logger.subsection(f"Train loader size      = {len(train_loader)}")
        self.logger.subsection(f"Validation loader size = {len(val_loader)}")
        self.logger.subsection(f"Test loader size       = {len(test_loader)}")
        
        self._clear_memory()
        self.optimizer.zero_grad()
     
        train_loader_len = len(train_loader)
        if self.config.overfit.enabled:
            single_batch      = next(iter(train_loader))
            data_loader       = [single_batch] * train_loader_len
            eval_train_loader = [single_batch]
            self.logger.warning(f"Overfitting mode enabled: training on a single batch repeated {train_loader_len} times.")
        else:
            data_loader       = train_loader
            eval_train_loader = train_loader
        
        epochs = self.epochs
        for epoch in tqdm(range(epochs), desc="Training"):
            epoch_num = epoch + 1
            
            self.train_epoch(data_loader, epoch, train_loader_len)
     
            val_results   = self.evaluate(val_loader, epoch, stage="validation")
            self.logger.info(f"Epoch {epoch} - Validation Loss: {val_results['avg_loss']:.4f}")
            self.train_losses.append(val_results["avg_loss"])
            
            train_results = self.evaluate(eval_train_loader, epoch, stage="train")
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_results['avg_loss']:.4f}")
            self.val_losses.append(val_results["avg_loss"])

            self._clear_memory()

            loss_comparison = {
                "train_loss": train_results['avg_loss'],
                "val_loss"  : val_results["avg_loss"],
            }
            
            self.tracker.log_dict("loss_comparison", loss_comparison, epoch)

            if val_results["avg_loss"] < self.best_val_loss:
                self.best_val_loss = float(val_results["avg_loss"])
                self.best_epoch    = int(epoch_num)
                self.best_metrics  = dict(val_results["metrics"])
                self.checkpoint.save(self, self.checkpoint_path, epoch_num)
                self.logger.subsection(f"New best model found at epoch {epoch_num} with validation loss {self.best_val_loss:.4f}")
        
            self.lr_scheduler.step(epoch)
            if self.early_stopping(val_results["avg_loss"], self.model, epoch):
                self.logger.warning(f"Early stopping triggered at epoch {epoch_num}. Best model from epoch {self.best_epoch} restored.")
                break

        self.shape_logger.save_markdown(path=Path(self.config.io.logdir) / "tensor_shape.md", sort_by_layer=True)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        train_final_results      = self.evaluate(train_loader, checkpoint["epoch"], stage="final_train")
        validation_final_results = self.evaluate(val_loader,   checkpoint["epoch"], stage="final_validation")
        test_final_results       = self.evaluate(test_loader,  checkpoint["epoch"], stage="final_test")
        
        return train_final_results, validation_final_results, test_final_results

 




