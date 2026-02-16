
import optuna
from pathlib import Path

from torch_geometric.loader import DataLoader
from .model import Model
from .trainer import Trainer
from core.logger import Logger

class Tuner:
    def __init__(self, norm_metrics=None, dataset_config=None, config=None, logger=None):
        self.study          = None
        self.norm_metrics   = norm_metrics
        self.dataset_config = dataset_config or {}
        self.config         = config
        self.logger         = logger
        self.epochs         = self.config.tuning.epochs
        self.n_trials       = self.config.tuning.n_trials
        self.warmup_trials  = self.config.tuning.warmup_trials
        self.random_state   = self.config.split.random_state
        self.warmup_steps   = self.config.tuning.warmup_steps

    def _suggest_hyperparameters(self, trial):
        gcn_layers      = 3
        gcn_hidden_dims = tuple(trial.suggest_categorical(f"gcn_hidden_{i}", [128, 192, 256, 384]) for i in range(gcn_layers))
        
        regression_layers      = 3
        regression_hidden_dims = tuple(trial.suggest_categorical(f"regression_hidden_{i}", [128, 256, 512]) for i in range(regression_layers))
        
        gat_dropout        = trial.suggest_float("gat_dropout", 0.0, 0.2, step=0.05)
        regression_dropout = trial.suggest_float("regression_dropout", 0.0, 0.2, step=0.05)
        gat_heads          = trial.suggest_categorical("gat_heads", [4, 6, 8])
        
        lr_regression_head = trial.suggest_float("lr_regression_head", 5e-5, 2e-3, log=True)
        lr_pool            = trial.suggest_float("lr_pool", 1e-5, 1e-3, log=True)
        lr_gcn             = trial.suggest_float("lr_gcn", 1e-5, 1e-3, log=True)
        
        weight_decay_regression_head = trial.suggest_float("weight_decay_regression_head", 1e-6, 1e-4, log=True)
        weight_decay_pool            = trial.suggest_float("weight_decay_pool", 1e-6, 1e-4, log=True)
        weight_decay_gcn             = trial.suggest_float("weight_decay_gcn", 1e-6, 5e-4, log=True)
        
        return {
            'gcn_hidden_dims'              : gcn_hidden_dims,
            'regression_hidden_dims'       : regression_hidden_dims,
            'gat_dropout'                  : gat_dropout,
            'regression_dropout'           : regression_dropout,
            'gat_heads'                    : gat_heads,
            'lr_regression_head'           : lr_regression_head,
            'lr_pool'                      : lr_pool,
            'lr_gcn'                       : lr_gcn,
            'weight_decay_regression_head' : weight_decay_regression_head,
            'weight_decay_pool'            : weight_decay_pool,
            'weight_decay_gcn'             : weight_decay_gcn,
        }

    def _objective(self, trial, train_dataset, val_dataset):
        params = self._suggest_hyperparameters(trial)
        
        self.config.model.gcn_hidden_dims        = params['gcn_hidden_dims']
        self.config.model.regression_hidden_dims = params['regression_hidden_dims']
        self.config.model.gat_dropout            = params['gat_dropout']
        self.config.model.regression_dropout     = params['regression_dropout']
        self.config.model.gat_heads              = params['gat_heads']
        
        self.config.optimizer.lr_regression_head           = params['lr_regression_head']
        self.config.optimizer.lr_pool                      = params['lr_pool']
        self.config.optimizer.lr_gcn                       = params['lr_gcn']
        self.config.optimizer.weight_decay_regression_head = params['weight_decay_regression_head']
        self.config.optimizer.weight_decay_pool            = params['weight_decay_pool']
        self.config.optimizer.weight_decay_gcn             = params['weight_decay_gcn']
        
        self.config.training.epochs     = self.config.tuning.epochs
        self.config.training.batch_size = 128
        self.config.training.verbose    = False
        
        trial_log_dir    = Path(self.config.io.tunerdir) / f"optuna_trial_{trial.number}"
        self.config.io.tunerdir = str(trial_log_dir)
        
        model = Model(self.config)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        logger = Logger(log_dir=self.config.io.tunerdir, name=f"trial_{trial.number}", level="WARNING")
                
        trainer = Trainer(
            model          = model,
            norm           = self.norm_metrics,
            config         = self.config,
            dataset_config = self.dataset_config,
            run_dir        = trial_log_dir,
            logger         = logger,
        )

        best_val = float("inf")

        try:
            for epoch in range(self.config.tuning.epochs):
                train_loss  = trainer.train_epoch(train_loader, epoch)
                val_results = trainer.evaluate(val_loader, epoch, stage="validation")
                val_loss    = val_results["avg_loss"]
                epoch_num   = epoch + 1

                if val_loss < best_val:
                    best_val = float(val_loss)

                trainer.lr_scheduler.step()

                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    logger.warning(f"Trial {trial.number} pruned at epoch {epoch_num} with validation loss {val_loss:.6f}.")
                    raise optuna.TrialPruned()

                if trainer.early_stopping(val_loss, trainer.model):
                    trainer.early_stopping.restore_model(trainer.model)
                    break
        finally:
            logger.close()
            self.config.io.writer.close()

        trial.set_user_attr("val_loss_curve", trainer.val_losses)
        return best_val

    def optimize(self, train_dataset, val_dataset, n_trials):
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.config.split.random_state)

        self.study = optuna.create_study(
            direction="minimize", 
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=self.warmup_trials, n_warmup_steps=self.warmup_steps)
        )

        self.study.optimize(
            lambda trial: self._objective(trial, train_dataset, val_dataset),
            n_trials=n_trials,
            show_progress_bar=True
        )

        self.logger.section("[Optimization Complete]")
        self.logger.subsection(f"Best validation loss: {self.study.best_value:.6f}")
        self.logger.subsection("Best hyperparameters:")
        for key, value in self.study.best_trial.params.items():
            self.logger.subsection(f"  {key}: {value}")

        return self.study

    def build_best_configs(self, epochs=300):
        best_params = self.study.best_trial.params
        
        self.config.model.gcn_hidden_dims        = tuple(best_params[f"gcn_hidden_{i}"] for i in range(3))
        self.config.model.regression_hidden_dims = tuple(best_params[f"regression_hidden_{i}"] for i in range(3))
        self.config.model.gat_dropout            = best_params["gat_dropout"]
        self.config.model.regression_dropout     = best_params["regression_dropout"]
        self.config.model.gat_heads              = best_params["gat_heads"]
        
        self.config.optimizer.lr_regression_head           = best_params["lr_regression_head"]
        self.config.optimizer.lr_pool                      = best_params["lr_pool"]
        self.config.optimizer.lr_gcn                       = best_params["lr_gcn"]
        self.config.optimizer.weight_decay_regression_head = best_params["weight_decay_regression_head"]
        self.config.optimizer.weight_decay_pool            = best_params["weight_decay_pool"]
        self.config.optimizer.weight_decay_gcn             = best_params["weight_decay_gcn"]
        
        self.config.training.epochs = epochs
        self.config.training.batch_size = 128
        
        self.logger.section("[Applied Best Hyperparameters to Global Config]")
        self.logger.subsection("Model Configuration:")
        self.logger.subsection(f"  GCN hidden dims: {self.config.model.gcn_hidden_dims}")
        self.logger.subsection(f"  Regression hidden dims: {self.config.model.regression_hidden_dims}")
        self.logger.subsection(f"  GAT heads: {self.config.model.gat_heads}")
        self.logger.subsection(f"  GAT dropout: {self.config.model.gat_dropout}")
        self.logger.subsection(f"  Regression dropout: {self.config.model.regression_dropout}")
        self.logger.subsection("Optimizer Configuration:")
        self.logger.subsection(f"  LR regression head: {self.config.optimizer.lr_regression_head}")
        self.logger.subsection(f"  LR pool: {self.config.optimizer.lr_pool}")
        self.logger.subsection(f"  LR GCN: {self.config.optimizer.lr_gcn}")
        self.logger.subsection(f"  Weight decay regression head: {self.config.optimizer.weight_decay_regression_head}")
        self.logger.subsection(f"  Weight decay pool: {self.config.optimizer.weight_decay_pool}")
        self.logger.subsection(f"  Weight decay GCN: {self.config.optimizer.weight_decay_gcn}")
        
        return self.config
