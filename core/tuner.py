
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
        self.base_tuner_directory = Path(self.config.io.tunerdir)

    def _suggest_hyperparameters(self, trial):
        # GPS backbone
        gps_hidden_dim = trial.suggest_categorical("gps_hidden_dim", [64, 128, 192, 256])
        gps_num_layers = trial.suggest_int("gps_num_layers", 2, 6)
        gps_heads      = trial.suggest_categorical("gps_heads", [2, 4, 8])
        gps_dropout    = trial.suggest_float("gps_dropout", 0.0, 0.2, step=0.05)
        gps_attn_dropout = trial.suggest_float("gps_attn_dropout", 0.0, 0.2, step=0.05)
        drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.2, step=0.05)
        
        # Multi-scale pooling
        pool_num_levels = trial.suggest_int("pool_num_levels", 2, 4)
        sag_ratio       = trial.suggest_float("sag_ratio", 0.3, 0.7, step=0.1)

        # Hierarchical regression head
        hierarchical_feature_dim = trial.suggest_categorical("hierarchical_feature_dim", [128, 256, 384])
        coord_embed_dim = trial.suggest_categorical("coord_embed_dim", [32, 64, 128])
        
        regression_layers      = trial.suggest_int("regression_layers", 2, 4)
        regression_hidden_dims = tuple(trial.suggest_categorical(f"regression_hidden_{i}", [64, 128, 256]) for i in range(regression_layers))
        regression_dropout = trial.suggest_float("regression_dropout", 0.0, 0.2, step=0.05)
        
        # Optimizer
        lr_regression_head = trial.suggest_float("lr_regression_head", 5e-5, 2e-3, log=True)
        lr_pool            = trial.suggest_float("lr_pool", 1e-5, 1e-3, log=True)
        lr_gcn             = trial.suggest_float("lr_gcn", 1e-5, 1e-3, log=True)
        
        weight_decay_regression_head = trial.suggest_float("weight_decay_regression_head", 1e-6, 1e-4, log=True)
        weight_decay_pool            = trial.suggest_float("weight_decay_pool", 1e-6, 1e-4, log=True)
        weight_decay_gcn             = trial.suggest_float("weight_decay_gcn", 1e-6, 5e-4, log=True)
        
        return {
            'gps_hidden_dim'               : gps_hidden_dim,
            'gps_num_layers'               : gps_num_layers,
            'gps_heads'                    : gps_heads,
            'gps_dropout'                  : gps_dropout,
            'gps_attn_dropout'             : gps_attn_dropout,
            'drop_path_rate'               : drop_path_rate,
            'pool_num_levels'              : pool_num_levels,
            'sag_ratio'                    : sag_ratio,
            'hierarchical_feature_dim'     : hierarchical_feature_dim,
            'coord_embed_dim'              : coord_embed_dim,
            'regression_hidden_dims'       : regression_hidden_dims,
            'regression_dropout'           : regression_dropout,
            'lr_regression_head'           : lr_regression_head,
            'lr_pool'                      : lr_pool,
            'lr_gcn'                       : lr_gcn,
            'weight_decay_regression_head' : weight_decay_regression_head,
            'weight_decay_pool'            : weight_decay_pool,
            'weight_decay_gcn'             : weight_decay_gcn,
        }

    def _objective(self, trial, train_dataset, val_dataset):
        params = self._suggest_hyperparameters(trial)
        
        self.config.model.gps_hidden_dim            = params['gps_hidden_dim']
        self.config.model.gps_num_layers            = params['gps_num_layers']
        self.config.model.gps_heads                 = params['gps_heads']
        self.config.model.gps_dropout               = params['gps_dropout']
        self.config.model.gps_attn_dropout          = params['gps_attn_dropout']
        self.config.model.drop_path_rate            = params['drop_path_rate']
        self.config.model.pool_num_levels           = params['pool_num_levels']
        self.config.model.sag_ratio                 = params['sag_ratio']
        self.config.model.hierarchical_feature_dim  = params['hierarchical_feature_dim']
        self.config.model.coord_embed_dim           = params['coord_embed_dim']
        self.config.model.regression_hidden_dims    = params['regression_hidden_dims']
        self.config.model.regression_dropout        = params['regression_dropout']
        
        self.config.optimizer.lr_regression_head           = params['lr_regression_head']
        self.config.optimizer.lr_pool                      = params['lr_pool']
        self.config.optimizer.lr_gcn                       = params['lr_gcn']
        self.config.optimizer.weight_decay_regression_head = params['weight_decay_regression_head']
        self.config.optimizer.weight_decay_pool            = params['weight_decay_pool']
        self.config.optimizer.weight_decay_gcn             = params['weight_decay_gcn']
        
        self.config.training.epochs     = self.config.tuning.epochs
        self.config.training.batch_size = 128
        self.config.training.verbose    = False
        self.config.tuning.tuning_mode  = True

        trial_log_dir = self.base_tuner_directory / f"optuna_trial_{trial.number}"
        self.config.io.tunerdir = str(trial_log_dir)

        model = Model(self.config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
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

        self.config.model.gps_hidden_dim           = best_params["gps_hidden_dim"]
        self.config.model.gps_num_layers           = best_params["gps_num_layers"]
        self.config.model.gps_heads                = best_params["gps_heads"]
        self.config.model.gps_dropout              = best_params["gps_dropout"]
        self.config.model.gps_attn_dropout         = best_params["gps_attn_dropout"]
        self.config.model.drop_path_rate           = best_params["drop_path_rate"]
        self.config.model.pool_num_levels          = best_params["pool_num_levels"]
        self.config.model.sag_ratio                = best_params["sag_ratio"]
        self.config.model.hierarchical_feature_dim = best_params["hierarchical_feature_dim"]
        self.config.model.coord_embed_dim          = best_params["coord_embed_dim"]
        self.config.model.regression_hidden_dims   = tuple(
            best_params[f"regression_hidden_{i}"] for i in range(best_params["regression_layers"])
        )
        self.config.model.regression_dropout       = best_params["regression_dropout"]

        self.config.optimizer.lr_regression_head           = best_params["lr_regression_head"]
        self.config.optimizer.lr_pool                      = best_params["lr_pool"]
        self.config.optimizer.lr_gcn                       = best_params["lr_gcn"]
        self.config.optimizer.weight_decay_regression_head = best_params["weight_decay_regression_head"]
        self.config.optimizer.weight_decay_pool            = best_params["weight_decay_pool"]
        self.config.optimizer.weight_decay_gcn             = best_params["weight_decay_gcn"]

        self.config.training.epochs = epochs

        self.logger.section("[Applied Best Hyperparameters to Global Config]")
        self.logger.subsection("Model Configuration:")
        self.logger.subsection(f"  GPS hidden dim: {self.config.model.gps_hidden_dim}")
        self.logger.subsection(f"  GPS layers: {self.config.model.gps_num_layers}")
        self.logger.subsection(f"  GPS heads: {self.config.model.gps_heads}")
        self.logger.subsection(f"  GPS dropout: {self.config.model.gps_dropout}")
        self.logger.subsection(f"  Regression hidden dims: {self.config.model.regression_hidden_dims}")
        self.logger.subsection(f"  Regression dropout: {self.config.model.regression_dropout}")
        self.logger.subsection("Optimizer Configuration:")
        self.logger.subsection(f"  LR regression head: {self.config.optimizer.lr_regression_head}")
        self.logger.subsection(f"  LR pool: {self.config.optimizer.lr_pool}")
        self.logger.subsection(f"  LR GCN: {self.config.optimizer.lr_gcn}")
        self.logger.subsection(f"  Weight decay regression head: {self.config.optimizer.weight_decay_regression_head}")
        self.logger.subsection(f"  Weight decay pool: {self.config.optimizer.weight_decay_pool}")
        self.logger.subsection(f"  Weight decay GCN: {self.config.optimizer.weight_decay_gcn}")

        return self.config
