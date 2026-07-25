from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib    import Path

import numpy as np
import torch


class Checkpoint:
    def __init__(self, logger, tracker, save_path: str, min_delta: float = 0.0):
        self.logger          = logger
        self.tracker         = tracker
        self.save_path       = save_path
        self.min_delta       = float(min_delta)
        self.best_val_loss   = float("inf")
        self.best_epoch      = -1

    def step(self, val_loss: float, epoch: int, trainer) -> bool:
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = float(val_loss)
            self.best_epoch    = int(epoch)
            self.save(trainer, self.save_path, epoch)
            self.logger.subsection(f"Checkpoint : new best  val_loss={self.best_val_loss:.4f}  -> {self.save_path}")
            return True
        else:
            self.logger.subsection(f"Checkpoint : no improvement  (best={self.best_val_loss:.4f} @ epoch {self.best_epoch + 1})")
            return False

    def save(self, trainer, path: str, epoch: int) -> None:
        checkpoint = trainer.capture_state(epoch)
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["best_epoch"]    = self.best_epoch

        self.logger.info(f"Saving checkpoint at epoch {epoch + 1} to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def restore_best(self, model, device) -> None:
        if self.best_epoch < 0:
            self.logger.warning("No improving checkpoint was saved; leaving the model in its final-epoch state.")
            return

        checkpoint = torch.load(self.save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["params"])

        self.logger.subsection(f"Restored best parameters from epoch {self.best_epoch + 1} (val_loss={self.best_val_loss:.4f}).")

    def state_dict(self) -> dict:
        return {
            "best_val_loss" : self.best_val_loss,
            "best_epoch"    : self.best_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_val_loss = float(state["best_val_loss"])
        self.best_epoch    = int(state["best_epoch"])


class WeightEma:
    def __init__(self, model, decay: float, enabled: bool):
        self.decay   = float(decay)
        self.enabled = bool(enabled)
        self.shadow  = {name: parameter.detach().clone() for name, parameter in model.named_parameters()} if self.enabled else {}

    @torch.no_grad()
    def update(self, model) -> None:
        if not self.enabled:
            return

        for name, parameter in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def applied(self, model):
        if not self.enabled:
            yield
            return

        with torch.no_grad():
            backup = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            for name, parameter in model.named_parameters():
                parameter.copy_(self.shadow[name])

        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        shadow = state["shadow"]

        if self.enabled != bool(shadow):
            raise ValueError("use_ema does not match the saved trainer state; resume with the same use_ema setting")

        self.shadow = {name: tensor.to(self.shadow[name].device).clone() for name, tensor in shadow.items()}


class TrainerState:

    FILENAME = "last.pt"

    @staticmethod
    def path(run_dir) -> Path:
        return Path(run_dir) / TrainerState.FILENAME

    @staticmethod
    def _rng_state(loader_generator) -> dict:
        return {
            "torch"  : torch.get_rng_state(),
            "cuda"   : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy"  : np.random.get_state(),
            "random" : random.getstate(),
            "loader" : loader_generator.get_state() if loader_generator is not None else None,
        }

    @staticmethod
    def capture(trainer, epoch: int, loader_generator=None) -> dict:
        return {
            "epoch"            : int(epoch),
            "global_step"      : int(trainer.global_step),
            "model"            : trainer.model.state_dict(),
            "optimizer"        : trainer.optimizer.state_dict(),
            "ema"              : trainer.ema.state_dict(),
            "warmup"           : trainer.warmup.state_dict(),
            "scheduler"        : trainer.scheduler.state_dict(),
            "early_stopping"   : trainer.early_stopping.state_dict(),
            "checkpoint"       : trainer.checkpoint.state_dict(),
            "gradient_clipper" : trainer.gradient_clipper.state_dict(),
            "train_losses"     : list(trainer.train_losses),
            "val_losses"       : list(trainer.val_losses),
            "rng"              : TrainerState._rng_state(loader_generator),
        }

    @staticmethod
    def save(trainer, epoch: int, path, loader_generator=None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(TrainerState.capture(trainer, epoch, loader_generator), str(path))

    @staticmethod
    def _restore_rng(rng: dict, loader_generator) -> None:
        torch.set_rng_state(rng["torch"])

        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

        np.random.set_state(rng["numpy"])
        random.setstate(rng["random"])

        if rng["loader"] is not None and loader_generator is not None:
            loader_generator.set_state(rng["loader"])

    @staticmethod
    def restore(trainer, path, loader_generator=None) -> int:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"resume=True but no trainer state exists at {path}; start a fresh run or point run_name at a run that saved state")

        state = torch.load(str(path), map_location="cpu", weights_only=False)

        trainer.model.load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.ema.load_state_dict(state["ema"])
        trainer.warmup.load_state_dict(state["warmup"])
        trainer.scheduler.load_state_dict(state["scheduler"])
        trainer.early_stopping.load_state_dict(state["early_stopping"])
        trainer.checkpoint.load_state_dict(state["checkpoint"])
        trainer.gradient_clipper.load_state_dict(state["gradient_clipper"])

        trainer.train_losses = list(state["train_losses"])
        trainer.val_losses   = list(state["val_losses"])
        trainer.global_step  = int(state["global_step"])

        TrainerState._restore_rng(state["rng"], loader_generator)

        return int(state["epoch"]) + 1
