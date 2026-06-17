from __future__ import annotations

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker

        self.enabled = self.config.ema.use_ema
        self.decay   = self.config.ema.ema_decay

        self.logger.section("[Exponential Moving Average]")
        self.logger.kv_table({
            "Enabled" : self.enabled,
            "Decay"   : self.decay,
        })

        self.shadow : dict[str, torch.Tensor] = {}
        self.backup : dict[str, torch.Tensor] = {}

        if self.enabled:
            self.shadow = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int | None = None) -> None:
        if not self.enabled:
            return

        total_divergence = 0.0
        shadow_norm      = 0.0
        model_norm       = 0.0

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            if name not in self.shadow:
                self.shadow[name] = parameter.detach().clone()
                self.logger.warning(f"EMA initialized shadow for new parameter '{name}'")
                continue

            divergence        = (self.shadow[name] - parameter.detach()).norm().item()
            total_divergence += divergence

            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

            shadow_norm += self.shadow[name].norm().item()
            model_norm  += parameter.norm().item()

        if step is not None:
            self.tracker.log_scalar("ema/param_divergence", total_divergence, step)
            self.tracker.log_scalar("ema/shadow_norm",      shadow_norm,      step)
            self.tracker.log_scalar("ema/model_norm",       model_norm,       step)
            self.tracker.log_scalar("ema/norm_ratio",       shadow_norm / (model_norm + 1e-8), step)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        if not self.enabled:
            return

        self.backup = {}

        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = parameter.detach().clone()
            parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self.enabled:
            return

        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.copy_(self.backup[name])

        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "enabled" : self.enabled,
            "decay"   : self.decay,
            "shadow"  : {name: tensor.detach().cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.enabled = state["enabled"]
        self.decay   = state["decay"]
        self.shadow  = state["shadow"]
        self.backup  = {}
