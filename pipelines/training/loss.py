from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class Loss:
    def __init__(self, loss_config, logger, tracker):
        self.type        = loss_config.type
        self.huber_delta = loss_config.huber_delta
        self.logger      = logger
        self.tracker     = tracker

        self.logger.section("[Loss Function]")
        self.logger.subsection(f"Objective: {self.type}")

    def _compute(self, predictions, targets):
        if self.type == "mse":
            return F.mse_loss(predictions, targets)
        if self.type == "huber":
            return F.huber_loss(predictions, targets, delta=self.huber_delta)
        raise ValueError(f"Unknown loss type '{self.type}'")

    def __call__(self, predictions, targets, step):
        total_loss = self._compute(predictions, targets)

        self.tracker.log_metrics("loss", {
            "total"     : total_loss.item(),
            "log_total" : float(np.log1p(total_loss.item())),
        }, step)

        return {"total_loss": total_loss}
