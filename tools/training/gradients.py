from __future__ import annotations

import numpy as np
import torch


class GradientClipper:
    def __init__(self, config, logger, tracker):
        self.logger  = logger
        self.tracker = tracker

        self.mode       = config.gradient_clipper.clip_mode
        self.threshold  = config.gradient_clipper.max_grad_norm if self.mode == "fixed" else None
        self.window     = config.gradient_clipper.adaptive_window
        self.percentile = config.gradient_clipper.adaptive_percentile
        self.mean_std_k = config.gradient_clipper.adaptive_mean_std_k
        self.epsilon    = config.gradient_clipper.clip_epsilon
        self.hist_freq  = config.gradient_clipper.log_histogram_freq

        self.history     : list[float] = []

        self.logger.section("[Gradient Clipper]")
        self.logger.subsection(f"Mode : {self.mode}")

        if self.mode == "fixed":
            self.logger.subsection(f"Threshold     : {self.threshold}")

        elif self.mode == "adaptive_percentile":
            self.logger.subsection(f"Window        : {self.window}")
            self.logger.subsection(f"Percentile    : {self.percentile}")

        elif self.mode == "adaptive_mean_std":
            self.logger.subsection(f"Window        : {self.window}")
            self.logger.subsection(f"Mean+k*Std  k : {self.mean_std_k}")

    @staticmethod
    def global_norm(model: torch.nn.Module) -> float:
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]

        if not grads:
            return 0.0

        per_param_norms = torch._foreach_norm(grads, 2)
        total_norm      = torch.norm(torch.stack(per_param_norms), 2)

        return total_norm.item()

    def _clip(self, model: torch.nn.Module, norm: float, max_norm: float) -> tuple[float, float]:
        scale = min(1.0, max_norm / (norm + self.epsilon))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(scale)

        norm_after = norm * scale
        return norm, norm_after

    def _compute_adaptive_threshold(self) -> float | None:
        if len(self.history) < self.window:
            return None

        window_data = np.asarray(self.history[-self.window:], dtype=np.float32)

        if self.mode == "adaptive_percentile":
            return float(np.percentile(window_data, self.percentile))

        else:
            return float(window_data.mean() + self.mean_std_k * window_data.std())

    def check_gradients(self, model: torch.nn.Module, global_step: int) -> bool:
        has_invalid = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self.logger.warning(f"NaN/Inf gradient detected in {name} at step {global_step}!")
                    has_invalid = True
                    break

        return has_invalid

    def maybe_clip(self, model: torch.nn.Module, global_step: int):
        if self.tracker.debug:
            self.check_gradients(model, global_step)

        norm = GradientClipper.global_norm(model)

        if norm > 100.0:
            self.logger.warning(f"Exploding gradient norm detected: {norm:.2f} at step {global_step}!")

        self.tracker.log_scalar("train/grad_norm_before_clip", norm, global_step)

        if self.mode == "disabled":
            return norm

        if self.mode == "fixed":
            threshold = self.threshold
        else:
            threshold = self._compute_adaptive_threshold()

        if threshold is None:
            return norm

        norm_before, norm_after = self._clip(model, norm, threshold)
        clip_ratio = norm_after / (norm_before + self.epsilon)

        self.tracker.log_scalar("train/grad_norm_after_clip",  norm_after,  global_step)
        self.tracker.log_scalar("train/grad_clip_ratio",       clip_ratio,  global_step)
        self.tracker.log_scalar("train/grad_clip_threshold",   threshold,   global_step)

        return norm_before

    def record(self, grad_norm_value: float, global_step: int):
        self.history.append(float(grad_norm_value))

        if global_step % self.hist_freq == 0 and len(self.history) >= self.hist_freq:
            self.tracker.log_histogram("train/grad_norm_dist", np.asarray(self.history[-self.hist_freq:], dtype=np.float32), global_step)
