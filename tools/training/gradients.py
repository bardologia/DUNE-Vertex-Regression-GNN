from __future__ import annotations

import math

import numpy as np
import torch


class GradientClipper:
    def __init__(self, config, logger, tracker, param_groups=None):
        self.logger       = logger
        self.tracker      = tracker
        self.param_groups = param_groups

        self.mode                = config.gradient_clipper.clip_mode
        self.threshold           = config.gradient_clipper.max_grad_norm if self.mode == "fixed" else None
        self.window              = config.gradient_clipper.adaptive_window
        self.percentile          = config.gradient_clipper.adaptive_percentile
        self.adaptive_mean_std_k = config.gradient_clipper.adaptive_mean_std_k
        self.epsilon             = config.gradient_clipper.clip_epsilon
        self.log_histogram_freq  = config.gradient_clipper.log_histogram_freq

        self.history     : list[float] = []

        fields = {"Mode": self.mode}

        if self.mode == "fixed":
            fields["Threshold"] = self.threshold

        elif self.mode == "adaptive_percentile":
            fields["Window"]     = self.window
            fields["Percentile"] = self.percentile

        elif self.mode == "adaptive_mean_std":
            fields["Window"]       = self.window
            fields["Mean+k*Std k"] = self.adaptive_mean_std_k

        fields["Histogram every"] = self.log_histogram_freq

        self.logger.section("[Gradient Clipper]")
        self.logger.kv_table(fields)

    @staticmethod
    def global_norm(model: torch.nn.Module) -> float:
        gradients = [parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None]

        if not gradients:
            return 0.0

        per_parameter_norms = torch._foreach_norm(gradients, 2)
        total_norm          = torch.norm(torch.stack(per_parameter_norms), 2)

        return total_norm.item()

    def _log_group_norms(self, global_step: int) -> None:
        if self.param_groups is None or len(self.param_groups) <= 1:
            return

        for index, group in enumerate(self.param_groups):
            gradients = [parameter.grad.detach() for parameter in group["params"] if parameter.grad is not None]

            if not gradients:
                continue

            group_norm = torch.norm(torch.stack(torch._foreach_norm(gradients, 2)), 2).item()
            name       = group.get("name", str(index))
            self.tracker.log_scalar(f"train/grad_norm/{name}", group_norm, global_step)

    def _clip(self, model: torch.nn.Module, norm: float, max_norm: float) -> tuple[float, float]:
        scale = min(1.0, max_norm / (norm + self.epsilon))
        if scale >= 1.0:
            return norm, norm

        gradients = [parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None]
        torch._foreach_mul_(gradients, scale)

        return norm, norm * scale

    def _compute_adaptive_threshold(self) -> float | None:
        if len(self.history) < self.window:
            return None

        window_data = np.asarray(self.history[-self.window:], dtype=np.float32)

        if self.mode == "adaptive_percentile":
            return float(np.percentile(window_data, self.percentile))

        else:
            return float(window_data.mean() + self.adaptive_mean_std_k * window_data.std())

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
        self._log_group_norms(global_step)

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
        if not math.isfinite(grad_norm_value):
            self.logger.warning(f"Non-finite gradient norm at step {global_step}; excluded from the adaptive clipping history.")
            return

        self.history.append(float(grad_norm_value))

        if global_step % self.log_histogram_freq == 0 and len(self.history) >= self.log_histogram_freq:
            self.tracker.log_histogram("train/grad_norm_dist", np.asarray(self.history[-self.log_histogram_freq:], dtype=np.float32), global_step)

    def state_dict(self) -> dict:
        keep = max(self.window, self.log_histogram_freq)
        return {"history": [float(value) for value in self.history[-keep:]]}

    def load_state_dict(self, state: dict) -> None:
        self.history = [float(value) for value in state["history"]]
