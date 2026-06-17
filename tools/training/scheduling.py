from __future__ import annotations

import math


class Warmup:
    def __init__(self, config, logger, tracker):
        self.config  = config
        self.logger  = logger
        self.tracker = tracker

        self.warmup_steps        = self.config.warmup.warmup_steps
        self.warmup_start_factor = self.config.warmup.warmup_start_factor
        self.enabled             = self.config.warmup.warmup_enabled
        self.mode                = self.config.warmup.warmup_mode
        self.poly_power          = self.config.warmup.warmup_poly_power

        self.current_step       = 0
        self.warmup_finished    = False
        self._logged_completion = False

        self.logger.section("[Warmup Scheduler]")
        self.logger.kv_table({
            "Enabled"      : self.enabled,
            "Steps"        : self.warmup_steps,
            "Mode"         : self.mode,
            "Start Factor" : self.warmup_start_factor,
        })

    def factor(self) -> float:
        if not self.enabled or self.warmup_steps <= 0:
            return 1.0

        if self.current_step >= self.warmup_steps:
            return 1.0

        progress = self.current_step / self.warmup_steps
        s        = self.warmup_start_factor

        if self.mode == "cosine":
            cos_factor = (1.0 - math.cos(math.pi * progress)) / 2.0
            return s + (1.0 - s) * cos_factor

        elif self.mode == "exponential":
            if s <= 0:
                return progress
            return s ** (1.0 - progress)

        elif self.mode == "polynomial":
            return s + (1.0 - s) * (progress ** self.poly_power)

        else:
            return s + (1.0 - s) * progress

    def step(self) -> float:
        if not self.enabled or self.warmup_steps <= 0:
            self.warmup_finished = True
            return 1.0

        if self.warmup_finished:
            return 1.0

        self.current_step += 1
        factor             = self.factor()
        self.tracker.log_scalar("lr/warmup_factor", factor, self.current_step)

        if self.current_step >= self.warmup_steps and not self.warmup_finished:
            self.warmup_finished = True
            if not self._logged_completion:
                self.logger.info(f"Warmup completed at step {self.current_step}.")
                self._logged_completion = True

        return factor

    def reset(self) -> None:
        self.current_step       = 0
        self.warmup_finished    = False
        self._logged_completion = False

    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0


class Scheduler:
    def __init__(self, base_lrs, warmup, config, logger, tracker):
        self.config  = config
        self.warmup  = warmup
        self.logger  = logger
        self.tracker = tracker

        self.base_lrs       = list(base_lrs)
        self.current_lrs    = list(base_lrs)
        self.scheduler_type = self.config.scheduler.type

        self._epoch_offset  = 0
        self._t_max_override = None

        self._log_scheduler_info()

    def set_total_epochs(self, epochs: int) -> None:
        resolved = max(1, int(epochs))
        if resolved == self._resolved_t_max():
            return

        self._t_max_override = resolved
        self._log_scheduler_info()

    def _resolved_t_max(self) -> int:
        return self._t_max_override if self._t_max_override is not None else self.config.scheduler.epochs

    def _cosine_annealing(self, epoch: int) -> float:
        T_max         = self._resolved_t_max()
        eta_min       = float(self.config.scheduler.eta_min)
        base_lr       = self.base_lrs[0]
        eta_min_ratio = eta_min / max(base_lr, 1e-12)
        progress      = min(1.0, epoch / max(1, T_max))
        return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))

    def _constant(self) -> float:
        return 1.0

    def _factor_for(self, epoch: int) -> float:
        if self.scheduler_type == "cosine_annealing":
            return self._cosine_annealing(epoch)

        if self.scheduler_type == "constant":
            return self._constant()

        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def reset(self, epoch_offset: int = 0) -> None:
        self._epoch_offset = epoch_offset
        self.current_lrs   = list(self.base_lrs)

    def step(self, epoch: int) -> list[float]:
        factor           = self._factor_for(epoch - self._epoch_offset)
        self.current_lrs = [lr * factor for lr in self.base_lrs]

        return list(self.current_lrs)

    def effective_lrs(self) -> list[float]:
        if self.warmup is not None and not self.warmup.is_finished():
            f = self.warmup.factor()
            return [lr * f for lr in self.current_lrs]

        return list(self.current_lrs)

    def _log_scheduler_info(self):
        self.logger.section("[Learning Rate Scheduler]")
        info = {
            "Scheduler Type" : self.scheduler_type,
            "Base LRs"       : self.base_lrs,
        }

        if self.scheduler_type == "cosine_annealing":
            info["T_max"]   = self._resolved_t_max()
            info["Eta Min"] = self.config.scheduler.eta_min

        info["Warmup Enabled"] = self.warmup.enabled if self.warmup else False
        self.logger.kv_table(info)
