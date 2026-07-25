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

        progress     = self.current_step / self.warmup_steps
        start_factor = self.warmup_start_factor

        if self.mode == "cosine":
            cosine_factor = (1.0 - math.cos(math.pi * progress)) / 2.0
            return start_factor + (1.0 - start_factor) * cosine_factor

        elif self.mode == "exponential":
            if start_factor <= 0:
                return progress
            return start_factor ** (1.0 - progress)

        elif self.mode == "polynomial":
            return start_factor + (1.0 - start_factor) * (progress ** self.poly_power)

        else:
            return start_factor + (1.0 - start_factor) * progress

    def step(self) -> float:
        if not self.enabled or self.warmup_steps <= 0:
            self.warmup_finished = True
            return 1.0

        if self.warmup_finished:
            return 1.0

        self.current_step += 1
        factor             = self.factor()

        if self.current_step >= self.warmup_steps and not self.warmup_finished:
            self.warmup_finished = True
            if not self._logged_completion:
                self.logger.info(f"Warmup completed at step {self.current_step}.")
                self._logged_completion = True

        return factor

    def is_finished(self) -> bool:
        return self.warmup_finished or not self.enabled or self.warmup_steps <= 0

    def reset(self) -> None:
        self.current_step       = 0
        self.warmup_finished    = False
        self._logged_completion = False

    def state_dict(self) -> dict:
        return {
            "current_step"      : self.current_step,
            "warmup_finished"   : self.warmup_finished,
            "logged_completion" : self._logged_completion,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_step       = int(state["current_step"])
        self.warmup_finished    = bool(state["warmup_finished"])
        self._logged_completion = bool(state["logged_completion"])


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

        self.plateau_mode      = self.config.scheduler.mode
        self.plateau_factor    = float(self.config.scheduler.factor)
        self.plateau_patience  = int(self.config.scheduler.patience)
        self.plateau_threshold = float(self.config.scheduler.threshold)
        self.plateau_cooldown  = int(self.config.scheduler.cooldown)
        self.plateau_min_lr    = float(self.config.scheduler.min_lr)

        self.best_metric      = math.inf if self.plateau_mode == "min" else -math.inf
        self.num_bad_epochs   = 0
        self.cooldown_counter = 0

        self._log_scheduler_info()

    def _progress(self, epoch: int) -> float:
        maximum_epochs = max(1, int(self.config.loop.epochs))
        return min(1.0, epoch / max(1, maximum_epochs - 1))

    def _decayed(self, factor: float) -> list[float]:
        eta_min = float(self.config.scheduler.eta_min)
        return [eta_min + (base_learning_rate - eta_min) * factor for base_learning_rate in self.base_lrs]

    def _cosine_annealing(self, epoch: int) -> list[float]:
        return self._decayed(0.5 * (1.0 + math.cos(math.pi * self._progress(epoch))))

    def _linear(self, epoch: int) -> list[float]:
        return self._decayed(1.0 - self._progress(epoch))

    def _polynomial(self, epoch: int) -> list[float]:
        return self._decayed((1.0 - self._progress(epoch)) ** float(self.config.scheduler.power))

    def _exponential(self, epoch: int) -> list[float]:
        eta_min = float(self.config.scheduler.eta_min)
        ratio   = max(eta_min / max(self.base_lrs[0], 1e-12), 1e-8)
        return [base_learning_rate * ratio ** self._progress(epoch) for base_learning_rate in self.base_lrs]

    def _step_decay(self, epoch: int) -> list[float]:
        step_size = max(1, int(self.config.scheduler.step_size))
        decayed   = float(self.config.scheduler.gamma) ** (epoch // step_size)
        return [max(base_learning_rate * decayed, float(self.config.scheduler.eta_min)) for base_learning_rate in self.base_lrs]

    def _constant(self) -> list[float]:
        return list(self.base_lrs)

    def _lrs_for(self, epoch: int) -> list[float]:
        if self.scheduler_type == "cosine_annealing":
            return self._cosine_annealing(epoch)

        if self.scheduler_type == "linear":
            return self._linear(epoch)

        if self.scheduler_type == "polynomial":
            return self._polynomial(epoch)

        if self.scheduler_type == "exponential":
            return self._exponential(epoch)

        if self.scheduler_type == "step":
            return self._step_decay(epoch)

        if self.scheduler_type == "constant":
            return self._constant()

        if self.scheduler_type == "reduce_on_plateau":
            return list(self.current_lrs)

        raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

    def step(self, epoch: int) -> list[float]:
        self.current_lrs = self._lrs_for(epoch - self._epoch_offset)
        return list(self.current_lrs)

    def step_metric(self, metric: float) -> list[float]:
        if self.scheduler_type != "reduce_on_plateau":
            return list(self.current_lrs)

        if self._is_improvement(metric):
            self.best_metric    = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs    = 0

        if self.num_bad_epochs > self.plateau_patience:
            self._reduce_lrs()
            self.cooldown_counter = self.plateau_cooldown
            self.num_bad_epochs   = 0

        return list(self.current_lrs)

    def _is_improvement(self, metric: float) -> bool:
        if self.plateau_mode == "min":
            return metric < self.best_metric * (1.0 - self.plateau_threshold)
        return metric > self.best_metric * (1.0 + self.plateau_threshold)

    def _reduce_lrs(self) -> None:
        reduced_lrs = [max(learning_rate * self.plateau_factor, self.plateau_min_lr) for learning_rate in self.current_lrs]

        if reduced_lrs == self.current_lrs:
            return

        self.current_lrs = reduced_lrs
        self.logger.info(f"Plateau detected after {self.plateau_patience} stagnant epochs; reducing learning rates to {[round(learning_rate, 8) for learning_rate in reduced_lrs]}.")

    def effective_lrs(self) -> list[float]:
        if self.warmup is not None and not self.warmup.is_finished():
            warmup_factor = self.warmup.factor()
            return [learning_rate * warmup_factor for learning_rate in self.current_lrs]

        return list(self.current_lrs)

    def reset(self, epoch_offset: int = 0) -> None:
        self._epoch_offset    = int(epoch_offset)
        self.current_lrs      = list(self.base_lrs)
        self.best_metric      = math.inf if self.plateau_mode == "min" else -math.inf
        self.num_bad_epochs   = 0
        self.cooldown_counter = 0

    def state_dict(self) -> dict:
        return {
            "current_lrs"      : list(self.current_lrs),
            "epoch_offset"     : self._epoch_offset,
            "best_metric"      : self.best_metric,
            "num_bad_epochs"   : self.num_bad_epochs,
            "cooldown_counter" : self.cooldown_counter,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current_lrs      = list(state["current_lrs"])
        self._epoch_offset    = int(state["epoch_offset"])
        self.best_metric      = float(state["best_metric"])
        self.num_bad_epochs   = int(state["num_bad_epochs"])
        self.cooldown_counter = int(state["cooldown_counter"])

    def _log_scheduler_info(self):
        self.logger.section("[Learning Rate Scheduler]")
        info = {
            "Scheduler Type" : self.scheduler_type,
            "Base LRs"       : self.base_lrs,
        }

        if self.scheduler_type in ("cosine_annealing", "linear", "polynomial", "exponential", "step"):
            info["T_max"]   = max(1, int(self.config.loop.epochs))
            info["Eta Min"] = self.config.scheduler.eta_min

        if self.scheduler_type == "polynomial":
            info["Power"] = self.config.scheduler.power

        if self.scheduler_type == "step":
            info["Step Size"] = self.config.scheduler.step_size
            info["Gamma"]     = self.config.scheduler.gamma

        if self.scheduler_type == "reduce_on_plateau":
            info["Mode"]      = self.plateau_mode
            info["Factor"]    = self.plateau_factor
            info["Patience"]  = self.plateau_patience
            info["Threshold"] = self.plateau_threshold
            info["Cooldown"]  = self.plateau_cooldown
            info["Min LR"]    = self.plateau_min_lr

        info["Warmup Enabled"] = self.warmup.enabled if self.warmup else False
        self.logger.kv_table(info)
