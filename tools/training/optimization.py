from __future__ import annotations


class LearningRateScaler:
    def __init__(self, optimizer_config, loop_config):
        self.optimizer_config = optimizer_config
        self.loop_config      = loop_config

    @property
    def reference_batch_size(self) -> int:
        reference = int(self.optimizer_config.reference_batch_size)
        if reference < 1:
            raise ValueError(f"optimizer.reference_batch_size must be at least 1; got {reference}")
        return reference

    @property
    def effective_batch_size(self) -> int:
        return int(self.loop_config.batch_size) * int(self.loop_config.gradient_accumulation_steps)

    @property
    def factor(self) -> float:
        return self.effective_batch_size / self.reference_batch_size

    @property
    def regression_head(self) -> float:
        return float(self.optimizer_config.learning_rate_regression_head) * self.factor

    @property
    def pool(self) -> float:
        return float(self.optimizer_config.learning_rate_pool) * self.factor

    @property
    def encoder(self) -> float:
        return float(self.optimizer_config.learning_rate_encoder) * self.factor

    def pin_to_effective_batch(self) -> int:
        self.optimizer_config.reference_batch_size = self.effective_batch_size
        return self.effective_batch_size

    def summary(self) -> dict:
        return {
            "Reference batch" : self.reference_batch_size,
            "Effective batch" : f"{self.effective_batch_size} ({self.loop_config.batch_size} x {self.loop_config.gradient_accumulation_steps} accumulated)",
            "LR scale"        : f"{self.factor:.4f}",
        }
