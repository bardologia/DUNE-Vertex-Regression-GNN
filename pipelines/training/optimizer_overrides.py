from __future__ import annotations

from dataclasses import fields


class OptimizerOverrideApplier:
    def __init__(self, training_config, model_config, explicit_paths, logger):
        self.optimizer_config = training_config.optimizer
        self.overrides        = dict(getattr(model_config, "optimizer_overrides", {}) or {})
        self.explicit_paths   = set(explicit_paths or ())
        self.logger           = logger

        self.valid_fields = {definition.name for definition in fields(self.optimizer_config)}

    def _validate(self) -> None:
        unknown = [key for key in self.overrides if key not in self.valid_fields]
        if unknown:
            raise ValueError(f"Model optimizer_overrides reference unknown OptimizerConfig field(s) {unknown}; valid fields: {sorted(self.valid_fields)}.")

    def _apply(self) -> tuple[dict, dict]:
        applied = {}
        skipped = {}

        for key, value in self.overrides.items():
            if f"training.optimizer.{key}" in self.explicit_paths:
                skipped[key] = getattr(self.optimizer_config, key)
                continue

            setattr(self.optimizer_config, key, value)
            applied[key] = value

        return applied, skipped

    def _report(self, applied, skipped) -> None:
        self.logger.section("[Per-Model Optimizer Overrides]")

        if applied:
            self.logger.kv_table(applied)

        for key, kept_value in skipped.items():
            self.logger.subsection(f"{key}: kept command-line value {kept_value}, model override not applied")

    def run(self) -> None:
        if not self.overrides:
            return

        self._validate()
        applied, skipped = self._apply()
        self._report(applied, skipped)
