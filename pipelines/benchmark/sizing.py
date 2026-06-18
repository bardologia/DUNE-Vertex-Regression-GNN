from __future__ import annotations

from dataclasses import dataclass, field

from models import get_model


@dataclass
class SizeMatchResult:
    model         : str
    width         : int
    overrides     : dict
    parameters    : int
    target        : int
    deviation_pct : float
    iterations    : int
    flags         : list = field(default_factory=list)


class ModelSizer:
    def __init__(self, config, logger):
        self.config    = config.size_match
        self.logger    = logger
        self.attribute = self.config.scaled_attribute

    @staticmethod
    def parameter_count(model):
        return int(sum(parameter.numel() for parameter in model.parameters()))

    def _count_at(self, model_name, width):
        model, _ = get_model(model_name, **{self.attribute: int(width)})
        count    = self.parameter_count(model)
        del model
        return count

    def reference_count(self):
        model, _ = get_model(self.config.reference_model)
        return self.parameter_count(model)

    def _bracket_upper(self, model_name, target):
        width = self.config.minimum_width
        while width < self.config.maximum_width and self._count_at(model_name, width) < target:
            width = min(self.config.maximum_width, width * 2)
        return width

    def match(self, model_name, target):
        low_width  = self.config.minimum_width
        high_width = self._bracket_upper(model_name, target)

        best_width      = low_width
        best_count      = self._count_at(model_name, low_width)
        best_deviation  = abs(best_count - target)
        iterations      = 1

        while low_width < high_width and iterations < self.config.max_iterations:
            mid_width = (low_width + high_width) // 2
            if mid_width in (low_width, high_width):
                break

            count      = self._count_at(model_name, mid_width)
            iterations += 1

            deviation = abs(count - target)
            if deviation < best_deviation:
                best_width, best_count, best_deviation = mid_width, count, deviation

            if count < target:
                low_width = mid_width
            else:
                high_width = mid_width

        flags         = []
        deviation_pct = 100.0 * (best_count - target) / target if target > 0 else 0.0

        if best_width <= self.config.minimum_width and best_count > target:
            flags.append("clamped at minimum width")
        if best_width >= self.config.maximum_width and best_count < target:
            flags.append("clamped at maximum width")
        if abs(deviation_pct) > 100.0 * self.config.tolerance:
            flags.append("outside tolerance")

        return SizeMatchResult(
            model         = model_name,
            width         = int(best_width),
            overrides     = {self.attribute: int(best_width)},
            parameters    = int(best_count),
            target        = int(target),
            deviation_pct = float(deviation_pct),
            iterations    = int(iterations),
            flags         = flags,
        )
