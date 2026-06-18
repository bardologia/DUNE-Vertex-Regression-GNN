from __future__ import annotations

import numpy as np


class Augmentation:
    def __init__(self, config):
        self.config = config

    @property
    def active(self) -> bool:
        return self.config.enabled

    def _photon_thinning(self, counts, generator):
        integer_counts = np.maximum(np.round(counts), 0).astype(np.int64)
        return generator.binomial(integer_counts, self.config.photon_thinning_survival).astype(np.float32)

    def _gain_jitter(self, light, generator):
        gain = 1.0 + generator.normal(0.0, self.config.gain_jitter_sigma)
        return light * max(gain, 0.0)

    def _light_noise(self, light, generator):
        hit_mask = light > 0.0
        if self.config.light_noise_mode == "additive":
            light = light + np.where(hit_mask, generator.normal(0.0, self.config.light_noise_sigma, size=light.shape), 0.0)
        else:
            factors = 1.0 + generator.normal(0.0, self.config.light_noise_sigma, size=light.shape)
            light   = np.where(hit_mask, light * factors, light)
        return np.maximum(light, 0.0)

    def _sensor_dropout(self, light, generator):
        hit_mask  = light > 0.0
        drop_mask = (generator.random(size=light.shape) < self.config.sensor_dropout_probability) & hit_mask
        light[drop_mask] = 0.0
        return light

    def _spurious_activation(self, light, generator):
        dark_mask     = light <= 0.0
        activate_mask = (generator.random(size=light.shape) < self.config.spurious_activation_probability) & dark_mask
        activations   = generator.uniform(0.0, self.config.spurious_activation_max_light, size=light.shape)
        light[activate_mask] = activations[activate_mask]
        return light

    def apply_to_counts(self, counts, generator):
        if self.config.enabled and self.config.photon_thinning_enabled:
            counts = self._photon_thinning(counts, generator)
        return counts

    def apply_to_light(self, light, generator):
        if not self.config.enabled:
            return light

        light = light.astype(np.float32).copy()
        if self.config.gain_jitter_enabled:
            light = self._gain_jitter(light, generator)
        if self.config.light_noise_enabled:
            light = self._light_noise(light, generator)
        if self.config.sensor_dropout_enabled:
            light = self._sensor_dropout(light, generator)
        if self.config.spurious_activation_enabled:
            light = self._spurious_activation(light, generator)

        return np.maximum(light, 0.0)
