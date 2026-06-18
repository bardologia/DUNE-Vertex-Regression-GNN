import numpy as np

from configuration import AugmentationConfig
from pipelines.dataset import Augmentation


def _generator():
    return np.random.default_rng(0)


def test_disabled_augmentation_is_identity():
    augmentation = Augmentation(AugmentationConfig(enabled=False))
    light        = np.array([0.0, 5.0, 10.0, 0.0, 3.0], dtype=np.float32)
    assert np.array_equal(augmentation.apply_to_light(light.copy(), _generator()), light)


def test_sensor_dropout_deactivates_hits():
    config = AugmentationConfig(enabled=True, sensor_dropout_enabled=True, sensor_dropout_probability=1.0,
                                spurious_activation_enabled=False, light_noise_enabled=False, gain_jitter_enabled=False)
    light  = np.array([5.0, 8.0, 0.0, 2.0], dtype=np.float32)
    result = Augmentation(config).apply_to_light(light.copy(), _generator())
    assert np.all(result == 0.0)


def test_spurious_activation_lights_dark_sensors():
    config = AugmentationConfig(enabled=True, sensor_dropout_enabled=False, light_noise_enabled=False, gain_jitter_enabled=False,
                                spurious_activation_enabled=True, spurious_activation_probability=1.0, spurious_activation_max_light=1.0)
    light  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    result = Augmentation(config).apply_to_light(light.copy(), _generator())
    assert np.all(result > 0.0)


def test_photon_thinning_reduces_counts():
    config = AugmentationConfig(enabled=True, photon_thinning_enabled=True, photon_thinning_survival=0.5)
    counts = np.full(5000, 100.0, dtype=np.float32)
    result = Augmentation(config).apply_to_counts(counts, _generator())
    assert result.mean() < counts.mean()
