import numpy as np
import pandas as pd

from configuration.data.general import HotChannelConfig
from pipelines.dataset          import EndcapFaceCorrector, HotChannelCorrector, ParquetEventReader


class _SilentLogger:
    def subsection(self, message):
        return None


def test_endcap_face_corrector_rotates_only_extreme_planes():
    z_values = np.array([0.0, 10.0, -10.0, 29.25, -29.25, 29.926, -29.926], dtype=np.float64)
    y_values = np.array([1.0,  2.0,   3.0,  4.00,   5.00,   6.000,   7.000], dtype=np.float64)
    x_values = np.array([1.0,  2.0,   3.0,  4.00,   5.00,   6.000,   7.000], dtype=np.float64)
    frame    = pd.DataFrame({"bin": np.arange(len(z_values)), "x": x_values, "y": y_values, "z": z_values})

    corrected = EndcapFaceCorrector(logger=_SilentLogger()).apply(frame)

    flipped_z = corrected["z"].to_numpy()
    flipped_y = corrected["y"].to_numpy()
    flipped_x = corrected["x"].to_numpy()

    assert flipped_z[5] == -29.926 and flipped_z[6] == 29.926
    assert flipped_y[5] == -6.000  and flipped_y[6] == -7.000
    assert np.array_equal(flipped_z[:5], z_values[:5])
    assert np.array_equal(flipped_y[:5], y_values[:5])
    assert np.array_equal(flipped_x, x_values)


def _hot_channel_setup():
    generator = np.random.default_rng(0)
    positions = generator.normal(size=(60, 3))
    light     = generator.integers(0, 10, size=(400, 60)).astype(np.int32)

    hot = 7
    light[:, hot] = generator.integers(50_000, 100_000, size=400).astype(np.int32)
    return positions, light, hot


def test_hot_channel_corrector_replaces_with_neighbour_mean():
    positions, light, hot = _hot_channel_setup()
    config = HotChannelConfig(active_fraction=0.999, median_factor=50.0, neighbor_count=8, min_events=100)

    corrected = HotChannelCorrector(positions, config, _SilentLogger()).run(light.copy())

    assert corrected[:, hot].max() < 50_000
    assert corrected.dtype == light.dtype
    untouched = [c for c in range(light.shape[1]) if c != hot]
    assert np.array_equal(corrected[:, untouched], light[:, untouched])


def test_hot_channel_corrector_skips_small_datasets():
    positions, light, hot = _hot_channel_setup()
    config = HotChannelConfig(min_events=10_000)

    corrected = HotChannelCorrector(positions, config, _SilentLogger()).run(light.copy())
    assert np.array_equal(corrected, light)


def test_parquet_reader_iterates_corrected(parquet_store):
    reader = ParquetEventReader(parquet_store).load_store()
    frames = list(reader.iterate_corrected())

    assert len(frames) > 0
    event_frame, target = frames[0]
    assert list(event_frame.columns) == ["bin", "x", "y", "z", "ly_amount"]
    assert len(target) == 3


def test_parquet_reader_augmentation_expands(parquet_store):
    reader    = ParquetEventReader(parquet_store).load_store()
    corrected = list(reader.iterate_corrected())
    augmented = list(reader.iterate_augmented())
    assert len(augmented) >= len(corrected)


def test_parquet_store_arrays(parquet_store):
    reader = ParquetEventReader(parquet_store).load_store()
    assert reader.light_matrix.shape[1] == len(reader.geometry_frame)
    assert reader.event_targets.shape[1] == 3
