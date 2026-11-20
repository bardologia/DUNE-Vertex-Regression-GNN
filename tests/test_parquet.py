import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pa_parquet
import pytest

from configuration.data.general import StoreLayout
from pipelines.dataset          import ChannelMapping, ParquetDatasetWriter, ParquetEventReader


def test_parquet_reader_iterates_corrected(parquet_store):
    reader = ParquetEventReader(parquet_store).load_store()
    frames = list(reader.iterate_corrected())

    assert len(frames) > 0
    event_frame, target = frames[0]
    assert list(event_frame.columns) == ["bin", "x", "y", "z", "ly_amount"]
    assert len(target) == 3


def test_parquet_octant_index_expands_every_event(parquet_store):
    reader = ParquetEventReader(parquet_store).load_store()
    frame  = reader.octant_frame

    assert "light" not in frame.columns
    assert len(frame) > len(reader.light_matrix)
    assert set(frame["base_event_id"]) == set(range(len(reader.light_matrix)))
    assert frame.groupby("base_event_id").size().min() >= 2


def test_parquet_store_arrays(parquet_store):
    reader = ParquetEventReader(parquet_store).load_store()
    assert reader.light_matrix.shape[1] == len(reader.geometry_frame) == 6909
    assert reader.event_targets.shape[1] == 3


def test_store_geometry_is_the_light_derived_geometry(parquet_store, light_derived):
    frame = ParquetEventReader(parquet_store).load_geometry().geometry_frame

    assert tuple(frame.columns) == StoreLayout.geometry_columns
    assert np.array_equal(frame["channel_index"].to_numpy(), np.arange(6909))
    assert np.array_equal(frame["raw_channel_index"].to_numpy(), light_derived.kept_rows)
    assert np.array_equal(frame["bin"].to_numpy(), light_derived.bins)
    assert np.abs(frame[["x", "y", "z"]].to_numpy(dtype=np.float64) - light_derived.positions).max() < 2e-6
    assert frame["face"].value_counts().to_dict() == {"left_wall": 2880, "right_wall": 2878, "back_endcap": 576, "front_endcap": 575}
    assert np.unique(np.round(frame[["x", "y", "z"]].to_numpy(dtype=np.float64), 4), axis=0).shape[0] == 6909


def test_store_column_types(parquet_store):
    schema = pa_parquet.read_schema(parquet_store / "geometry.parquet")
    assert [schema.field(name).type for name in StoreLayout.geometry_columns] == [pa.int32(), pa.int32(), pa.int32(), pa.int32(), pa.string(), pa.float32(), pa.float32(), pa.float32()]


def test_store_light_equals_the_raw_counts_of_the_kept_rows(parquet_store, sample_csv_dir):
    reader = ParquetEventReader(parquet_store).load_store()
    names  = pa_parquet.read_table(parquet_store / "events.parquet", columns=["source_filename"]).column("source_filename").to_pylist()
    rows   = reader.geometry_frame["raw_channel_index"].to_numpy()

    assert reader.light_matrix.dtype == np.int32
    for event, name in enumerate(names):
        raw = pd.read_csv(sample_csv_dir / name)["ly_amount"].to_numpy()
        assert np.array_equal(reader.light_matrix[event], raw[rows].astype(np.int32)), name


def test_store_fingerprint_follows_the_geometry(parquet_store, raw_frame, quiet_logger):
    reader = ParquetEventReader(parquet_store).load_geometry()
    again  = ParquetEventReader(parquet_store).load_geometry()
    built  = ChannelMapping(quiet_logger).build(raw_frame)
    edited = reader.geometry_frame.copy()

    edited.loc[5, "x"] = edited.loc[5, "x"] + np.float32(0.001)

    assert again.geometry_fingerprint == reader.geometry_fingerprint
    assert ParquetEventReader.fingerprint(built) == reader.geometry_fingerprint
    assert ParquetEventReader.fingerprint(edited) != reader.geometry_fingerprint


def test_reader_refuses_a_store_without_the_raw_channel_index(tmp_path, parquet_store):
    old_store = tmp_path / "parquet"
    shutil.copytree(parquet_store, old_store)

    geometry = pa_parquet.read_table(old_store / "geometry.parquet").drop(["raw_channel_index", "module_index", "face"])
    pa_parquet.write_table(geometry, old_store / "geometry.parquet")

    with pytest.raises(ValueError, match="raw_channel_index"):
        ParquetEventReader(old_store).load_store()


def test_writer_refuses_a_raw_layout_with_other_bins(tmp_path, sample_csv_dir, quiet_logger):
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()

    source              = sorted(sample_csv_dir.glob("*.csv"))[0]
    frame               = pd.read_csv(source)
    frame.loc[2, "bin"] = 14.0
    frame.to_csv(raw_directory / source.name, index=False)

    with pytest.raises(ValueError, match="raw bins are not exactly"):
        ParquetDatasetWriter(raw_directory, tmp_path / "out", worker_count=1, logger=quiet_logger).run()
