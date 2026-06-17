from pipelines.dataset import ParquetEventReader, ParquetGraphSource

from configuration import DatasetConfig


def test_parquet_reader_iterates_corrected(parquet_store):
    reader  = ParquetEventReader(parquet_store).load_store()
    frames  = list(reader.iterate_corrected())

    assert len(frames) > 0
    event_frame, target = frames[0]
    assert list(event_frame.columns) == ["bin", "x", "y", "z", "ly_amount"]
    assert len(target) == 3


def test_parquet_reader_augmentation_expands(parquet_store):
    reader        = ParquetEventReader(parquet_store).load_store()
    corrected     = list(reader.iterate_corrected())
    augmented     = list(reader.iterate_augmented())
    assert len(augmented) >= len(corrected)


def test_parquet_graph_source_renames_light(parquet_store, quiet_logger):
    config = DatasetConfig()
    config.data.source            = "parquet"
    config.data.parquet_store_dir = parquet_store

    dataframes, targets = ParquetGraphSource(quiet_logger, config).load()
    assert len(dataframes) == len(targets)
    assert config.data.light_column in dataframes[0].columns
    assert list(targets.columns) == list(config.data.coordinate_columns)
