from pipelines.dataset import ParquetEventReader


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
