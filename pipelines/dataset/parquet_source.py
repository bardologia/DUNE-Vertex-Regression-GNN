from __future__ import annotations

import pandas as pd

from pipelines.dataset.parquet_store import ParquetEventReader


class ParquetGraphSource:
    def __init__(self, logger, config):
        self.logger             = logger
        self.config             = config
        self.store_directory    = config.data.parquet_store_dir
        self.augment_octants    = config.data.augment_octants
        self.light_column       = config.data.light_column
        self.coordinate_columns = config.data.coordinate_columns

    def load(self):
        self.logger.section("[Parquet Source]")
        self.logger.subsection(f"Store: {self.store_directory} | augment_octants={self.augment_octants}")

        reader = ParquetEventReader(self.store_directory).load_store()

        dataframes = []
        targets    = []
        iterator   = reader.iterate_augmented() if self.augment_octants else reader.iterate_corrected()

        for item in iterator:
            event_frame, target = item[0], item[1]
            dataframes.append(event_frame.rename(columns={"ly_amount": self.light_column}))
            targets.append(target)

        coordinate_dataframe = pd.DataFrame(targets, columns=list(self.coordinate_columns))
        self.logger.subsection(f"Loaded {len(dataframes)} events from the Parquet store")
        return dataframes, coordinate_dataframe
