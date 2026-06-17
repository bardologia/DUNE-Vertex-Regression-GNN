from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ray

from pipelines.dataset.ray_executor import RayExecutor


class DataLoader:
    def __init__(self, logger, config):
        self.logger              = logger
        self.config              = config
        self.executor            = RayExecutor(logger, config.ray)

        self.input_dir           = config.data.input_dir
        self.max_files           = config.data.max_files
        self.light_column        = config.data.light_column
        self.coordinate_columns  = config.data.coordinate_columns
        self.header_rows_to_skip  = config.data.header_rows_to_skip

    def _select_files(self, input_directory) -> list:
        input_dir = Path(input_directory) if input_directory else self.input_dir
        all_files = list(Path(input_dir).glob("*.csv"))

        if self.max_files is None or self.max_files >= len(all_files):
            return all_files

        random_generator = np.random.default_rng(self.config.split.random_state)
        indices          = random_generator.choice(len(all_files), size=self.max_files, replace=False)
        return [all_files[index] for index in indices]

    def _parse_files(self, file_list: list) -> list:
        self.executor.ensure_initialized()

        def parse_file_batch(file_path_batch, header_rows_to_skip, light_column):
            batch_results = []
            for file_path_string in file_path_batch:
                file_path   = Path(file_path_string)
                parts       = file_path.stem.split("_")
                coordinates = (float(parts[1]), float(parts[2]), float(parts[3]))

                dataframe = pd.read_csv(
                    file_path,
                    skiprows = header_rows_to_skip,
                    names    = ["bin", "x", "y", "z", light_column],
                    comment  = "b",
                    dtype    = {"x": float, "y": float, "z": float, light_column: float},
                )
                batch_results.append((dataframe, coordinates))
            return batch_results

        remote_parse  = ray.remote(parse_file_batch)
        file_strings  = [str(file_path) for file_path in file_list]
        file_batches  = self.executor.make_batches(file_strings)
        batch_results = self.executor.run_batches(
            remote_parse,
            [(file_batch, self.header_rows_to_skip, self.light_column) for file_batch in file_batches],
            "Loading files (Ray batches)",
        )

        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        return results

    def load(self, input_directory=None):
        self.logger.section("[Data Loading]")
        file_list = self._select_files(input_directory)
        self.logger.subsection(f"Loading {len(file_list)} files")

        if len(file_list) == 0:
            self.logger.warning("No CSV files found; returning empty dataset.")
            return [], pd.DataFrame(columns=list(self.coordinate_columns))

        results              = self._parse_files(file_list)
        dataframes           = [dataframe for dataframe, _ in results]
        coordinate_rows      = [coordinates for _, coordinates in results]
        coordinate_dataframe = pd.DataFrame(coordinate_rows, columns=list(self.coordinate_columns))

        self.logger.subsection(f"Loaded {len(dataframes)} dataframes with coordinates")
        return dataframes, coordinate_dataframe
