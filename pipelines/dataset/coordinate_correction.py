import os
import shutil
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from tools.monitoring.logger import Logger


class CoordinateTransform:

    def __init__(self):
        self.prefix_token = "pmod3"

    def apply_to_target(self, target_x: float, target_y: float, target_z: float):
        sensor_x = target_y
        sensor_y = -target_x
        sensor_z = target_z
        return sensor_x, sensor_y, sensor_z

    def _negated_token(self, value_token: str) -> str:
        numeric_value = float(value_token)
        if numeric_value == 0.0:
            return "0"
        if value_token.startswith("-"):
            return value_token[1:]
        return "-" + value_token

    def corrected_stem(self, original_stem: str) -> str:
        tokens = original_stem.split("_")

        prefix_token   = tokens[0]
        target_x_token = tokens[1]
        target_y_token = tokens[2]
        target_z_token = tokens[3]
        trailing       = tokens[4:]

        sensor_x_token = target_y_token
        sensor_y_token = self._negated_token(target_x_token)
        sensor_z_token = target_z_token

        corrected_tokens = [prefix_token, sensor_x_token, sensor_y_token, sensor_z_token] + trailing
        return "_".join(corrected_tokens)


class DatasetCorrector:

    def __init__(self, input_directory: Path, output_directory: Path, logger: Logger = None):
        self.input_directory     = Path(input_directory)
        self.output_directory    = Path(output_directory)
        self.raw_directory       = self.output_directory / "raw"
        self.corrected_directory = self.output_directory / "corrected"
        self.manifest_path       = self.output_directory / "coordinate_correction_manifest.csv"

        self.logger    = logger
        self.transform = CoordinateTransform()

    def discover_files(self):
        self.logger.section("[Coordinate Correction]")
        discovered_files = sorted(self.input_directory.glob("*.csv"))
        self.logger.subsection(f"Discovered {len(discovered_files)} raw CSV files in {self.input_directory}")
        return discovered_files

    def prepare_output_directories(self):
        os.makedirs(self.raw_directory, exist_ok=True)
        os.makedirs(self.corrected_directory, exist_ok=True)
        self.logger.subsection(f"Raw archive directory: {self.raw_directory}")
        self.logger.subsection(f"Corrected dataset directory: {self.corrected_directory}")

    def archive_raw_file(self, source_path: Path) -> Path:
        raw_target_path = self.raw_directory / source_path.name
        if raw_target_path.exists():
            raw_target_path.unlink()
        os.link(source_path, raw_target_path)
        return raw_target_path

    def write_corrected_file(self, source_path: Path) -> Path:
        corrected_name        = self.transform.corrected_stem(source_path.stem) + source_path.suffix
        corrected_target_path = self.corrected_directory / corrected_name
        shutil.copyfile(source_path, corrected_target_path)
        return corrected_target_path

    def build_manifest_row(self, source_path: Path, corrected_path: Path) -> dict:
        tokens = source_path.stem.split("_")

        target_x = float(tokens[1])
        target_y = float(tokens[2])
        target_z = float(tokens[3])

        sensor_x, sensor_y, sensor_z = self.transform.apply_to_target(target_x, target_y, target_z)

        return {
            "raw_filename"       : source_path.name,
            "raw_target_x"       : target_x,
            "raw_target_y"       : target_y,
            "raw_target_z"       : target_z,
            "corrected_sensor_x" : sensor_x,
            "corrected_sensor_y" : sensor_y,
            "corrected_sensor_z" : sensor_z,
            "corrected_filename" : corrected_path.name,
        }

    def write_manifest(self, manifest_rows: list) -> Path:
        manifest_frame = pd.DataFrame(manifest_rows)
        manifest_frame.to_csv(self.manifest_path, index=False)
        self.logger.subsection(f"Manifest written to {self.manifest_path} ({len(manifest_frame)} rows)")
        return self.manifest_path

    def run(self):
        discovered_files = self.discover_files()
        self.prepare_output_directories()

        manifest_rows = []
        with self.logger.track() as progress:
            task_id = progress.add_task("Correcting coordinates", total=len(discovered_files))
            for source_path in discovered_files:
                raw_target_path = self.archive_raw_file(source_path)
                corrected_path  = self.write_corrected_file(source_path)
                manifest_rows.append(self.build_manifest_row(source_path, corrected_path))
                progress.advance(task_id)

        self.write_manifest(manifest_rows)
        self.logger.subsection(
            f"Completed correction of {len(manifest_rows)} files "
            f"(raw archived, corrected written) \n"
        )
        return manifest_rows


class DatasetAugmentor:

    SIGN_COMBINATIONS = list(itertools.product((1, -1), repeat=3))

    def __init__(self, input_directory: Path, output_directory: Path, worker_count: int = 6, logger: Logger = None):
        self.input_directory     = Path(input_directory)
        self.output_directory    = Path(output_directory)
        self.augmented_directory = self.output_directory / "augmented"
        self.manifest_path       = self.output_directory / "octant_augmentation_manifest.csv"

        self.worker_count = worker_count
        self.logger       = logger

    def discover_files(self):
        self.logger.section("[Octant Augmentation]")
        discovered_files = sorted(self.input_directory.glob("*.csv"))
        self.logger.subsection(f"Discovered {len(discovered_files)} raw CSV files in {self.input_directory}")
        return discovered_files

    def prepare_output_directory(self):
        os.makedirs(self.augmented_directory, exist_ok=True)
        self.logger.subsection(f"Augmented dataset directory: {self.augmented_directory}")
        self.logger.subsection(f"Worker processes: {self.worker_count}")

    @staticmethod
    def _negated_token(value_token: str) -> str:
        numeric_value = float(value_token)
        if numeric_value == 0.0:
            return "0"
        if value_token.startswith("-"):
            return value_token[1:]
        return "-" + value_token

    @staticmethod
    def _octant_tag(sign_x: int, sign_y: int, sign_z: int) -> str:
        symbol = {1: "p", -1: "n"}
        return symbol[sign_x] + symbol[sign_y] + symbol[sign_z]

    @staticmethod
    def reflect_single_source(source_path_string: str, augmented_directory_string: str):
        source_path          = Path(source_path_string)
        augmented_directory  = Path(augmented_directory_string)
        transform            = CoordinateTransform()

        tokens = source_path.stem.split("_")

        prefix_token   = tokens[0]
        target_x_token = tokens[1]
        target_y_token = tokens[2]
        target_z_token = tokens[3]
        trailing       = tokens[4:]

        target_x = float(target_x_token)
        target_y = float(target_y_token)
        target_z = float(target_z_token)
        sensor_x, sensor_y, sensor_z = transform.apply_to_target(target_x, target_y, target_z)

        octant_x_token = target_y_token
        octant_y_token = DatasetAugmentor._negated_token(target_x_token)
        octant_z_token = target_z_token

        source_frame = pd.read_csv(source_path)

        produced_rows  = []
        emitted_targets = set()
        for sign_x, sign_y, sign_z in DatasetAugmentor.SIGN_COMBINATIONS:
            reflected_x_token = octant_x_token if sign_x > 0 else DatasetAugmentor._negated_token(octant_x_token)
            reflected_y_token = octant_y_token if sign_y > 0 else DatasetAugmentor._negated_token(octant_y_token)
            reflected_z_token = octant_z_token if sign_z > 0 else DatasetAugmentor._negated_token(octant_z_token)

            target_key = (reflected_x_token, reflected_y_token, reflected_z_token)
            if target_key in emitted_targets:
                continue
            emitted_targets.add(target_key)

            octant_tag      = DatasetAugmentor._octant_tag(sign_x, sign_y, sign_z)
            reflected_tokens = [prefix_token, reflected_x_token, reflected_y_token, reflected_z_token] + trailing + [octant_tag]
            reflected_name   = "_".join(reflected_tokens) + source_path.suffix
            reflected_path   = augmented_directory / reflected_name

            reflected_frame      = source_frame.copy()
            reflected_frame["x"] = reflected_frame["x"] * sign_x
            reflected_frame["y"] = reflected_frame["y"] * sign_y
            reflected_frame["z"] = reflected_frame["z"] * sign_z
            reflected_frame.to_csv(reflected_path, index=False)

            produced_rows.append({
                "source_filename"    : source_path.name,
                "octant"             : octant_tag,
                "target_x"           : sign_x * sensor_x,
                "target_y"           : sign_y * sensor_y,
                "target_z"           : sign_z * sensor_z,
                "augmented_filename" : reflected_name,
            })

        return produced_rows

    def write_manifest(self, manifest_rows: list) -> Path:
        manifest_frame = pd.DataFrame(manifest_rows)
        manifest_frame.to_csv(self.manifest_path, index=False)
        self.logger.subsection(f"Manifest written to {self.manifest_path} ({len(manifest_frame)} rows)")
        return self.manifest_path

    def run(self):
        discovered_files = self.discover_files()
        self.prepare_output_directory()

        augmented_directory_string = str(self.augmented_directory)
        manifest_rows = []
        with self.logger.track() as progress:
            task_id = progress.add_task("Augmenting octants", total=len(discovered_files))
            with ProcessPoolExecutor(max_workers=self.worker_count) as executor:
                futures = [
                    executor.submit(self.reflect_single_source, str(source_path), augmented_directory_string)
                    for source_path in discovered_files
                ]
                for future in as_completed(futures):
                    manifest_rows.extend(future.result())
                    progress.advance(task_id)

        self.write_manifest(manifest_rows)
        self.logger.subsection(
            f"Completed octant augmentation: {len(discovered_files)} source files "
            f"expanded to {len(manifest_rows)} events \n"
        )
        return manifest_rows


__all__ = [
    "CoordinateTransform",
    "DatasetCorrector",
    "DatasetAugmentor",
]
