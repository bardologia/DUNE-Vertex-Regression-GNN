from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from configuration.shared import IoNames


class StoreGeometryRecord:
    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / IoNames.store_geometry_file

    def save(self, store_geometry: dict) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(store_geometry, indent=IoNames.json_indent), encoding="utf-8")
        return self.path

    def recorded(self) -> dict:
        if not self.path.parent.is_dir():
            raise FileNotFoundError(f"{self.path.parent} does not exist; pass the directory of a completed run.")
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path.parent} holds no {IoNames.store_geometry_file}: the run was trained on a Parquet store built before the documented channel mapping, whose sensor positions are wrong for every odd raw row and which kept the three face totals as channels. Its checkpoint and statistics cannot be evaluated, resumed or reused on the rebuilt store; train a new run.")
        return json.loads(self.path.read_text(encoding="utf-8"))

    @classmethod
    def for_run(cls, run_directory: Path) -> "StoreGeometryRecord":
        record = cls(Path(run_directory) / IoNames.metadata_dir)
        record.recorded()
        return record

    def require(self, store_geometry: dict) -> None:
        recorded = self.recorded()
        if recorded["geometry_fingerprint"] != store_geometry["geometry_fingerprint"]:
            raise ValueError(f"{self.path.parent} was built on store geometry {recorded['geometry_fingerprint']} ({recorded['channel_count']} channels) but the store in use has {store_geometry['geometry_fingerprint']} ({store_geometry['channel_count']} channels); a checkpoint, its statistics and its results only hold for the sensor geometry they were trained on.")


class CacheGeometryStamp:
    @staticmethod
    def payload(fingerprint: str) -> dict:
        return {"geometry_fingerprint": np.array(fingerprint)}

    @staticmethod
    def require(archive, cache_path: Path, fingerprint: str) -> None:
        keys = archive.files if hasattr(archive, "files") else list(archive)
        if "geometry_fingerprint" not in keys:
            raise ValueError(f"{cache_path} carries no geometry_fingerprint: it was built from a Parquet store before the documented channel mapping, with misplaced sensors and the three face totals as channels. Delete it and let the console rebuild it from the current store.")

        stamped = str(archive["geometry_fingerprint"])
        if stamped != fingerprint:
            raise ValueError(f"{cache_path} was built from store geometry {stamped} but the current store has {fingerprint}. Delete it and let the console rebuild it from the current store.")
