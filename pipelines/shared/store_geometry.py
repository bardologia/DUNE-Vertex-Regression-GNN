"""Records of the Parquet-store geometry that a run or a cache was built from, and the checks that refuse any other geometry.

The store geometry is identified by ``ParquetEventReader.fingerprint``, the
sha256 of the stored channel table. A run keeps it in
``metadata/store_geometry.json`` and an npz cache under the key
``geometry_fingerprint``. Runs and caches written before the documented
channel mapping carry neither, so they are refused instead of being scored or
displayed against sensor positions they were not built from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from configuration.shared import IoNames


class StoreGeometryRecord:
    """Store-geometry record of one run directory.

    Attributes:
        FILENAME : Name of the record file inside the directory.
        path     : Path of the record file.
    """

    def __init__(self, directory: Path) -> None:
        """Resolves the record path.

        Args:
            directory : Directory that holds the record, the ``metadata`` directory of a run or the ``pipeline`` directory of a benchmark.
        """
        self.path = Path(directory) / IoNames.store_geometry_file

    def save(self, store_geometry: dict) -> Path:
        """Writes the record.

        Args:
            store_geometry : Mapping with the store ``geometry_fingerprint`` and ``channel_count``, from ``DatasetPipeline.store_geometry``.

        Returns:
            The record path.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(store_geometry, indent=IoNames.json_indent), encoding="utf-8")
        return self.path

    def recorded(self) -> dict:
        """Returns the record.

        Returns:
            The mapping with ``geometry_fingerprint`` and ``channel_count``.

        Raises:
            FileNotFoundError : If the directory does not exist, or holds no record, as every run trained before the documented channel mapping.
        """
        if not self.path.parent.is_dir():
            raise FileNotFoundError(f"{self.path.parent} does not exist; pass the directory of a completed run.")
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path.parent} holds no {IoNames.store_geometry_file}: the run was trained on a Parquet store built before the documented channel mapping, whose sensor positions are wrong for every odd raw row and which kept the three face totals as channels. Its checkpoint and statistics cannot be evaluated, resumed or reused on the rebuilt store; train a new run.")
        return json.loads(self.path.read_text(encoding="utf-8"))

    @classmethod
    def for_run(cls, run_directory: Path) -> "StoreGeometryRecord":
        """Returns the record of a run after checking that the run holds one.

        Pipelines that open a log file inside the run call this before the Logger,
        which opens its file in overwrite mode, so a run trained before the
        documented channel mapping is refused with its logs intact.

        Args:
            run_directory : Directory of a completed run, ``runs/<model>_<stamp>``.

        Returns:
            The record of the run's ``metadata`` directory.

        Raises:
            FileNotFoundError : If the run directory does not exist or holds no record.
        """
        record = cls(Path(run_directory) / IoNames.metadata_dir)
        record.recorded()
        return record

    def require(self, store_geometry: dict) -> None:
        """Checks that the record names the given store geometry.

        Args:
            store_geometry : Mapping with the ``geometry_fingerprint`` of the store about to be used.

        Raises:
            FileNotFoundError : If the directory holds no record.
            ValueError        : If the recorded fingerprint differs from the given one.
        """
        recorded = self.recorded()
        if recorded["geometry_fingerprint"] != store_geometry["geometry_fingerprint"]:
            raise ValueError(f"{self.path.parent} was built on store geometry {recorded['geometry_fingerprint']} ({recorded['channel_count']} channels) but the store in use has {store_geometry['geometry_fingerprint']} ({store_geometry['channel_count']} channels); a checkpoint, its statistics and its results only hold for the sensor geometry they were trained on.")


class CacheGeometryStamp:
    """Store-geometry stamp of an npz cache.

    Attributes:
        KEY : Archive key holding the fingerprint of the store geometry the cache was built from.
    """

    KEY = "geometry_fingerprint"

    @classmethod
    def payload(cls, fingerprint: str) -> dict:
        """Returns the archive entry that stamps a cache with a store geometry.

        Args:
            fingerprint : Fingerprint of the store geometry the cache is built from.

        Returns:
            A one-entry mapping to merge into the ``np.savez_compressed`` payload.
        """
        return {cls.KEY: np.array(fingerprint)}

    @classmethod
    def require(cls, archive, cache_path: Path, fingerprint: str) -> None:
        """Checks that a loaded cache carries the given store geometry.

        Args:
            archive     : Loaded npz archive, or a mapping of its arrays.
            cache_path  : Path of the cache, named in the error.
            fingerprint : Fingerprint of the store geometry the cache must have been built from.

        Raises:
            ValueError : If the cache carries no stamp, as every cache written before the documented channel mapping, or a stamp of another geometry.
        """
        keys = archive.files if hasattr(archive, "files") else list(archive)
        if cls.KEY not in keys:
            raise ValueError(f"{cache_path} carries no {cls.KEY}: it was built from a Parquet store before the documented channel mapping, with misplaced sensors and the three face totals as channels. Delete it and let the console rebuild it from the current store.")

        stamped = str(archive[cls.KEY])
        if stamped != fingerprint:
            raise ValueError(f"{cache_path} was built from store geometry {stamped} but the current store has {fingerprint}. Delete it and let the console rebuild it from the current store.")
