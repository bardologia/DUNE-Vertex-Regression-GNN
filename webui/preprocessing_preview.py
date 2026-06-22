from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib     import Path

import numpy as np

from configuration.data.general      import AugmentationConfig, DatasetConfig, HotChannelConfig, PhysicsConfig
from pipelines.dataset.augmentation  import Augmentation
from pipelines.dataset.hot_channels  import HotChannelCorrector
from pipelines.dataset.parquet_store import ParquetEventReader

from project_paths     import ProjectPaths
from server_logger     import ServerLogger
from subprocess_runner import SubprocessRunner


class PreprocessingPreview:

    FILENAME    = "raw_light.npz"
    STAGE_NAMES = ("raw", "cleaned", "scaled", "detected", "augmented")

    def __init__(self, paths: ProjectPaths, logger: ServerLogger) -> None:
        self.paths  = paths
        self.logger = logger
        self.runner = SubprocessRunner(paths, logger)
        self.lock   = threading.Lock()

        self.loaded = False
        self.status = {"state": "idle", "stage": "", "error": ""}

        self.geometry     = None
        self.light_matrix = None
        self.octant_base  = None
        self.octant_signs = None
        self.octant_gt    = None
        self.octant_tags  = None
        self.bounds_min   = [0.0, 0.0, 0.0]
        self.bounds_max   = [0.0, 0.0, 0.0]

        self.hot_cache_key = None
        self.hot_corrector = None
        self.hot_channels  = None
        self.hot_neighbors = None

    def defaults(self) -> dict:
        physics = PhysicsConfig()
        return {
            "scaling"      : {"scale_factor": physics.scale_factor},
            "efficiency"   : {"detection_efficiency": physics.detection_efficiency, "efficiency_seed": physics.efficiency_seed},
            "outlier"      : asdict(HotChannelConfig()),
            "augmentation" : asdict(AugmentationConfig()),
        }

    def start_load(self) -> dict:
        with self.lock:
            if self.status["state"] == "loading":
                return {"ok": False, "error": "already loading the raw-light source"}
            if self.loaded:
                return {"ok": True}
            self.status = {"state": "loading", "stage": "preparing", "error": ""}

        threading.Thread(target=self._load_worker, daemon=True).start()
        return {"ok": True}

    def load_status(self) -> dict:
        with self.lock:
            payload = dict(self.status)
            if payload["state"] == "ready":
                payload["defaults"] = self.defaults()
        return payload

    def events(self) -> dict:
        with self.lock:
            if not self.loaded:
                return {"ok": False, "error": "preview source not loaded"}
            return {
                "ok"           : True,
                "count"        : int(self.octant_gt.shape[0]),
                "gt"           : self.octant_gt.tolist(),
                "bounds_min"   : self.bounds_min,
                "bounds_max"   : self.bounds_max,
                "detector_min" : self.geometry.min(axis=0).tolist(),
                "detector_max" : self.geometry.max(axis=0).tolist(),
                "defaults"     : self.defaults(),
            }

    def preview(self, index: int, config: dict) -> dict:
        with self.lock:
            if not self.loaded:
                return {"ok": False, "error": "preview source not loaded"}

        if index < 0 or index >= self.octant_base.shape[0]:
            return {"ok": False, "error": f"index {index} out of range [0, {self.octant_base.shape[0]})"}

        base_event_id = int(self.octant_base[index])
        signs         = self.octant_signs[index]
        positions     = self.geometry * signs[None, :]
        raw_row       = self.light_matrix[base_event_id].astype(np.float64)

        stages = self._stages(positions, raw_row, base_event_id, index, config)

        active     = np.nonzero(raw_row > 0.0)[0]
        framed     = positions[active] if active.size else positions
        bounds_min = framed.min(axis=0).astype(np.float32).tolist()
        bounds_max = framed.max(axis=0).astype(np.float32).tolist()

        return {
            "ok"            : True,
            "index"         : index,
            "base_event_id" : base_event_id,
            "octant"        : str(self.octant_tags[index]),
            "gt"            : self.octant_gt[index].tolist(),
            "stages"        : stages,
            "bounds_min"    : bounds_min,
            "bounds_max"    : bounds_max,
        }

    def _stages(self, positions, raw_row, base_event_id, index, config) -> list:
        cleaned   = self._cleaned(raw_row, config["outlier"])
        scaled    = self._scaled(cleaned, config["scaling"])
        detected  = self._detected(scaled, base_event_id, config["efficiency"])
        augmented = self._augmented(detected, index, config["augmentation"])

        return [
            self._stage_payload("raw",       positions, raw_row),
            self._stage_payload("cleaned",   positions, cleaned),
            self._stage_payload("scaled",    positions, scaled),
            self._stage_payload("detected",  positions, detected),
            self._stage_payload("augmented", positions, augmented),
        ]

    def _cleaned(self, raw_row, outlier) -> np.ndarray:
        if not bool(outlier["enabled"]):
            return raw_row.copy()

        key = (round(float(outlier["active_fraction"]), 9), round(float(outlier["median_factor"]), 9), int(outlier["neighbor_count"]), int(outlier["min_events"]))
        if key != self.hot_cache_key:
            config    = HotChannelConfig(enabled=True, active_fraction=float(outlier["active_fraction"]), median_factor=float(outlier["median_factor"]), neighbor_count=int(outlier["neighbor_count"]), min_events=int(outlier["min_events"]))
            corrector = HotChannelCorrector(self.geometry, config, self.logger)

            self.hot_corrector                    = corrector
            self.hot_channels, self.hot_neighbors = corrector.fit(self.light_matrix)
            self.hot_cache_key                    = key

        return self.hot_corrector.apply_to_event(raw_row, self.hot_channels, self.hot_neighbors).astype(np.float64)

    def _scaled(self, cleaned, scaling) -> np.ndarray:
        scale = float(scaling["scale_factor"])
        return np.maximum(np.round(np.asarray(cleaned, dtype=np.float64) * scale), 0.0)

    def _detected(self, scaled, base_event_id, efficiency) -> np.ndarray:
        generator = np.random.default_rng(int(efficiency["efficiency_seed"]) + int(base_event_id))
        counts    = np.asarray(scaled, dtype=np.int64)
        return generator.binomial(counts, float(efficiency["detection_efficiency"])).astype(np.float64)

    def _augmented(self, detected, index, augmentation) -> np.ndarray:
        config = self._augmentation_config(augmentation)
        if not config.enabled:
            return np.asarray(detected, dtype=np.float64)

        engine    = Augmentation(config)
        generator = np.random.default_rng(config.seed + int(index))
        counts    = engine.apply_to_counts(np.asarray(detected, dtype=np.float32), generator)
        light     = engine.apply_to_light(counts, generator)
        return light.astype(np.float64)

    def _augmentation_config(self, augmentation) -> AugmentationConfig:
        return AugmentationConfig(
            enabled                         = bool(augmentation["enabled"]),
            seed                            = int(augmentation["seed"]),
            sensor_dropout_enabled          = bool(augmentation["sensor_dropout_enabled"]),
            sensor_dropout_probability      = float(augmentation["sensor_dropout_probability"]),
            spurious_activation_enabled     = bool(augmentation["spurious_activation_enabled"]),
            spurious_activation_probability = float(augmentation["spurious_activation_probability"]),
            spurious_activation_max_light   = float(augmentation["spurious_activation_max_light"]),
            light_noise_enabled             = bool(augmentation["light_noise_enabled"]),
            light_noise_sigma               = float(augmentation["light_noise_sigma"]),
            light_noise_mode                = str(augmentation["light_noise_mode"]),
            photon_thinning_enabled         = bool(augmentation["photon_thinning_enabled"]),
            photon_thinning_survival        = float(augmentation["photon_thinning_survival"]),
            gain_jitter_enabled             = bool(augmentation["gain_jitter_enabled"]),
            gain_jitter_sigma               = float(augmentation["gain_jitter_sigma"]),
        )

    def _stage_payload(self, name, positions, light) -> dict:
        light  = np.asarray(light, dtype=np.float64)
        active = np.nonzero(light > 0.0)[0]
        pos    = positions[active].astype(np.float32)
        lit    = light[active].astype(np.float32)

        return {
            "name"        : name,
            "positions"   : pos.tolist(),
            "light"       : lit.tolist(),
            "n_active"    : int(active.size),
            "total_light" : float(lit.sum()),
        }

    def _load_worker(self) -> None:
        try:
            cache = self.paths.explorer_cache_dir / self.FILENAME
            if not cache.is_file():
                self._set_stage("building the raw-light cache")
                self.runner.spawn(self.paths.main_dir / "export_raw_light.py", [], "raw_light_export")
            if not cache.is_file():
                raise FileNotFoundError(f"export finished but {cache} was not produced; check the export log")

            self._set_stage("loading caches")
            self._load_arrays(cache)

            with self.lock:
                self.loaded = True
                self.status = {"state": "ready", "stage": "ready", "error": ""}

            self.logger.ok(f"preprocessing preview ready: {self.octant_gt.shape[0]} octant events")
        except Exception as error:
            with self.lock:
                self.loaded = False
                self.status = {"state": "error", "stage": "", "error": str(error)}

            self.logger.error(f"preprocessing preview load failed: {error}")

    def _set_stage(self, stage: str) -> None:
        with self.lock:
            self.status["stage"] = stage

    def _load_arrays(self, cache: Path) -> None:
        archive  = np.load(cache)
        geometry = archive["geometry_positions"].astype(np.float32)
        light    = archive["light_matrix"]

        reader = ParquetEventReader(DatasetConfig().data.parquet_store_dir).load_store(load_octant_light=False)
        frame  = reader.octant_frame

        with self.lock:
            self.geometry     = geometry
            self.light_matrix = light
            self.octant_base  = frame["base_event_id"].values.astype(np.int64)
            self.octant_signs = frame[["sign_x", "sign_y", "sign_z"]].values.astype(np.float32)
            self.octant_gt    = frame[["target_x", "target_y", "target_z"]].values.astype(np.float32)
            self.octant_tags  = frame["octant"].values.astype(str)
            self.bounds_min   = self.octant_gt.min(axis=0).tolist()
            self.bounds_max   = self.octant_gt.max(axis=0).tolist()
