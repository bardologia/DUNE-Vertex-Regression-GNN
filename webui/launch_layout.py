from __future__ import annotations

import copy


class LayoutError(Exception):
    pass


class LaunchLayout:

    PICK_RUN = {"kind": "run"}

    CH_DEVICE    = {"kind": "choice", "options": ["cuda", "cpu"]}
    CH_DATA_TERM = {"kind": "choice", "options": ["mse", "huber"]}
    CH_PLATEAU   = {"kind": "choice", "options": ["min", "max"]}
    CH_NOISE     = {"kind": "choice", "options": ["multiplicative", "additive"]}
    CH_SPLIT     = {"kind": "choice", "options": ["test", "val", "train"]}
    CH_SCHEDULER = {"kind": "choice", "options": ["reduce_on_plateau", "cosine_annealing", "linear", "polynomial", "exponential", "step", "constant"]}
    CH_WARMUP    = {"kind": "choice", "options": ["linear", "cosine", "exponential", "polynomial"]}
    CH_CLIP      = {"kind": "choice", "options": ["fixed", "adaptive_percentile", "adaptive_mean_std", "disabled"]}
    CH_FRAME     = {"kind": "choice", "options": ["isotropic", "axiswise"]}

    MULTI_SPLITS = {"kind": "multi", "empty": "select at least one split", "choices": [
        {"value": "train", "label": "train"},
        {"value": "val",   "label": "val"},
        {"value": "test",  "label": "test"},
    ]}

    MULTI_BASELINES = {"kind": "multi", "empty": "select at least one model", "choices": [
        {"value": "lightgbm", "label": "LightGBM"},
        {"value": "xgboost",  "label": "XGBoost"},
    ]}

    NUM_EPOCHS   = {"kind": "number", "int": True, "min": 1, "max": 1000, "presets": [25, 50, 100, 200, 300]}
    NUM_BATCH    = {"kind": "number", "int": True, "min": 1, "max": 4096, "presets": [8, 16, 32, 64, 128, 256]}
    NUM_WORKERS  = {"kind": "number", "int": True, "min": 0, "max": 64,   "presets": [0, 2, 4, 8, 16]}
    NUM_SEED     = {"kind": "number", "int": True, "min": 0, "max": 9999, "presets": [0, 1, 42, 1234]}
    NUM_PATIENCE = {"kind": "number", "int": True, "min": 1, "max": 200,  "presets": [5, 10, 15, 25, 50]}
    NUM_NEIGHBOR = {"kind": "number", "int": True, "min": 2, "max": 64,   "presets": [4, 8, 16, 32]}
    NUM_RBF      = {"kind": "number", "int": True, "min": 4, "max": 64,   "presets": [8, 16, 32]}
    NUM_CPU      = {"kind": "number", "int": True, "min": 1, "max": 64,   "presets": [4, 8, 10, 16]}
    NUM_TRIALS   = {"kind": "number", "int": True, "min": 1, "max": 1000, "presets": [20, 50, 100, 200]}
    NUM_FOLDS    = {"kind": "number", "int": True, "min": 2, "max": 20,   "presets": [3, 5, 10]}
    NUM_TREES    = {"kind": "number", "int": True, "min": 100, "max": 20000, "presets": [1000, 2000, 3000, 5000]}
    NUM_ROUNDS   = {"kind": "number", "int": True, "min": 10, "max": 1000,  "presets": [50, 100, 200]}
    NUM_TOPCHAN  = {"kind": "number", "int": True, "min": 1, "max": 32,     "presets": [3, 5, 8, 12]}

    TEMPLATES = {
        "dataset": [
            {"title": "Store", "fields": [
                "data.raw_input_dir",
                "data.build_store",
                {"path": "data.store_worker_count", "widget": NUM_CPU},
                "data.stats_sample_size",
                {"path": "data.target_frame", "widget": CH_FRAME},
            ]},
            {"title": "Sampling", "fields": [
                "data.augment_octants",
                "data.subset_fraction",
                "data.coordinate_columns",
                "data.position_columns",
                "data.detector_half_extent",
            ]},
            {"title": "Hot channels", "fields": [
                {"gate": "data.hot_channels.enabled", "fields": [
                    "data.hot_channels.active_fraction",
                    "data.hot_channels.median_factor",
                    "data.hot_channels.neighbor_count",
                    "data.hot_channels.min_events",
                ]},
            ]},
            {"title": "Graph", "fields": [
                {"path": "graph.k_neighbors", "widget": NUM_NEIGHBOR},
                "graph.bidirectional",
                "graph.active_only",
                "graph.max_active_nodes",
                "graph.global_node",
                {"path": "graph.edge_rbf_count", "widget": NUM_RBF},
                "graph.edge_rbf_max",
            ]},
            {"title": "Features", "fields": [
                "graph.direction_features",
                "graph.inertia_features",
                "graph.rank_features",
                "graph.graph_scalar_features",
                "graph.graph_moment_features",
                "graph.graph_quantiles",
            ]},
            {"title": "Readout physics", "fields": [
                "physics.scale_factor",
                "physics.detection_efficiency",
                "physics.efficiency_seed",
            ]},
            {"title": "Split", "fields": [
                "split.test_size",
                "split.val_ratio",
                "split.random_state",
            ]},
            {"title": "Augmentation", "fields": [
                "augmentation.enabled",
                "augmentation.seed",
            ]},
            {"title": "Sensor effects", "fields": [
                {"gate": "augmentation.sensor_dropout_enabled", "fields": ["augmentation.sensor_dropout_probability"]},
                {"gate": "augmentation.spurious_activation_enabled", "fields": ["augmentation.spurious_activation_probability", "augmentation.spurious_activation_max_light"]},
                {"gate": "augmentation.light_noise_enabled", "fields": ["augmentation.light_noise_sigma", {"path": "augmentation.light_noise_mode", "widget": CH_NOISE}]},
                {"gate": "augmentation.photon_thinning_enabled", "fields": ["augmentation.photon_thinning_survival"]},
                {"gate": "augmentation.gain_jitter_enabled", "fields": ["augmentation.gain_jitter_sigma"]},
            ]},
        ],
        "optimizer": [
            {"title": "Learning rates", "fields": [
                "optimizer.learning_rate_regression_head",
                "optimizer.learning_rate_pool",
                "optimizer.learning_rate_encoder",
                {"path": "optimizer.reference_batch_size", "widget": NUM_BATCH},
            ]},
            {"title": "Weight decay", "fields": [
                "optimizer.weight_decay_regression_head",
                "optimizer.weight_decay_pool",
                "optimizer.weight_decay_encoder",
            ]},
            {"title": "AdamW", "fields": [
                "optimizer.betas",
                "optimizer.eps",
            ]},
        ],
        "schedule": [
            {"title": "Scheduler", "fields": [
                {"path": "scheduler.type", "widget": CH_SCHEDULER},
                "scheduler.eta_min",
                "scheduler.power",
                "scheduler.step_size",
                "scheduler.gamma",
            ]},
            {"title": "Plateau", "fields": [
                {"path": "scheduler.mode", "widget": CH_PLATEAU},
                "scheduler.factor",
                "scheduler.patience",
                "scheduler.threshold",
                "scheduler.cooldown",
                "scheduler.min_lr",
            ]},
            {"title": "Warmup", "fields": [
                {"gate": "warmup.warmup_enabled", "fields": [
                    "warmup.warmup_steps",
                    "warmup.warmup_start_factor",
                    {"path": "warmup.warmup_mode", "widget": CH_WARMUP},
                    "warmup.warmup_poly_power",
                ]},
            ]},
            {"title": "Early stopping", "fields": [
                {"path": "early_stopping.patience", "widget": NUM_PATIENCE},
                "early_stopping.min_delta",
                "early_stopping.restore_best",
            ]},
        ],
        "gradients": [
            {"title": "Clipping", "fields": [
                {"path": "gradient_clipper.clip_mode", "widget": CH_CLIP},
                "gradient_clipper.max_grad_norm",
                "gradient_clipper.clip_epsilon",
            ]},
            {"title": "Adaptive window", "fields": [
                "gradient_clipper.adaptive_window",
                "gradient_clipper.adaptive_percentile",
                "gradient_clipper.adaptive_mean_std_k",
            ]},
            {"title": "Logging", "fields": [
                "gradient_clipper.log_histogram_freq",
            ]},
        ],
        "loss": [
            {"title": "Data term", "fields": [
                {"path": "loss.data_term", "widget": CH_DATA_TERM},
                "loss.huber_delta",
                "loss.data_weight",
            ]},
            {"title": "Geometry", "fields": [
                "loss.euclidean_weight",
            ]},
            {"title": "Light falloff", "fields": [
                "loss.light_falloff_weight",
                "loss.light_falloff_epsilon",
            ]},
        ],
        "loop": [
            {"title": "Steps", "fields": [
                "loop.gradient_accumulation_steps",
                "loop.validation_frequency",
                "loop.use_amp",
            ]},
            {"title": "Data feed", "fields": [
                {"path": "loop.num_workers", "widget": NUM_WORKERS},
                "loop.prefetch_factor",
                "loop.pin_memory",
                "loop.persistent_workers",
            ]},
            {"title": "Weight averaging", "fields": [
                {"gate": "loop.use_ema", "fields": ["loop.ema_decay"]},
            ]},
            {"title": "Robustness", "fields": [
                "loop.resume",
                "loop.abort_on_nonfinite_loss",
                "loop.log_debug",
                "loop.tuning_mode",
            ]},
        ],
        "resources": [
            {"title": "Allocator cache", "fields": [
                "memory.clear_cache_every_n_steps",
                "memory.clear_cache_after_eval",
                "memory.clear_cache_after_epoch",
            ]},
            {"title": "VRAM reservation", "fields": [
                {"gate": "memory.reserve_vram", "fields": ["memory.vram_keep_free_gb"]},
            ]},
            {"title": "Resource monitor", "fields": [
                {"gate": "resources.enabled", "fields": [
                    "resources.poll_interval_sec",
                    "resources.log_to_tensorboard",
                    "resources.warn_ram_pct",
                    "resources.warn_vram_pct",
                    "resources.warn_swap_pct",
                    "resources.warn_shm_pct",
                    "resources.warn_cooldown_sec",
                ]},
            ]},
        ],
        "preflight": [
            {"title": "Overfit gate", "fields": [
                {"gate": "overfit_check.enabled", "fields": [
                    "overfit_check.n_examples",
                    "overfit_check.max_steps",
                    "overfit_check.steps_per_epoch",
                    "overfit_check.pass_loss_ratio",
                    "overfit_check.stop_threshold",
                ]},
            ]},
            {"title": "Batch-size finder", "fields": [
                {"gate": "pretrain.find_batch_size", "fields": [
                    "pretrain.vram_budget_gb",
                    "pretrain.max_batch",
                    "pretrain.measure_steps",
                ]},
            ]},
            {"title": "Loader tuner", "fields": [
                {"gate": "pretrain.tune_loader", "fields": [
                    "pretrain.worker_counts",
                    "pretrain.prefetch_factors",
                    "pretrain.warmup_batches",
                    "pretrain.timed_batches",
                    "pretrain.data_wait_target",
                ]},
            ]},
            {"title": "Probe seed", "fields": [
                "pretrain.seed",
            ]},
        ],
        "output": [
            {"title": "Paths", "fields": [
                "io.tuner_base_dir",
                "io.checkpoint_name",
            ]},
        ],
    }

    TRAINING_SECTIONS = [
        {"key": "data", "title": "Dataset", "panels": [
            {"kind": "fields", "title": "Dataset", "template": "dataset", "at": "dataset"},
        ]},
        {"key": "optimisation", "title": "Optimisation", "panels": [
            {"kind": "fields", "title": "Optimizer",  "template": "optimizer", "at": "training"},
            {"kind": "fields", "title": "Schedule",   "template": "schedule",  "at": "training"},
            {"kind": "fields", "title": "Gradients",  "template": "gradients", "at": "training"},
        ]},
        {"key": "loss", "title": "Loss", "panels": [
            {"kind": "fields", "title": "Loss terms", "template": "loss", "at": "training"},
        ]},
        {"key": "runtime", "title": "Runtime", "panels": [
            {"kind": "fields", "title": "Training loop",       "template": "loop",      "at": "training"},
            {"kind": "fields", "title": "Memory and monitors", "template": "resources", "at": "training"},
        ]},
        {"key": "preflight", "title": "Preflight", "panels": [
            {"kind": "fields", "title": "Preflight checks", "template": "preflight", "at": "training"},
        ]},
        {"key": "output", "title": "Output", "panels": [
            {"kind": "fields", "title": "Run output", "template": "output", "at": "training"},
        ]},
    ]

    UNIVERSAL_ESSENTIALS = [
        {"path": "dataset.data.parquet_store_dir"},
        {"path": "training.io.log_base_dir"},
        {"path": "training.loop.device", "widget": CH_DEVICE},
        {"path": "training.loop.epochs", "widget": NUM_EPOCHS},
        {"path": "training.loop.batch_size", "widget": NUM_BATCH},
    ]

    LAYOUTS = {
        "train": {
            "essentials": [
                "run_name",
                {"path": "seed", "widget": NUM_SEED},
                "tensorboard",
                "infer_after",
            ] + UNIVERSAL_ESSENTIALS,
            "sections": [
                {"key": "model", "title": "Model", "panels": [
                    {"kind": "special", "panel": "model_card", "fields": ["model_name"]},
                    {"kind": "fields", "title": "Architecture", "groups": [
                        {"title": "Overrides", "fields": ["model_overrides"]},
                    ]},
                ]},
            ] + TRAINING_SECTIONS,
        },
        "tune": {
            "essentials": [
                {"path": "tuning.n_trials", "widget": NUM_TRIALS},
                {"path": "tuning.epochs", "widget": NUM_EPOCHS},
                {"path": "tuning.batch_size", "widget": NUM_BATCH},
                {"path": "dataset.data.parquet_store_dir"},
                {"path": "training.loop.device", "widget": CH_DEVICE},
            ],
            "sections": [
                {"key": "model", "title": "Model", "panels": [
                    {"kind": "special", "panel": "model_card", "fields": ["model_name"]},
                ]},
                {"key": "search", "title": "Search", "panels": [
                    {"kind": "fields", "title": "Optuna study", "groups": [
                        {"title": "Sampler", "fields": ["tuning.startup_trials", "tuning.warmup_trials", "tuning.warmup_steps", "tuning.random_state"]},
                        {"title": "Trial baseline", "fields": [
                            {"path": "training.loop.epochs", "widget": NUM_EPOCHS},
                            {"path": "training.loop.batch_size", "widget": NUM_BATCH},
                            "training.io.log_base_dir",
                        ]},
                    ]},
                ]},
            ] + TRAINING_SECTIONS,
        },
        "cross_validate": {
            "essentials": [
                "run_name",
                {"path": "seed", "widget": NUM_SEED},
                {"path": "cross_validation.n_folds", "widget": NUM_FOLDS},
            ] + UNIVERSAL_ESSENTIALS,
            "sections": [
                {"key": "model", "title": "Model", "panels": [
                    {"kind": "special", "panel": "model_card", "fields": ["model_name"]},
                    {"kind": "fields", "title": "Architecture", "groups": [
                        {"title": "Overrides", "fields": ["model_overrides"]},
                    ]},
                ]},
                {"key": "folds", "title": "Folds", "panels": [
                    {"kind": "fields", "title": "Fold plan", "groups": [
                        {"title": "Partitioning", "fields": ["cross_validation.validation_fraction", "cross_validation.stratified", "cross_validation.shuffle", "cross_validation.random_state"]},
                    ]},
                ]},
            ] + TRAINING_SECTIONS,
        },
        "benchmark": {
            "essentials": [
                "run_name",
                {"path": "seed", "widget": NUM_SEED},
                "resume",
                {"path": "dataset.data.parquet_store_dir"},
                {"path": "training.io.log_base_dir"},
                {"path": "training.loop.device", "widget": CH_DEVICE},
            ],
            "sections": [
                {"key": "zoo", "title": "Model zoo", "panels": [
                    {"kind": "special", "panel": "model_toggle", "fields": ["skip_models"]},
                    {"kind": "fields", "title": "Capacity matching", "groups": [
                        {"title": "Size match", "fields": [
                            {"gate": "size_match.enabled", "fields": [
                                "size_match.reference_model",
                                "size_match.scaled_attribute",
                                "size_match.tolerance",
                                "size_match.max_iterations",
                                "size_match.minimum_width",
                                "size_match.maximum_width",
                            ]},
                        ]},
                    ]},
                ]},
                {"key": "gates", "title": "Screening", "panels": [
                    {"kind": "fields", "title": "Screening stages", "groups": [
                        {"title": "Overfit gate", "fields": [
                            {"gate": "overfit.enabled", "fields": [
                                "overfit.sample_count",
                                "overfit.epochs",
                                "overfit.batch_size",
                                "overfit.stop_threshold",
                                "overfit.require_convergence",
                                "overfit.abort_on_fail",
                            ]},
                        ]},
                        {"title": "Max batch probe", "fields": [
                            {"gate": "max_batch.enabled", "fields": [
                                "max_batch.vram_budget_gb",
                                "max_batch.maximum_batch",
                                "max_batch.measure_steps",
                                "max_batch.sample_count",
                            ]},
                        ]},
                    ]},
                ]},
                {"key": "comparison", "title": "Comparison", "panels": [
                    {"kind": "fields", "title": "Benchmark training", "groups": [
                        {"title": "Trial budget", "fields": [
                            {"path": "benchmark_training.epochs", "widget": NUM_EPOCHS},
                            "benchmark_training.use_measured_batch",
                            {"path": "training.loop.epochs", "widget": NUM_EPOCHS},
                            {"path": "training.loop.batch_size", "widget": NUM_BATCH},
                        ]},
                        {"title": "Ranking", "fields": ["comparison.rank_metric"]},
                    ]},
                ]},
            ] + TRAINING_SECTIONS,
        },
        "baseline": {
            "essentials": [
                "run_name",
                {"path": "seed", "widget": NUM_SEED},
                {"path": "models", "widget": MULTI_BASELINES},
                {"path": "baseline.search.n_trials", "widget": NUM_TRIALS},
                {"path": "dataset.data.parquet_store_dir"},
                {"path": "baseline.io.log_base_dir"},
            ],
            "sections": [
                {"key": "search", "title": "Search", "panels": [
                    {"kind": "fields", "title": "Optuna study", "groups": [
                        {"title": "Sampler", "fields": ["baseline.search.startup_trials", "baseline.search.random_state"]},
                        {"title": "Boosting budget", "fields": [
                            {"path": "baseline.booster.max_estimators", "widget": NUM_TREES},
                            {"path": "baseline.booster.early_stopping_rounds", "widget": NUM_ROUNDS},
                            {"path": "baseline.booster.thread_count", "widget": NUM_CPU},
                            {"path": "baseline.booster.device", "widget": CH_DEVICE},
                        ]},
                    ]},
                ]},
                {"key": "features", "title": "Features", "panels": [
                    {"kind": "fields", "title": "Summary features", "groups": [
                        {"title": "Aggregates", "fields": [
                            {"path": "baseline.features.top_channel_count", "widget": NUM_TOPCHAN},
                            "baseline.features.quantiles",
                        ]},
                    ]},
                ]},
                {"key": "data", "title": "Dataset", "panels": [
                    {"kind": "fields", "title": "Dataset", "template": "dataset", "at": "dataset"},
                ]},
            ],
        },
        "infer": {
            "sections": [
                {"key": "inference", "title": "Inference", "panels": [
                    {"kind": "fields", "title": "Inference", "groups": [
                        {"title": "Run", "fields": [{"path": "run_directory", "widget": PICK_RUN}]},
                        {"title": "Evaluation", "fields": [
                            {"path": "device", "widget": CH_DEVICE},
                            {"path": "splits", "widget": MULTI_SPLITS},
                            {"path": "batch_size", "widget": NUM_BATCH},
                        ]},
                    ]},
                ]},
            ],
        },
        "explain": {
            "essentials": [
                {"path": "run_directory", "widget": PICK_RUN},
                {"path": "device", "widget": CH_DEVICE},
                {"path": "split", "widget": CH_SPLIT},
                {"path": "batch_size", "widget": NUM_BATCH},
                "run_name",
                "max_events",
            ],
            "sections": [
                {"key": "methods", "title": "Attribution", "panels": [
                    {"kind": "fields", "title": "Attribution methods", "groups": [
                        {"title": "Permutation", "fields": [
                            {"gate": "permutation", "fields": ["permutation_repeats", "permutation_seed", "grouped"]},
                        ]},
                        {"title": "Occlusion and saliency", "fields": [
                            {"gate": "occlusion", "fields": []},
                            {"gate": "gradient", "fields": []},
                        ]},
                        {"title": "Expected gradients", "fields": [
                            {"gate": "expected_gradients", "fields": ["eg_samples", "eg_events", "eg_seed"]},
                        ]},
                        {"title": "SHAP", "fields": [
                            {"gate": "shap", "fields": ["shap_nsamples", "shap_events", "shap_seed"]},
                        ]},
                    ]},
                ]},
                {"key": "report", "title": "Report", "panels": [
                    {"kind": "fields", "title": "Report", "groups": [
                        {"title": "Ranking", "fields": ["primary_metric", "top_k"]},
                    ]},
                ]},
            ],
        },
        "build_parquet_store": {
            "sections": [
                {"key": "store", "title": "Store", "panels": [
                    {"kind": "fields", "title": "Parquet store", "groups": [
                        {"title": "Paths", "fields": [
                            "raw_input_dir",
                            "output_dir",
                            {"path": "worker_count", "widget": NUM_CPU},
                        ]},
                        {"title": "Hot channels", "fields": [
                            {"gate": "hot_channels.enabled", "fields": [
                                "hot_channels.active_fraction",
                                "hot_channels.median_factor",
                                "hot_channels.neighbor_count",
                                "hot_channels.min_events",
                            ]},
                        ]},
                    ]},
                ]},
            ],
        },
    }

    def _join(self, prefix: str, name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    def _field_entry(self, item, prefix: str, widgets: dict) -> dict:
        if isinstance(item, str):
            return {"path": self._join(prefix, item)}

        if "gate" in item:
            return {"gate": self._join(prefix, item["gate"]), "fields": [self._field_entry(sub, prefix, widgets) for sub in item["fields"]]}

        path = self._join(prefix, item["path"])
        if "widget" in item:
            widgets[path] = item["widget"]
        return {"path": path}

    def _expand_groups(self, groups, prefix: str, widgets: dict) -> list:
        return [{"title": group.get("title"), "fields": [self._field_entry(item, prefix, widgets) for item in group["fields"]]} for group in groups]

    def _expand_panel(self, panel, widgets: dict) -> dict:
        if panel["kind"] == "special":
            return {"kind": "special", "panel": panel["panel"], "fields": list(panel["fields"])}

        if panel["kind"] == "hidden":
            return {"kind": "hidden", "fields": list(panel["fields"])}

        groups = self.TEMPLATES[panel["template"]] if "template" in panel else panel["groups"]
        return {"kind": "fields", "title": panel.get("title"), "groups": self._expand_groups(groups, panel.get("at", ""), widgets)}

    def _expand(self, key: str) -> dict:
        spec    = self.LAYOUTS[key]
        widgets = {}

        essentials = [self._field_entry(item, "", widgets) for item in spec.get("essentials", [])]
        sections   = [{"key": section["key"], "title": section["title"], "panels": [self._expand_panel(panel, widgets) for panel in section["panels"]]} for section in spec["sections"]]

        return {"essentials": essentials, "sections": sections, "widgets": widgets}

    def _entry_claims(self, entry, claimed: list) -> None:
        if "gate" in entry:
            claimed.append(entry["gate"])
            for sub in entry["fields"]:
                self._entry_claims(sub, claimed)
            return

        claimed.append(entry["path"])

    def _claims(self, layout: dict) -> list:
        claimed = []

        for entry in layout["essentials"]:
            self._entry_claims(entry, claimed)

        for section in layout["sections"]:
            for panel in section["panels"]:
                if panel["kind"] in ("special", "hidden"):
                    claimed.extend(panel["fields"])
                    continue

                for group in panel["groups"]:
                    for entry in group["fields"]:
                        self._entry_claims(entry, claimed)

        return claimed

    def _validate(self, key: str, layout: dict, leaves: list) -> None:
        paths   = [leaf["path"] for leaf in leaves]
        known   = set(paths)
        claimed = self._claims(layout)

        seen       = set()
        duplicates = sorted({path for path in claimed if path in seen or seen.add(path)})
        unknown    = sorted(set(claimed) - known)
        unclaimed  = [path for path in paths if path not in seen]

        problems = []
        if unknown:
            problems.append(f"layout for {key} names unknown fields: {', '.join(unknown)}")
        if duplicates:
            problems.append(f"layout for {key} claims fields twice: {', '.join(duplicates)}")
        if unclaimed:
            problems.append(f"layout for {key} leaves fields unclaimed: {', '.join(unclaimed)}")

        if problems:
            raise LayoutError("\n".join(problems))

    def build(self, key: str, leaves: list) -> dict:
        if key not in self.LAYOUTS:
            raise LayoutError(f"no launch layout declared for {key}")

        layout = copy.deepcopy(self._expand(key))
        self._validate(key, layout, leaves)

        layout["mode"] = "single" if len(layout["sections"]) == 1 and not layout["essentials"] else "sections"
        return layout
