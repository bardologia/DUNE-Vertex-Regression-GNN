from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServerSettings:
    host             : str = "127.0.0.1"
    port             : int = 8765
    protocol_version : str = "HTTP/1.1"


@dataclass(frozen=True)
class ApiRoutes:
    index         : str = "/"
    static_prefix : str = "/static/"

    system        : str = "/api/system"
    system_status : str = f"{system}/status"
    system_gpu    : str = f"{system}/gpu"
    interpreters  : str = "/api/interpreters"
    models        : str = "/api/models"

    scripts        : str = "/api/scripts"
    scripts_prefix : str = f"{scripts}/"
    config_suffix  : str = "/config"
    source_suffix  : str = "/source"

    runs      : str = "/api/runs"
    runs_file : str = f"{runs}/file"

    launch           : str = "/api/launch"
    processes        : str = "/api/processes"
    processes_prefix : str = f"{processes}/"
    stream_suffix    : str = "/stream"
    kill_suffix      : str = "/kill"

    tensorboard        : str = "/api/tensorboard"
    tensorboard_prefix : str = f"{tensorboard}/"
    tensorboard_start  : str = f"{tensorboard}/start"
    stop_suffix        : str = "/stop"

    events         : str = "/api/events"
    events_sources : str = f"{events}/sources"
    events_status  : str = f"{events}/status"
    events_list    : str = f"{events}/list"
    events_detail  : str = f"{events}/detail"
    events_load    : str = f"{events}/load"

    preprocess         : str = "/api/preprocess"
    preprocess_status  : str = f"{preprocess}/status"
    preprocess_events  : str = f"{preprocess}/events"
    preprocess_load    : str = f"{preprocess}/load"
    preprocess_preview : str = f"{preprocess}/preview"


@dataclass(frozen=True)
class HttpSettings:
    json_content_type      : str = "application/json"
    binary_content_type    : str = "application/octet-stream"
    stream_content_type    : str = "text/event-stream"
    no_cache_control       : str = "no-cache"
    run_file_cache_control : str = "max-age=300"
    stream_connection      : str = "keep-alive"
    stream_allowed_origin  : str = "*"
    stream_keepalive_s     : int = 15


@dataclass(frozen=True)
class LaunchSettings:
    preferred_environment : str   = "conda:Dune"
    tensorboard_flags     : tuple = ("1", "true", "yes", "on")

    introspection_timeout_s   : int = 180
    introspection_error_lines : int = 4


@dataclass(frozen=True)
class ProcessSettings:
    override_name_pattern : str   = r"^[A-Za-z_][A-Za-z0-9_.]*$"
    finished_history      : int   = 50
    pipe_read_bytes       : int   = 4096
    stop_grace_s          : float = 6.0


@dataclass(frozen=True)
class TensorboardSettings:
    host              : str   = "127.0.0.1"
    path_prefix       : str   = "/tb/"
    id_length         : int   = 8
    reload_interval   : str   = "30"
    startup_timeout_s : float = 90.0
    probe_interval_s  : float = 0.5
    probe_timeout_s   : int   = 2
    shutdown_grace_s  : float = 5.0
    log_tail_chars    : int   = 500

    proxy_timeout_s        : int   = 60
    proxy_accept_encoding  : str   = "identity"
    proxy_request_headers  : tuple = ("Content-Type", "Accept", "X-XSRF-Protected")
    proxy_response_headers : tuple = ("Content-Type", "Content-Encoding", "Cache-Control", "Location")


@dataclass(frozen=True)
class SystemMonitorSettings:
    process_limit       : int   = 12
    gpu_poll_interval_s : float = 4.0


@dataclass(frozen=True)
class ResultsBrowsing:
    max_depth          : int   = 6
    nested_run_depth   : int   = 2
    best_metric_splits : tuple = ("test", "val", "train")
    best_metric_name   : str   = "euclidean_mean"
    coordinate_names   : tuple = ("x", "y", "z")
    error_metric_names : tuple = ("mae", "rmse")


@dataclass(frozen=True)
class EventExplorerSettings:
    splits           : tuple = ("test", "val", "train")
    datasets         : tuple = ({"name": "raw", "label": "raw events"}, {"name": "augmented", "label": "octant-augmented"})
    error_percentile : float = 90.0


@dataclass(frozen=True)
class ScriptRegistry:
    entry_configs : tuple = (
        ("train",               {"entry_config_module": "configuration.entry.general",          "entry_config_class": "TrainEntryConfig"}),
        ("tune",                {"entry_config_module": "configuration.entry.general",          "entry_config_class": "TuneEntryConfig"}),
        ("cross_validate",      {"entry_config_module": "configuration.entry.general",          "entry_config_class": "CrossValidationEntryConfig"}),
        ("benchmark",           {"entry_config_module": "configuration.benchmark.general",      "entry_config_class": "BenchmarkConfig"}),
        ("baseline",            {"entry_config_module": "configuration.entry.general",          "entry_config_class": "BaselineEntryConfig"}),
        ("infer",               {"entry_config_module": "configuration.entry.general",          "entry_config_class": "InferenceEntryConfig"}),
        ("explain",             {"entry_config_module": "configuration.explainability.general", "entry_config_class": "FeatureImportanceConfig"}),
        ("build_parquet_store", {"entry_config_module": "configuration.entry.general",          "entry_config_class": "DatasetEntryConfig"}),
    )

    metadata : tuple = (
        ("train",               {"label": "Train",               "category": "Training", "purpose": "Train one GNN model end to end, logging checkpoints and metrics to runs/."}),
        ("tune",                {"label": "Tune",                "category": "Training", "purpose": "Run the Optuna hyperparameter search over the model and training configuration."}),
        ("cross_validate",      {"label": "Cross-Validate",      "category": "Training", "purpose": "Run stratified K-fold cross-validation, aggregating held-out test metrics across folds."}),
        ("benchmark",           {"label": "Benchmark",           "category": "Training", "purpose": "Benchmark the whole model zoo: capacity matching, overfit gate, max-batch probe, training, and a ranked comparison report."}),
        ("baseline",            {"label": "Baseline",            "category": "Training", "purpose": "Tune and train the LightGBM and XGBoost baselines on tabular summary features of the same splits, writing leaderboard-comparable runs."}),
        ("build_parquet_store", {"label": "Build Parquet Store", "category": "Data",     "purpose": "Materialise the raw events into the Parquet store: sensor positions from the documented module mapping of the raw bins, face totals dropped, and octant expansion."}),
        ("infer",               {"label": "Infer",               "category": "Analysis", "purpose": "Run the full inference analysis (metrics, plots, animations, report) on a training run."}),
        ("explain",             {"label": "Explain",             "category": "Analysis", "purpose": "Attribute the graph node and edge features of a trained run via permutation, occlusion, and gradient saliency, producing plots and a report."}),
    )

    order : tuple = ("build_parquet_store", "train", "infer", "explain", "tune", "cross_validate", "benchmark", "baseline")


@dataclass(frozen=True)
class ModelFamilies:
    families : tuple = (
        {
            "family": "Attention and transformer",
            "models": [
                {"key": "gps",           "name": "GPS",               "blurb": "Local message passing plus global self-attention", "recommended": True},
                {"key": "gps_lite",      "name": "GPS Lite",          "blurb": "Narrow GPS variant for fast iteration"},
                {"key": "gps_cascade",   "name": "GPS Cascade",       "blurb": "GPS trunk with a cascade regression head"},
                {"key": "gatv2",         "name": "GATv2",             "blurb": "Dynamic attention over neighbours"},
                {"key": "gatv2_cascade", "name": "GATv2 Cascade",     "blurb": "GATv2 trunk with a cascade regression head"},
                {"key": "transformer",   "name": "Graph Transformer", "blurb": "Full transformer convolution with edge features"},
            ],
        },
        {
            "family": "Message passing",
            "models": [
                {"key": "gcn",          "name": "GCN",          "blurb": "Spectral graph convolution baseline"},
                {"key": "graphsage",    "name": "GraphSAGE",    "blurb": "Sampled neighbour aggregation"},
                {"key": "gine",         "name": "GINE",         "blurb": "Isomorphism network with edge features"},
                {"key": "gine_cascade", "name": "GINE Cascade", "blurb": "GINE trunk with a cascade regression head"},
                {"key": "general_conv", "name": "GeneralConv",  "blurb": "Configurable general-purpose convolution"},
                {"key": "res_gated",    "name": "ResGated",     "blurb": "Residual gated graph convolution"},
            ],
        },
        {
            "family": "Aggregation and dynamic graphs",
            "models": [
                {"key": "pna",      "name": "PNA",      "blurb": "Principal neighbourhood aggregation, multi-aggregator"},
                {"key": "edgeconv", "name": "EdgeConv", "blurb": "Dynamic k-NN graph rebuilt in feature space"},
            ],
        },
    )


@dataclass(frozen=True)
class LaunchForms:
    pick_run : tuple = (("kind", "run"),)

    ch_device    : tuple = (("kind", "choice"), ("options", ["cuda", "cpu"]))
    ch_data_term : tuple = (("kind", "choice"), ("options", ["mse", "huber"]))
    ch_plateau   : tuple = (("kind", "choice"), ("options", ["min", "max"]))
    ch_noise     : tuple = (("kind", "choice"), ("options", ["multiplicative", "additive"]))
    ch_split     : tuple = (("kind", "choice"), ("options", ["test", "val", "train"]))
    ch_scheduler : tuple = (("kind", "choice"), ("options", ["reduce_on_plateau", "cosine_annealing", "linear", "polynomial", "exponential", "step", "constant"]))
    ch_warmup    : tuple = (("kind", "choice"), ("options", ["linear", "cosine", "exponential", "polynomial"]))
    ch_clip      : tuple = (("kind", "choice"), ("options", ["fixed", "adaptive_percentile", "adaptive_mean_std", "disabled"]))
    ch_frame     : tuple = (("kind", "choice"), ("options", ["isotropic", "axiswise"]))

    multi_splits : tuple = (("kind", "multi"), ("empty", "select at least one split"), ("choices", [
        {"value": "train", "label": "train"},
        {"value": "val",   "label": "val"},
        {"value": "test",  "label": "test"},
    ]))

    multi_baselines : tuple = (("kind", "multi"), ("empty", "select at least one model"), ("choices", [
        {"value": "lightgbm", "label": "LightGBM"},
        {"value": "xgboost",  "label": "XGBoost"},
    ]))

    num_epochs   : tuple = (("kind", "number"), ("int", True), ("min", 1), ("max", 1000), ("presets", [25, 50, 100, 200, 300]))
    num_batch    : tuple = (("kind", "number"), ("int", True), ("min", 1), ("max", 4096), ("presets", [8, 16, 32, 64, 128, 256]))
    num_workers  : tuple = (("kind", "number"), ("int", True), ("min", 0), ("max", 64), ("presets", [0, 2, 4, 8, 16]))
    num_seed     : tuple = (("kind", "number"), ("int", True), ("min", 0), ("max", 9999), ("presets", [0, 1, 42, 1234]))
    num_patience : tuple = (("kind", "number"), ("int", True), ("min", 1), ("max", 200), ("presets", [5, 10, 15, 25, 50]))
    num_neighbor : tuple = (("kind", "number"), ("int", True), ("min", 2), ("max", 64), ("presets", [4, 8, 16, 32]))
    num_rbf      : tuple = (("kind", "number"), ("int", True), ("min", 4), ("max", 64), ("presets", [8, 16, 32]))
    num_cpu      : tuple = (("kind", "number"), ("int", True), ("min", 1), ("max", 64), ("presets", [4, 8, 10, 16]))
    num_trials   : tuple = (("kind", "number"), ("int", True), ("min", 1), ("max", 1000), ("presets", [20, 50, 100, 200]))
    num_folds    : tuple = (("kind", "number"), ("int", True), ("min", 2), ("max", 20), ("presets", [3, 5, 10]))
    num_trees    : tuple = (("kind", "number"), ("int", True), ("min", 100), ("max", 20000), ("presets", [1000, 2000, 3000, 5000]))
    num_rounds   : tuple = (("kind", "number"), ("int", True), ("min", 10), ("max", 1000), ("presets", [50, 100, 200]))
    num_topchan  : tuple = (("kind", "number"), ("int", True), ("min", 1), ("max", 32), ("presets", [3, 5, 8, 12]))

    templates : tuple = (
        ("dataset", [
            {"title": "Store", "fields": [
                "data.raw_input_dir",
                "data.build_store",
                {"path": "data.store_worker_count", "widget": num_cpu},
                "data.stats_sample_size",
                {"path": "data.target_frame", "widget": ch_frame},
            ]},
            {"title": "Sampling", "fields": [
                "data.augment_octants",
                "data.subset_fraction",
                "data.coordinate_columns",
                "data.position_columns",
                "data.detector_half_extent",
            ]},
            {"title": "Graph", "fields": [
                {"path": "graph.k_neighbors", "widget": num_neighbor},
                "graph.bidirectional",
                "graph.active_only",
                "graph.max_active_nodes",
                "graph.global_node",
                {"path": "graph.edge_rbf_count", "widget": num_rbf},
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
                {"gate": "augmentation.light_noise_enabled", "fields": ["augmentation.light_noise_sigma", {"path": "augmentation.light_noise_mode", "widget": ch_noise}]},
                {"gate": "augmentation.photon_thinning_enabled", "fields": ["augmentation.photon_thinning_survival"]},
                {"gate": "augmentation.gain_jitter_enabled", "fields": ["augmentation.gain_jitter_sigma"]},
            ]},
        ]),
        ("optimizer", [
            {"title": "Learning rates", "fields": [
                "optimizer.learning_rate_regression_head",
                "optimizer.learning_rate_pool",
                "optimizer.learning_rate_encoder",
                {"path": "optimizer.reference_batch_size", "widget": num_batch},
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
        ]),
        ("schedule", [
            {"title": "Scheduler", "fields": [
                {"path": "scheduler.type", "widget": ch_scheduler},
                "scheduler.eta_min",
                "scheduler.power",
                "scheduler.step_size",
                "scheduler.gamma",
            ]},
            {"title": "Plateau", "fields": [
                {"path": "scheduler.mode", "widget": ch_plateau},
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
                    {"path": "warmup.warmup_mode", "widget": ch_warmup},
                    "warmup.warmup_poly_power",
                ]},
            ]},
            {"title": "Early stopping", "fields": [
                {"path": "early_stopping.patience", "widget": num_patience},
                "early_stopping.min_delta",
                "early_stopping.restore_best",
            ]},
        ]),
        ("gradients", [
            {"title": "Clipping", "fields": [
                {"path": "gradient_clipper.clip_mode", "widget": ch_clip},
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
        ]),
        ("loss", [
            {"title": "Data term", "fields": [
                {"path": "loss.data_term", "widget": ch_data_term},
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
        ]),
        ("loop", [
            {"title": "Steps", "fields": [
                "loop.gradient_accumulation_steps",
                "loop.validation_frequency",
                "loop.use_amp",
            ]},
            {"title": "Data feed", "fields": [
                {"path": "loop.num_workers", "widget": num_workers},
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
        ]),
        ("resources", [
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
        ]),
        ("preflight", [
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
        ]),
        ("output", [
            {"title": "Paths", "fields": [
                "io.tuner_base_dir",
                "io.checkpoint_name",
            ]},
        ]),
    )

    training_sections : tuple = (
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
    )

    universal_essentials : tuple = (
        {"path": "dataset.data.parquet_store_dir"},
        {"path": "training.io.log_base_dir"},
        {"path": "training.loop.device", "widget": ch_device},
        {"path": "training.loop.epochs", "widget": num_epochs},
        {"path": "training.loop.batch_size", "widget": num_batch},
    )

    layouts : tuple = (
        ("train", {
            "essentials": [
                "run_name",
                {"path": "seed", "widget": num_seed},
                "tensorboard",
                "infer_after",
                *universal_essentials,
            ],
            "sections": [
                {"key": "model", "title": "Model", "panels": [
                    {"kind": "special", "panel": "model_card", "fields": ["model_name"]},
                    {"kind": "fields", "title": "Architecture", "groups": [
                        {"title": "Overrides", "fields": ["model_overrides"]},
                    ]},
                ]},
                *training_sections,
            ],
        }),
        ("tune", {
            "essentials": [
                {"path": "tuning.n_trials", "widget": num_trials},
                {"path": "tuning.epochs", "widget": num_epochs},
                {"path": "tuning.batch_size", "widget": num_batch},
                {"path": "dataset.data.parquet_store_dir"},
                {"path": "training.loop.device", "widget": ch_device},
            ],
            "sections": [
                {"key": "model", "title": "Model", "panels": [
                    {"kind": "special", "panel": "model_card", "fields": ["model_name"]},
                ]},
                {"key": "search", "title": "Search", "panels": [
                    {"kind": "fields", "title": "Optuna study", "groups": [
                        {"title": "Sampler", "fields": ["tuning.startup_trials", "tuning.warmup_trials", "tuning.warmup_steps", "tuning.random_state"]},
                        {"title": "Trial baseline", "fields": [
                            {"path": "training.loop.epochs", "widget": num_epochs},
                            {"path": "training.loop.batch_size", "widget": num_batch},
                            "training.io.log_base_dir",
                        ]},
                    ]},
                ]},
                *training_sections,
            ],
        }),
        ("cross_validate", {
            "essentials": [
                "run_name",
                {"path": "seed", "widget": num_seed},
                {"path": "cross_validation.n_folds", "widget": num_folds},
                *universal_essentials,
            ],
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
                *training_sections,
            ],
        }),
        ("benchmark", {
            "essentials": [
                "run_name",
                {"path": "seed", "widget": num_seed},
                "resume",
                {"path": "dataset.data.parquet_store_dir"},
                {"path": "training.io.log_base_dir"},
                {"path": "training.loop.device", "widget": ch_device},
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
                            {"path": "benchmark_training.epochs", "widget": num_epochs},
                            "benchmark_training.use_measured_batch",
                            {"path": "training.loop.epochs", "widget": num_epochs},
                            {"path": "training.loop.batch_size", "widget": num_batch},
                        ]},
                        {"title": "Ranking", "fields": ["comparison.rank_metric"]},
                    ]},
                ]},
                *training_sections,
            ],
        }),
        ("baseline", {
            "essentials": [
                "run_name",
                {"path": "seed", "widget": num_seed},
                {"path": "models", "widget": multi_baselines},
                {"path": "baseline.search.n_trials", "widget": num_trials},
                {"path": "dataset.data.parquet_store_dir"},
                {"path": "baseline.io.log_base_dir"},
            ],
            "sections": [
                {"key": "search", "title": "Search", "panels": [
                    {"kind": "fields", "title": "Optuna study", "groups": [
                        {"title": "Sampler", "fields": ["baseline.search.startup_trials", "baseline.search.random_state"]},
                        {"title": "Boosting budget", "fields": [
                            {"path": "baseline.booster.max_estimators", "widget": num_trees},
                            {"path": "baseline.booster.early_stopping_rounds", "widget": num_rounds},
                            {"path": "baseline.booster.thread_count", "widget": num_cpu},
                            {"path": "baseline.booster.device", "widget": ch_device},
                        ]},
                    ]},
                ]},
                {"key": "features", "title": "Features", "panels": [
                    {"kind": "fields", "title": "Summary features", "groups": [
                        {"title": "Aggregates", "fields": [
                            {"path": "baseline.features.top_channel_count", "widget": num_topchan},
                            "baseline.features.quantiles",
                        ]},
                    ]},
                ]},
                {"key": "data", "title": "Dataset", "panels": [
                    {"kind": "fields", "title": "Dataset", "template": "dataset", "at": "dataset"},
                ]},
            ],
        }),
        ("infer", {
            "sections": [
                {"key": "inference", "title": "Inference", "panels": [
                    {"kind": "fields", "title": "Inference", "groups": [
                        {"title": "Run", "fields": [{"path": "run_directory", "widget": pick_run}]},
                        {"title": "Evaluation", "fields": [
                            {"path": "device", "widget": ch_device},
                            {"path": "splits", "widget": multi_splits},
                            {"path": "batch_size", "widget": num_batch},
                        ]},
                    ]},
                ]},
            ],
        }),
        ("explain", {
            "essentials": [
                {"path": "run_directory", "widget": pick_run},
                {"path": "device", "widget": ch_device},
                {"path": "split", "widget": ch_split},
                {"path": "batch_size", "widget": num_batch},
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
        }),
        ("build_parquet_store", {
            "sections": [
                {"key": "store", "title": "Store", "panels": [
                    {"kind": "fields", "title": "Parquet store", "groups": [
                        {"title": "Paths", "fields": [
                            "raw_input_dir",
                            "output_dir",
                            {"path": "worker_count", "widget": num_cpu},
                        ]},
                    ]},
                ]},
            ],
        }),
    )

