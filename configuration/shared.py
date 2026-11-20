from __future__ import annotations

from dataclasses import dataclass
from pathlib     import Path


@dataclass(frozen=True)
class IoNames:
    project_root : Path = Path(__file__).resolve().parents[1]

    data_dir           : str = "data"
    data_frames_dir    : str = "data_frames"
    parquet_dir        : str = "parquet"
    event_explorer_dir : str = "event_explorer"
    runs_dir           : str = "runs"
    tuning_dir         : str = "tuning"
    logs_dir           : str = "logs"

    default_log_name  : str = "experiment"
    log_suffix        : str = ".log"
    output_log_suffix : str = ".out"
    run_tag_format    : str = "%Y%m%d_%H%M%S"
    run_stamp_format  : str = "%Y%m%d-%H%M%S"
    json_indent       : int = 2
    shm_dir           : str = "/dev/shm"

    parquet_store_log  : str = "parquet_store"
    tune_log           : str = "tune_{model}"
    light_centroid_log : str = "light_centroid_check"
    throughput_figure  : str = "throughput_vs_batch_size.png"
    data_wait_figure   : str = "data_wait_vs_workers.png"
    gpu_util_figure    : str = "gpu_util_vs_throughput.png"
    feed_ratio_figure  : str = "feed_ratio_vs_workers.png"

    raw_event_glob      : str = "*.csv"
    geometry_table_file : str = "geometry.parquet"
    events_table_file   : str = "events.parquet"
    octants_table_file  : str = "octants.parquet"

    run_dir                  : str = "{model}_{run}"
    tensorboard_dir          : str = "tensorboard"
    metadata_dir             : str = "metadata"
    checkpoints_dir          : str = "checkpoints"
    analysis_dir             : str = "analysis"
    best_model_file          : str = "best_model.pt"
    trainer_state_file       : str = "last.pt"
    resolved_config_file     : str = "resolved_config.json"
    split_base_ids_file      : str = "split_base_ids.json"
    metrics_file             : str = "metrics.json"
    normalization_stats_file : str = "normalization_stats.json"
    store_geometry_file      : str = "store_geometry.json"
    completion_file          : str = "complete.json"
    model_summary_file       : str = "model_summary.md"

    overfit_check_dir     : str = "overfit_check"
    overfit_report_file   : str = "overfit_report.json"
    pretrain_dir          : str = "pretrain"
    pretrain_log          : str = "pretrain"
    pretrain_report_file  : str = "pretrain_report.json"
    probe_dir             : str = "probe"
    probe_checkpoint_file : str = "probe.pt"
    trial_dir             : str = "trial_{number}"
    best_trial_file       : str = "best_trial.json"

    npz_suffix   : str = ".npz"
    plots_dir    : str = "plots"
    results_file : str = "results.json"
    report_file  : str = "report.md"

    inference_log                 : str = "inference"
    analysis_report_file          : str = "analysis_report.md"
    predicted_vs_true_figure      : str = "predicted_vs_true_{coordinate}.png"
    residual_histogram_figure     : str = "residual_histogram_{coordinate}.png"
    error_vs_true_figure          : str = "error_vs_true_{coordinate}.png"
    euclidean_histogram_figure    : str = "euclidean_error_histogram.png"
    euclidean_cumulative_figure   : str = "euclidean_error_cumulative.png"
    error_volume_figure           : str = "error_three_dimensional.png"
    rotating_prediction_animation : str = "rotating_predicted_vs_true.gif"
    rotating_error_animation      : str = "rotating_error_volume.gif"
    event_explorer_cache_file     : str = "event_explorer.npz"
    raw_light_cache_file          : str = "raw_light.npz"
    dataset_explorer_log          : str = "dataset_explorer"
    raw_light_log                 : str = "raw_light"

    explain_log                  : str = "explain"
    explainability_dir           : str = "explainability"
    feature_importance_dir       : str = "feature_importance_{stamp}"
    named_feature_importance_dir : str = "feature_importance_{run}_{stamp}"
    shap_plots_dir               : str = "shap"
    node_importance_file         : str = "node_importance.csv"
    edge_importance_file         : str = "edge_importance.csv"
    permutation_figure           : str = "permutation_{part}.png"
    occlusion_figure             : str = "occlusion_{part}.png"
    gradient_figure              : str = "gradient_{domain}_{method}.png"
    expected_gradients_figure    : str = "expected_gradients_{domain}_{axis}.png"
    kernel_shap_figure           : str = "kernel_shap_{domain}_{axis}.png"
    shap_bar_figure              : str = "shap_bar_{domain}_{axis}.png"
    shap_beeswarm_figure         : str = "shap_beeswarm_{domain}_{axis}.png"

    baseline_log              : str = "baseline"
    baseline_run_dir          : str = "baseline_{family}_{run}"
    feature_importance_figure : str = "feature_importance.png"
    feature_importance_file   : str = "feature_importance.json"
    lightgbm_model_file       : str = "lightgbm_{axis}.txt"
    xgboost_model_file        : str = "xgboost_{axis}.ubj"

    benchmark_log          : str = "benchmark"
    benchmark_run_dir      : str = "benchmark_{run}"
    benchmark_pipeline_dir : str = "pipeline"
    benchmark_state_file   : str = "state.json"
    size_match_file        : str = "size_match.json"
    overfit_results_file   : str = "overfit_results.json"
    max_batch_file         : str = "max_batch.json"
    training_results_file  : str = "training_results.json"
    benchmark_report_file  : str = "benchmark_report.md"
    benchmark_summary_file : str = "benchmark_summary.json"
    benchmark_overfit_dir  : str = "overfit"
    benchmark_training_dir : str = "training"
    benchmark_training_run : str = "run"
    comparison_dir         : str = "comparison"

    cv_log          : str = "cv_{model}"
    cv_run_dir      : str = "cv_{model}_{run}"
    fold_run        : str = "fold_{index}"
    cv_metrics_file : str = "cross_validation_metrics.json"
    cv_report_file  : str = "cross_validation_report.md"

    sweep_log                      : str = "sweep"
    sweeps_dir                     : str = "sweeps"
    energy_efficiency_dir          : str = "energy_efficiency_{stamp}"
    named_energy_efficiency_dir    : str = "energy_efficiency_{run}_{stamp}"
    sweep_results_table            : str = "results.csv"
    heatmap_figure                 : str = "heatmap_{metric}.png"
    euclidean_vs_scale_figure      : str = "euclidean_mean_vs_scale.png"
    euclidean_vs_efficiency_figure : str = "euclidean_mean_vs_efficiency.png"


@dataclass(frozen=True)
class LogSettings:
    level       : str   = "INFO"
    level_names : tuple = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    file_format         : str = "[%(asctime)s] [%(levelname)s] %(message)s"
    timestamp_format    : str = "%Y-%m-%d %H:%M:%S"
    console_time_format : str = "[%H:%M:%S]"
    file_rule_character : str = "="
    file_rule_width     : int = 100

    console_color_system        : str   = "truecolor"
    console_theme               : tuple = (("section", "bold cyan"), ("subsection", "white"), ("key", "bold magenta"), ("value", "bright_white"), ("ok", "bold green"), ("warn", "bold yellow"), ("err", "bold red"), ("muted", "white"), ("logging.level.debug", "white"), ("logging.level.info", "white"), ("logging.level.warning", "bold yellow"), ("logging.level.error", "bold red"), ("logging.level.critical", "bold red"))
    accent_style                : str   = "cyan"
    heading_style               : str   = "bold cyan"
    subsection_style            : str   = "bold white"
    elapsed_style               : str   = "dim"
    progress_description_format : str   = "[progress.description]{task.description}"
    live_refresh_per_second     : int   = 4
    progress_refresh_per_second : int   = 8

    terminal_columns  : int = 120
    terminal_lines    : str = "32"
    job_stream_buffer : int = 4000
    job_stream_queue  : int = 8000
