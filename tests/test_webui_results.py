import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webui"))

from project_paths  import ProjectPaths
from results_browser import ResultsBrowser


class StubServerLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _run_directory(runs_dir, name, metrics):
    directory = runs_dir / name
    (directory / "metadata").mkdir(parents=True)

    (directory / "metadata" / "resolved_config.json").write_text(json.dumps({"model_name": "graphsage"}), encoding="utf-8")
    if metrics is not None:
        (directory / "metadata" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    return directory


@pytest.fixture()
def browser(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    paths          = ProjectPaths()
    paths.runs_dir = runs_dir

    return ResultsBrowser(paths, StubServerLogger()), runs_dir


def test_run_listing_exposes_per_coordinate_errors(browser):
    results, runs_dir = browser
    _run_directory(runs_dir, "graphsage_case", {
        "unit"   : "m",
        "splits" : {
            "test" : {"euclidean_mean": 4.5, "mae_x": 2.1, "mae_y": 2.0, "mae_z": 3.4, "rmse_x": 2.9, "rmse_y": 2.8, "rmse_z": 4.5},
            "val"  : {"euclidean_mean": 4.9, "mae_x": 2.3, "mae_y": 2.2, "mae_z": 3.6, "rmse_x": 3.1, "rmse_y": 3.0, "rmse_z": 4.7},
        },
    })

    run = results.list_runs()["runs"][0]

    assert run["coordinate_errors"]["split"] == "test"
    assert run["coordinate_errors"]["unit"] == "m"
    assert run["coordinate_errors"]["values"]["mae"] == {"x": 2.1, "y": 2.0, "z": 3.4}
    assert run["coordinate_errors"]["values"]["rmse"] == {"x": 2.9, "y": 2.8, "z": 4.5}


def test_run_listing_falls_back_to_the_next_split(browser):
    results, runs_dir = browser
    _run_directory(runs_dir, "graphsage_val_only", {
        "unit"   : "m",
        "splits" : {"val": {"euclidean_mean": 4.9, "mae_x": 2.3, "mae_y": 2.2, "mae_z": 3.6, "rmse_x": 3.1, "rmse_y": 3.0, "rmse_z": 4.7}},
    })

    run = results.list_runs()["runs"][0]

    assert run["coordinate_errors"]["split"] == "val"
    assert run["coordinate_errors"]["values"]["mae"]["z"] == 3.6


def test_run_without_metrics_reports_no_coordinate_errors(browser):
    results, runs_dir = browser
    _run_directory(runs_dir, "graphsage_untrained", None)

    run = results.list_runs()["runs"][0]

    assert run["coordinate_errors"] is None
    assert run["best_metric"] is None
