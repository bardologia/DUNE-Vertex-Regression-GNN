import sys
from dataclasses import fields, is_dataclass
from pathlib     import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webui"))

from launch_layout import LaunchLayout

from configuration.benchmark      import BenchmarkConfig
from configuration.entry          import BaselineEntryConfig, CrossValidationEntryConfig, DatasetEntryConfig, InferenceEntryConfig, TrainEntryConfig, TuneEntryConfig
from configuration.explainability import FeatureImportanceConfig


ENTRY_CONFIGS = {
    "train"               : TrainEntryConfig,
    "tune"                : TuneEntryConfig,
    "cross_validate"      : CrossValidationEntryConfig,
    "benchmark"           : BenchmarkConfig,
    "baseline"            : BaselineEntryConfig,
    "infer"               : InferenceEntryConfig,
    "explain"             : FeatureImportanceConfig,
    "build_parquet_store" : DatasetEntryConfig,
}


def _leaves(node, prefix=""):
    for definition in fields(node):
        value = getattr(node, definition.name)
        path  = prefix + definition.name

        if is_dataclass(value):
            yield from _leaves(value, path + ".")
        else:
            yield {"path": path}


@pytest.mark.parametrize("script_key", sorted(ENTRY_CONFIGS))
def test_layout_claims_every_configuration_field(script_key):
    leaves = list(_leaves(ENTRY_CONFIGS[script_key]()))
    layout = LaunchLayout().build(script_key, leaves)

    assert layout["sections"]
    assert layout["mode"] in ("single", "sections")


def test_every_registered_script_declares_a_layout():
    assert set(LaunchLayout.LAYOUTS) == set(ENTRY_CONFIGS)


def test_layout_widgets_only_reference_known_paths():
    for script_key, config_class in ENTRY_CONFIGS.items():
        leaves = list(_leaves(config_class()))
        layout = LaunchLayout().build(script_key, leaves)
        known  = {leaf["path"] for leaf in leaves}

        assert set(layout["widgets"]) <= known


@pytest.mark.parametrize("script_key", sorted(ENTRY_CONFIGS))
def test_layout_widgets_declare_a_renderable_kind(script_key):
    layout = LaunchLayout().build(script_key, list(_leaves(ENTRY_CONFIGS[script_key]())))

    for path, widget in layout["widgets"].items():
        assert widget["kind"] in ("choice", "multi", "number", "run"), path

        if widget["kind"] == "choice":
            assert widget["options"], path
        if widget["kind"] == "multi":
            assert widget["choices"], path
        if widget["kind"] == "number":
            assert {"min", "max", "presets"} <= set(widget), path
            assert widget["presets"], path


@pytest.mark.parametrize("script_key", sorted(ENTRY_CONFIGS))
def test_special_panels_declare_a_renderable_panel(script_key):
    leaves = list(_leaves(ENTRY_CONFIGS[script_key]()))
    layout = LaunchLayout().build(script_key, leaves)
    known  = {leaf["path"] for leaf in leaves}

    for section in layout["sections"]:
        for panel in section["panels"]:
            if panel["kind"] != "special":
                continue
            assert panel["panel"] in ("model_card", "model_toggle"), panel["panel"]
            assert panel["fields"]
            assert set(panel["fields"]) <= known
