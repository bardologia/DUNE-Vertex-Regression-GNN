import importlib

_EXPORTS = {
    "Logger"          : "monitoring",
    "ModelSummary"    : "monitoring",
    "NullLogger"      : "monitoring",
    "NullTracker"     : "monitoring",
    "ResourceMonitor" : "monitoring",
    "ShapeLogger"     : "monitoring",
    "Tracker"         : "monitoring",
    "MarkdownDoc"     : "reporting",
    "MarkdownTable"   : "reporting",
    "ScalarFormatter" : "reporting",
    "ConfigCli"       : "runtime",
    "Detacher"        : "runtime",
    "Reproducibility" : "runtime",
    "Checkpoint"      : "training",
    "EarlyStopping"   : "training",
    "GradientClipper" : "training",
    "Scheduler"       : "training",
    "Warmup"          : "training",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
