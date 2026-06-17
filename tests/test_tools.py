from tools import (
    Checkpoint,
    EarlyStopping,
    ExponentialMovingAverage,
    GradientClipper,
    Logger,
    MarkdownDoc,
    MarkdownTable,
    Scheduler,
    Tracker,
    Warmup,
)


def test_tools_importable():
    for component in (Logger, Tracker, Warmup, Scheduler, EarlyStopping, GradientClipper, Checkpoint, ExponentialMovingAverage):
        assert component is not None


def test_markdown_document_renders():
    table = MarkdownTable(["Layer", "Parameters"], align=["left", "right"])
    table.add_row("encoder", "1000")

    document = MarkdownDoc("Summary").paragraph("body").table(table)
    rendered = document.render()

    assert "Summary" in rendered
    assert "encoder" in rendered


def test_logger_basic(quiet_logger):
    quiet_logger.section("section")
    quiet_logger.kv_table({"alpha": 1, "beta": 2.0})
    quiet_logger.metrics_table([{"a": 1, "b": 2}], ["a", "b"], title="metrics")
