from tools import MarkdownDoc, MarkdownTable, Scheduler, Tracker


def test_scheduler_reduces_on_plateau_then_cools_down():
    from configuration.entry import TrainEntryConfig

    class SilentLogger:
        def section(self, *args, **kwargs): pass
        def kv_table(self, *args, **kwargs): pass
        def subsection(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

    config                            = TrainEntryConfig().training
    config.scheduler.type             = "reduce_on_plateau"
    config.scheduler.patience         = 1
    config.scheduler.factor           = 0.5
    config.scheduler.threshold        = 0.0
    config.scheduler.cooldown         = 1
    config.scheduler.min_lr           = 1e-8
    config.warmup.warmup_enabled      = False

    scheduler = Scheduler([0.1], None, config, SilentLogger(), Tracker(writer=None))

    assert scheduler.step_metric(1.0) == [0.1]
    assert scheduler.step_metric(1.0) == [0.1]
    assert scheduler.step_metric(1.0) == [0.05]

    assert scheduler.step_metric(1.0) == [0.05]
    assert scheduler.step_metric(1.0) == [0.05]
    assert scheduler.step_metric(1.0) == [0.025]

    assert scheduler.step_metric(0.1) == [0.025]




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
