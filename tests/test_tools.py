import numpy as np
import pytest

from configuration.entry import TrainEntryConfig
from tools import EarlyStopping, GradientClipper, MarkdownDoc, MarkdownTable, Scheduler, Tracker, Warmup
from tools.runtime.reproducibility import Reproducibility, RngSnapshot


class SilentLogger:
    def section(self, *args, **kwargs): pass
    def kv_table(self, *args, **kwargs): pass
    def subsection(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass


def test_scheduler_reduces_on_plateau_then_cools_down():
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


@pytest.mark.parametrize("scheduler_type", ["cosine_annealing", "linear", "polynomial", "exponential", "step", "constant"])
def test_scheduler_types_decay_monotonically(scheduler_type):
    config                       = TrainEntryConfig().training
    config.scheduler.type        = scheduler_type
    config.scheduler.eta_min     = 1e-5
    config.scheduler.step_size   = 2
    config.loop.epochs           = 10
    config.warmup.warmup_enabled = False

    scheduler = Scheduler([0.1], None, config, SilentLogger(), Tracker(writer=None))
    values    = [scheduler.step(epoch)[0] for epoch in range(10)]

    assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:]))
    assert values[0] == pytest.approx(0.1)
    if scheduler_type != "constant":
        assert values[-1] < values[0]


def test_scheduler_state_round_trips_the_plateau_counters():
    config                       = TrainEntryConfig().training
    config.scheduler.type        = "reduce_on_plateau"
    config.scheduler.patience    = 1
    config.scheduler.threshold   = 0.0
    config.warmup.warmup_enabled = False

    scheduler = Scheduler([0.1], None, config, SilentLogger(), Tracker(writer=None))
    scheduler.step_metric(1.0)
    scheduler.step_metric(1.0)
    scheduler.step_metric(1.0)
    state = scheduler.state_dict()

    restored = Scheduler([0.1], None, config, SilentLogger(), Tracker(writer=None))
    restored.load_state_dict(state)

    assert restored.current_lrs == scheduler.current_lrs
    assert restored.num_bad_epochs == scheduler.num_bad_epochs
    assert restored.cooldown_counter == scheduler.cooldown_counter


def test_warmup_state_round_trips_the_step_counter():
    config                     = TrainEntryConfig().training
    config.warmup.warmup_steps = 4

    warmup = Warmup(config, SilentLogger(), Tracker(writer=None))
    warmup.step()
    warmup.step()

    restored = Warmup(config, SilentLogger(), Tracker(writer=None))
    restored.load_state_dict(warmup.state_dict())

    assert restored.current_step == 2
    assert restored.factor() == warmup.factor()


def test_gradient_clipper_history_excludes_non_finite_norms():
    config                            = TrainEntryConfig().training
    config.gradient_clipper.clip_mode = "adaptive_percentile"

    clipper = GradientClipper(config, SilentLogger(), Tracker(writer=None))

    clipper.record(1.0, 0)
    clipper.record(float("nan"), 1)
    clipper.record(float("inf"), 2)
    clipper.record(2.0, 3)

    assert clipper.history == [1.0, 2.0]


def test_early_stopping_state_round_trips():
    config = TrainEntryConfig().training

    stopper = EarlyStopping(config, SilentLogger(), Tracker(writer=None))
    stopper(1.0, 0)
    stopper(2.0, 1)

    restored = EarlyStopping(config, SilentLogger(), Tracker(writer=None))
    restored.load_state_dict(stopper.state_dict())

    assert restored.best_loss == 1.0
    assert restored.counter == 1
    assert restored.best_epoch == 0


def test_rng_snapshot_restores_the_global_streams():
    Reproducibility.seed_everything(7)
    snapshot = RngSnapshot()
    expected = np.random.rand(3)

    np.random.rand(10)
    snapshot.restore()

    assert np.allclose(np.random.rand(3), expected)


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
