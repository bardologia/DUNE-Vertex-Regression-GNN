import torch

from tools.benchmarking      import LoaderSpec, SweepReport
from tools.training          import TrainerState
from tools.training.pretraining import BatchSizeFinder


class RecordingLogger:
    def __init__(self):
        self.lines = []

    def section(self, *args, **kwargs):
        pass

    def kv_table(self, *args, **kwargs):
        pass

    def subsection(self, message):
        self.lines.append(message)

    def info(self, message):
        self.lines.append(message)

    def warning(self, message):
        self.lines.append(message)


def _finder(peaks, budget_gb, ceiling, logger):
    def trial_step(batch_size):
        if peaks[batch_size] is None:
            raise torch.cuda.OutOfMemoryError("probe out of memory")
        return peaks[batch_size]

    return BatchSizeFinder(trial_step=trial_step, budget_gb=budget_gb, ceiling=ceiling, device=torch.device("cpu"), logger=logger)


def test_batch_size_finder_candidates_are_powers_of_two():
    assert BatchSizeFinder.candidates(10) == [1, 2, 4, 8]
    assert BatchSizeFinder.candidates(1) == [1]


def test_batch_size_finder_stops_at_the_budget():
    peaks  = {1: 0.5, 2: 1.0, 4: 2.0, 8: 4.5}
    result = _finder(peaks, budget_gb=3.0, ceiling=8, logger=RecordingLogger()).run()

    assert result["status"] == "PASS"
    assert result["batch_size"] == 4
    assert result["peak_gb"] == 2.0
    assert [trial["status"] for trial in result["trials"]] == ["FIT", "FIT", "FIT", "OVER"]


def test_batch_size_finder_keeps_the_last_fit_before_an_oom():
    peaks  = {1: 0.5, 2: 1.0, 4: None}
    result = _finder(peaks, budget_gb=8.0, ceiling=4, logger=RecordingLogger()).run()

    assert result["status"] == "PASS"
    assert result["batch_size"] == 2
    assert result["trials"][-1]["status"] == "OOM"


def test_batch_size_finder_fails_when_batch_one_is_too_large():
    result = _finder({1: 9.0}, budget_gb=1.0, ceiling=1, logger=RecordingLogger()).run()

    assert result["status"] == "FAIL"
    assert "batch size 1" in result["error"]


def test_batch_size_finder_calls_the_oom_hook():
    calls = []
    finder = BatchSizeFinder(
        trial_step = lambda batch_size: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("boom")),
        budget_gb  = 1.0,
        ceiling    = 1,
        device     = torch.device("cpu"),
        logger     = RecordingLogger(),
        on_oom     = lambda: calls.append("released"),
    )

    finder.run()
    assert calls == ["released"]


def _sweep_record(num_workers, throughput, wait_fraction, feed_ratio=0.5):
    spec = LoaderSpec(batch_size=16, num_workers=num_workers)
    return {
        **spec.as_record(),
        "status"                   : "ok",
        "end_to_end_samples_per_s" : throughput,
        "data_wait_fraction"       : wait_fraction,
        "feed_ratio"               : feed_ratio,
        "gpu_util_mean"            : 80.0,
    }


def test_sweep_report_prefers_a_saturated_configuration():
    records = [
        _sweep_record(0, 300.0, 0.40),
        _sweep_record(4, 220.0, 0.02),
        _sweep_record(8, 210.0, 0.01),
    ]

    recommendation = SweepReport(records, wait_threshold=0.05).recommendation

    assert recommendation["found"] is True
    assert recommendation["cpu_bound"] is False
    assert recommendation["num_workers"] == 4


def test_sweep_report_reports_cpu_bound_when_nothing_saturates():
    records        = [_sweep_record(0, 100.0, 0.6), _sweep_record(4, 180.0, 0.5)]
    recommendation = SweepReport(records, wait_threshold=0.05).recommendation

    assert recommendation["cpu_bound"] is True
    assert recommendation["num_workers"] == 4


def test_sweep_report_finds_nothing_when_every_spec_failed():
    records = [{**LoaderSpec(batch_size=8, num_workers=0).as_record(), "status": "oom"}]

    assert SweepReport(records).recommendation["found"] is False


def test_trainer_state_path_is_inside_the_run_directory(tmp_path):
    assert TrainerState.path(tmp_path) == tmp_path / "last.pt"
