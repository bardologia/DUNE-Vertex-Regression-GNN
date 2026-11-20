from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass
from pathlib     import Path
from typing      import Callable, Optional

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import pynvml
import torch
import torch.nn as nn
from torch.utils.data       import Dataset
from torch_geometric.loader import DataLoader as GraphDataLoader

from configuration.shared          import IoNames, MeasurementUnits
from configuration.tools           import LoaderSweepPlotStyle, LoaderTuningSettings
from tools.runtime.reproducibility import Reproducibility


@dataclass(frozen=True)
class LoaderSpec:
    batch_size         : int
    num_workers        : int
    prefetch_factor    : int  = 4
    pin_memory         : bool = True
    persistent_workers : bool = True

    @property
    def label(self) -> str:
        return f"bs{self.batch_size}_w{self.num_workers}_pf{self.prefetch_factor}_pin{int(self.pin_memory)}_persist{int(self.persistent_workers)}"

    def as_record(self) -> dict:
        return {
            "batch_size"         : self.batch_size,
            "num_workers"        : self.num_workers,
            "prefetch_factor"    : self.prefetch_factor if self.num_workers > 0 else 0,
            "pin_memory"         : self.pin_memory,
            "persistent_workers" : self.persistent_workers and self.num_workers > 0,
        }


class GpuUtilizationSampler:
    def __init__(self, gpu_index: Optional[int] = None, interval_s: float = LoaderTuningSettings.sample_interval_s) -> None:
        self.interval_s = float(interval_s)
        self.gpu_index  = self._resolve_index(gpu_index)

        self._handle       = None
        self._nvml_ok      = False
        self._stop_evt     = threading.Event()
        self._thread       = None
        self._util_samples = []
        self._vram_samples = []

    @staticmethod
    def _resolve_index(gpu_index: Optional[int]) -> int:
        if gpu_index is not None:
            return int(gpu_index)

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible.strip():
            return int(visible.split(",")[0])

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())

        return 0

    def start(self) -> None:
        try:
            pynvml.nvmlInit()
            self._handle  = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml_ok = True
        except pynvml.NVMLError:
            self._nvml_ok = False
            return

        self._util_samples = []
        self._vram_samples = []
        self._stop_evt.clear()

        self._thread = threading.Thread(target=self._run, name="GpuUtilizationSampler", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

        self._util_samples.append(float(util.gpu))
        self._vram_samples.append(float(mem.used) / MeasurementUnits.bytes_per_gib)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            self._sample()
            self._stop_evt.wait(self.interval_s)

    def stop(self) -> dict:
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=max(self.interval_s * LoaderTuningSettings.join_intervals, LoaderTuningSettings.join_floor_s))
            self._thread = None

        if self._nvml_ok:
            pynvml.nvmlShutdown()
            self._nvml_ok = False

        if not self._util_samples:
            return {"gpu_available": False, "gpu_util_mean": float("nan"), "gpu_util_max": float("nan"), "vram_peak_gb": float("nan"), "gpu_n_samples": 0}

        return {
            "gpu_available" : True,
            "gpu_util_mean" : float(np.mean(self._util_samples)),
            "gpu_util_max"  : float(np.max(self._util_samples)),
            "vram_peak_gb"  : float(np.max(self._vram_samples)),
            "gpu_n_samples" : int(len(self._util_samples)),
        }


class GpuFeedBenchmark:
    def __init__(
        self,
        dataset        : Dataset,
        model          : nn.Module,
        to_model_input : Callable,
        forward_loss   : Callable,
        device         : torch.device,
        warmup_batches : int,
        timed_batches  : int,
        use_amp        : bool          = False,
        seed           : int           = 0,
        gpu_index      : Optional[int] = None,
        cpu_threads    : Optional[int] = None,
    ) -> None:
        self.dataset        = dataset
        self.model          = model.to(device)
        self.to_model_input = to_model_input
        self.forward_loss   = forward_loss
        self.device         = device
        self.use_amp        = bool(use_amp)
        self.seed           = int(seed)
        self.warmup_batches = int(warmup_batches)
        self.timed_batches  = int(timed_batches)
        self.gpu_index      = gpu_index

        if cpu_threads is not None:
            torch.set_num_threads(max(1, int(cpu_threads)))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LoaderTuningSettings.learning_rate)

    @staticmethod
    def _worker_init(worker_id: int) -> None:
        torch.set_num_threads(1)

    def _build_loader(self, spec: LoaderSpec) -> GraphDataLoader:
        loader_kwargs = dict(
            batch_size  = spec.batch_size,
            num_workers = spec.num_workers,
            pin_memory  = spec.pin_memory,
            shuffle     = True,
            drop_last   = True,
            generator   = Reproducibility.generator(self.seed),
        )

        if spec.num_workers > 0:
            loader_kwargs["persistent_workers"] = spec.persistent_workers
            loader_kwargs["prefetch_factor"]    = spec.prefetch_factor
            loader_kwargs["worker_init_fn"]     = GpuFeedBenchmark._worker_init

        return GraphDataLoader(self.dataset, **loader_kwargs)

    def _measure_loader_only(self, loader, spec: LoaderSpec) -> float:
        iterator = self._iterate(loader, self.warmup_batches + self.timed_batches)

        for _ in range(self.warmup_batches):
            next(iterator)

        start = time.perf_counter()
        count = 0
        for _ in range(self.timed_batches):
            next(iterator)
            count += 1
        elapsed = time.perf_counter() - start

        return count * spec.batch_size / max(elapsed, LoaderTuningSettings.time_floor_s)

    def _reference_batch(self, loader):
        batch = next(self._iterate(loader, 1))
        return self.to_model_input(batch, self.device)

    def _measure_compute_ceiling(self, reference_input, spec: LoaderSpec) -> float:
        self.model.train()

        for _ in range(self.warmup_batches):
            self._train_step(reference_input)
        self._synchronize()

        start = time.perf_counter()
        for _ in range(self.timed_batches):
            self._train_step(reference_input)
        self._synchronize()
        elapsed = time.perf_counter() - start

        return self.timed_batches * spec.batch_size / max(elapsed, LoaderTuningSettings.time_floor_s)

    def _measure_end_to_end(self, loader, spec: LoaderSpec) -> dict:
        self.model.train()

        sampler = GpuUtilizationSampler(gpu_index=self.gpu_index)
        sampler.start()

        self._synchronize()

        data_seconds    = 0.0
        compute_seconds = 0.0
        counted         = 0
        index           = 0
        previous        = time.perf_counter()

        for batch in self._iterate(loader, self.warmup_batches + self.timed_batches):
            after_fetch = time.perf_counter()
            model_input = self.to_model_input(batch, self.device)
            self._train_step(model_input)
            self._synchronize()
            after_step  = time.perf_counter()

            if index >= self.warmup_batches:
                data_seconds    += after_fetch - previous
                compute_seconds += after_step - after_fetch
                counted         += 1

            index   += 1
            previous = time.perf_counter()

        gpu_summary = sampler.stop()

        total_seconds = data_seconds + compute_seconds
        throughput    = counted * spec.batch_size / max(total_seconds, LoaderTuningSettings.time_floor_s)
        wait_fraction = data_seconds / max(total_seconds, LoaderTuningSettings.time_floor_s)

        return {
            "end_to_end_samples_per_s" : throughput,
            "data_wait_fraction"       : wait_fraction,
            **gpu_summary,
        }

    @staticmethod
    def _iterate(loader, n_batches: int):
        if len(loader) == 0:
            raise RuntimeError("DataLoader produced zero batches; dataset smaller than one batch with drop_last=True")

        produced = 0
        while produced < n_batches:
            for batch in loader:
                yield batch
                produced += 1
                if produced >= n_batches:
                    return

    def _train_step(self, model_input):
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            loss = self.forward_loss(self.model, model_input)

        loss.backward()
        self.optimizer.step()

        return loss

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure(self, spec: LoaderSpec) -> dict:
        loader = self._build_loader(spec)

        loader_only     = self._measure_loader_only(loader, spec)
        reference       = self._reference_batch(loader)
        compute_ceiling = self._measure_compute_ceiling(reference, spec)
        end_to_end      = self._measure_end_to_end(loader, spec)

        feed_ratio = loader_only / max(compute_ceiling, LoaderTuningSettings.time_floor_s)
        efficiency = end_to_end["end_to_end_samples_per_s"] / max(compute_ceiling, LoaderTuningSettings.time_floor_s)

        record = {
            **spec.as_record(),
            "loader_only_samples_per_s"     : loader_only,
            "compute_ceiling_samples_per_s" : compute_ceiling,
            "feed_ratio"                    : feed_ratio,
            "compute_efficiency"            : efficiency,
            **end_to_end,
        }

        del reference
        del loader
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return record


class DataLoaderSweep:
    def __init__(self, benchmark: GpuFeedBenchmark, specs: list[LoaderSpec], on_result: Optional[Callable[[dict], None]] = None) -> None:
        self.benchmark = benchmark
        self.specs     = list(specs)
        self.on_result = on_result

    def _measure_spec(self, spec: LoaderSpec) -> dict:
        try:
            record = self.benchmark.measure(spec)
            record["status"] = "ok"
            return record
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {**spec.as_record(), "status": "oom"}
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                return {**spec.as_record(), "status": "oom"}
            raise

    def run(self) -> list[dict]:
        results = []

        for spec in self.specs:
            record = self._measure_spec(spec)
            results.append(record)
            if self.on_result is not None:
                self.on_result(record)

        return results


class SweepReport:
    def __init__(self, results: list[dict], wait_threshold: float = LoaderTuningSettings.wait_threshold) -> None:
        self.wait_threshold = float(wait_threshold)
        self.frame          = pd.DataFrame(results)
        self.ok_frame       = self.frame[self.frame["status"] == "ok"].copy()

    @property
    def dataframe(self):
        return self.frame

    @property
    def recommendation(self) -> dict:
        if self.ok_frame.empty:
            return {"found": False, "reason": "no successful configurations"}

        saturated = self._saturated()
        cpu_bound = saturated.empty

        pool = self.ok_frame if cpu_bound else saturated
        pool = pool.sort_values(by=["end_to_end_samples_per_s", "num_workers", "batch_size"], ascending=[False, True, True])
        best = pool.iloc[0]

        return {
            "found"                    : True,
            "cpu_bound"                : bool(cpu_bound),
            "batch_size"               : int(best["batch_size"]),
            "num_workers"              : int(best["num_workers"]),
            "prefetch_factor"          : int(best["prefetch_factor"]),
            "pin_memory"               : bool(best["pin_memory"]),
            "persistent_workers"       : bool(best["persistent_workers"]),
            "end_to_end_samples_per_s" : float(best["end_to_end_samples_per_s"]),
            "data_wait_fraction"       : float(best["data_wait_fraction"]),
            "gpu_util_mean"            : float(best["gpu_util_mean"]),
            "feed_ratio"               : float(best["feed_ratio"]),
        }

    def _saturated(self):
        return self.ok_frame[(self.ok_frame["data_wait_fraction"] <= self.wait_threshold) | (self.ok_frame["feed_ratio"] >= 1.0)]

    def plot_throughput_vs_batch(self, fig_dir: Path) -> Path:
        figure, axes = self._new_axes()

        for workers, group in self.ok_frame.groupby("num_workers"):
            ordered = group.sort_values("batch_size")
            axes.plot(ordered["batch_size"], ordered["end_to_end_samples_per_s"], marker=LoaderSweepPlotStyle.marker, label=f"{workers} workers")

        axes.set_xscale("log", base=2)
        axes.set_xlabel("Batch size (samples)")
        axes.set_ylabel("End-to-end throughput (samples/s)")
        axes.set_title("End-to-end training throughput vs batch size")
        axes.legend(title="DataLoader workers", bbox_to_anchor=LoaderSweepPlotStyle.legend_anchor, loc="upper left", borderaxespad=0.0)

        return self._save(figure, fig_dir, IoNames.throughput_figure)

    def plot_data_wait_vs_workers(self, fig_dir: Path) -> Path:
        figure, axes = self._new_axes()

        for batch_size, group in self.ok_frame.groupby("batch_size"):
            ordered = group.sort_values("num_workers")
            axes.plot(ordered["num_workers"], LoaderSweepPlotStyle.percent * ordered["data_wait_fraction"], marker=LoaderSweepPlotStyle.marker, label=f"batch {batch_size}")

        axes.axhline(LoaderSweepPlotStyle.percent * self.wait_threshold, color=LoaderSweepPlotStyle.threshold_color, linestyle=LoaderSweepPlotStyle.threshold_style, linewidth=LoaderSweepPlotStyle.threshold_width, label=f"{LoaderSweepPlotStyle.percent * self.wait_threshold:.0f}% target")
        axes.set_xlabel("DataLoader workers")
        axes.set_ylabel("GPU idle waiting for data (%)")
        axes.set_title("Data-starvation share of step time vs worker count")
        axes.legend(bbox_to_anchor=LoaderSweepPlotStyle.legend_anchor, loc="upper left", borderaxespad=0.0)

        return self._save(figure, fig_dir, IoNames.data_wait_figure)

    def plot_gpu_util_vs_throughput(self, fig_dir: Path) -> Path:
        figure, axes = self._new_axes()

        scatter = axes.scatter(
            self.ok_frame["end_to_end_samples_per_s"],
            self.ok_frame["gpu_util_mean"],
            c=self.ok_frame["num_workers"],
            cmap=LoaderSweepPlotStyle.colormap,
            s=LoaderSweepPlotStyle.scatter_size,
        )

        axes.set_xlabel("End-to-end throughput (samples/s)")
        axes.set_ylabel("Mean GPU utilization (%)")
        axes.set_title("Achieved GPU utilization vs throughput")

        colorbar = figure.colorbar(scatter, ax=axes)
        colorbar.set_label("DataLoader workers")

        return self._save(figure, fig_dir, IoNames.gpu_util_figure)

    def plot_feed_ratio_vs_workers(self, fig_dir: Path) -> Path:
        figure, axes = self._new_axes()

        for batch_size, group in self.ok_frame.groupby("batch_size"):
            ordered = group.sort_values("num_workers")
            axes.plot(ordered["num_workers"], ordered["feed_ratio"], marker=LoaderSweepPlotStyle.marker, label=f"batch {batch_size}")

        axes.axhline(1.0, color=LoaderSweepPlotStyle.threshold_color, linestyle=LoaderSweepPlotStyle.threshold_style, linewidth=LoaderSweepPlotStyle.threshold_width, label="GPU-bound threshold")
        axes.set_xlabel("DataLoader workers")
        axes.set_ylabel("Feed ratio (loader / GPU ceiling)")
        axes.set_title("CPU feed capacity relative to the GPU compute ceiling")
        axes.legend(bbox_to_anchor=LoaderSweepPlotStyle.legend_anchor, loc="upper left", borderaxespad=0.0)

        return self._save(figure, fig_dir, IoNames.feed_ratio_figure)

    def _new_axes(self):
        figure, axes = plt.subplots(figsize=LoaderSweepPlotStyle.figure_size)
        return figure, axes

    def _save(self, figure, fig_dir: Path, name: str) -> Path:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        out_path = fig_dir / name
        figure.savefig(out_path, dpi=LoaderSweepPlotStyle.dpi)
        plt.close(figure)

        return out_path

    def save_all(self, fig_dir: Path) -> list[Path]:
        return [
            self.plot_throughput_vs_batch(fig_dir),
            self.plot_data_wait_vs_workers(fig_dir),
            self.plot_gpu_util_vs_throughput(fig_dir),
            self.plot_feed_ratio_vs_workers(fig_dir),
        ]
