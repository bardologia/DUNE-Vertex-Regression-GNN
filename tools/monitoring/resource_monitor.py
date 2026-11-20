from __future__ import annotations

import os
import threading
import time

import psutil
import pynvml

from configuration.monitoring.general import ResourceMonitorSettings
from configuration.shared             import IoNames, MeasurementUnits


class ResourceMonitor:
    def __init__(self, config, logger, tracker, step_getter):
        self.cfg         = config
        self.logger      = logger
        self.tracker     = tracker
        self.step_getter = step_getter

        self._load_config()
        self._init_process()
        self._init_nvml()
        self._init_disk_tracking()
        self._init_threading()
        self._init_peak_tracking()

    def _load_config(self):
        self.enabled         = bool(self.cfg.enabled)
        self.interval        = float(self.cfg.poll_interval_sec)
        self.log_to_tb       = bool(self.cfg.log_to_tensorboard)
        self.warn_ram_pct    = float(self.cfg.warn_ram_pct)
        self.warn_vram_pct   = float(self.cfg.warn_vram_pct)
        self.warn_swap_pct   = float(self.cfg.warn_swap_pct)
        self.warn_shm_pct    = float(self.cfg.warn_shm_pct)
        self.warn_cooldown_s = float(self.cfg.warn_cooldown_sec)

    def _init_process(self):
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)
        psutil.cpu_percent(None, percpu=False)

    def _init_nvml(self):
        self._nvml_ok     = False
        self._gpu_handles = []
        try:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            self._nvml_ok = True
        except Exception:
            pass

    def _init_disk_tracking(self):
        self._last_disk_io = psutil.disk_io_counters()
        self._last_disk_t  = time.time()

    def _init_threading(self):
        self._stop_evt    = threading.Event()
        self._thread      = None
        self._sample_idx  = 0
        self._last_warn_t = {}

    def _init_peak_tracking(self):
        self.peak = {
            "ram_used_gb"  : 0.0,
            "ram_pct"      : 0.0,
            "proc_rss_gb"  : 0.0,
            "swap_used_gb" : 0.0,
            "shm_used_gb"  : 0.0,
            "vram_used_gb" : 0.0,
            "vram_pct"     : 0.0,
        }

    @staticmethod
    def _bytes_to_gb(x):
        return float(x) / MeasurementUnits.bytes_per_gib

    def _get_shm_usage(self):
        try:
            usage = psutil.disk_usage(IoNames.shm_dir)
            return self._bytes_to_gb(usage.used), float(usage.percent)
        except (FileNotFoundError, PermissionError, OSError):
            return 0.0, 0.0

    def _maybe_warn(self, key, message):
        now  = time.time()
        last = self._last_warn_t.get(key, 0.0)
        if now - last < self.warn_cooldown_s:
            return
        self._last_warn_t[key] = now
        self.logger.warning(f"[ResourceMonitor] {message}")

    def _sample_ram_metrics(self, metrics):
        vm = psutil.virtual_memory()
        metrics["ram_used_gb"]      = self._bytes_to_gb(vm.used)
        metrics["ram_available_gb"] = self._bytes_to_gb(vm.available)
        metrics["ram_total_gb"]     = self._bytes_to_gb(vm.total)
        metrics["ram_pct"]          = float(vm.percent)

    def _sample_swap_metrics(self, metrics):
        sm = psutil.swap_memory()
        metrics["swap_used_gb"] = self._bytes_to_gb(sm.used)
        metrics["swap_pct"]     = float(sm.percent)

    def _sample_process_memory_metrics(self, metrics):
        try:
            mi = self.process.memory_full_info()
            metrics["proc_rss_gb"]    = self._bytes_to_gb(mi.rss)
            metrics["proc_vms_gb"]    = self._bytes_to_gb(mi.vms)
            metrics["proc_uss_gb"]    = self._bytes_to_gb(getattr(mi, "uss", 0))
            metrics["proc_shared_gb"] = self._bytes_to_gb(getattr(mi, "shared", 0))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            mi = self.process.memory_info()
            metrics["proc_rss_gb"] = self._bytes_to_gb(mi.rss)
            metrics["proc_vms_gb"] = self._bytes_to_gb(mi.vms)

    def _sample_process_stats(self, metrics):
        try:
            metrics["proc_num_threads"] = float(self.process.num_threads())
            metrics["proc_num_fds"]     = float(self.process.num_fds())
        except (psutil.AccessDenied, AttributeError):
            pass

    def _sample_process_children(self, metrics):
        try:
            children = self.process.children(recursive=True)
            metrics["proc_num_children"] = float(len(children))
            child_rss = 0
            for c in children:
                try:
                    child_rss += c.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            metrics["proc_children_rss_gb"] = self._bytes_to_gb(child_rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    def _sample_cpu_metrics(self, metrics):
        metrics["cpu_pct"]      = float(psutil.cpu_percent(None, percpu=False))
        metrics["proc_cpu_pct"] = float(self.process.cpu_percent(None))
        try:
            la1, la5, la15 = os.getloadavg()
            metrics["loadavg_1m"]  = float(la1)
            metrics["loadavg_5m"]  = float(la5)
            metrics["loadavg_15m"] = float(la15)
        except (AttributeError, OSError):
            pass

    def _sample_shm_metrics(self, metrics):
        shm_used_gb, shm_pct = self._get_shm_usage()
        metrics["shm_used_gb"] = shm_used_gb
        metrics["shm_pct"]     = shm_pct

    def _sample_disk_io_metrics(self, metrics):
        now = time.time()
        try:
            io = psutil.disk_io_counters()
            dt = max(now - self._last_disk_t, ResourceMonitorSettings.min_disk_interval_s)
            if io is not None and self._last_disk_io is not None:
                metrics["disk_read_mb_s"]  = (io.read_bytes - self._last_disk_io.read_bytes) / dt / MeasurementUnits.bytes_per_mib
                metrics["disk_write_mb_s"] = (io.write_bytes - self._last_disk_io.write_bytes) / dt / MeasurementUnits.bytes_per_mib
            self._last_disk_io = io
            self._last_disk_t  = now
        except (PermissionError, OSError):
            pass

    def _sample_gpu_nvml_metrics(self, metrics):
        busiest_used  = 0.0
        busiest_total = 0.0
        busiest_pct   = -1.0
        busiest_index = -1

        if self._nvml_ok:
            for i, h in enumerate(self._gpu_handles):
                try:
                    mem      = pynvml.nvmlDeviceGetMemoryInfo(h)
                    util     = pynvml.nvmlDeviceGetUtilizationRates(h)
                    used_gb  = self._bytes_to_gb(mem.used)
                    total_gb = self._bytes_to_gb(mem.total)
                    free_gb  = self._bytes_to_gb(mem.free)
                    pct      = ResourceMonitorSettings.percent_scale * mem.used / max(mem.total, 1)

                    metrics[f"gpu{i}_vram_used_gb"]  = used_gb
                    metrics[f"gpu{i}_vram_free_gb"]  = free_gb
                    metrics[f"gpu{i}_vram_total_gb"] = total_gb
                    metrics[f"gpu{i}_vram_pct"]      = pct
                    metrics[f"gpu{i}_util_pct"]      = float(util.gpu)
                    metrics[f"gpu{i}_mem_util_pct"]  = float(util.memory)

                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                        metrics[f"gpu{i}_temp_c"] = float(temp)
                    except Exception:
                        pass

                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(h) / ResourceMonitorSettings.milliwatts_per_watt
                        metrics[f"gpu{i}_power_w"] = float(power)
                    except Exception:
                        pass

                    if pct > busiest_pct:
                        busiest_used, busiest_total, busiest_pct, busiest_index = used_gb, total_gb, pct, i
                except Exception:
                    continue

        return busiest_used, busiest_total, busiest_index

    def _update_peak_metrics(self, metrics):
        for k in list(self.peak.keys()):
            if k in metrics and metrics[k] > self.peak[k]:
                self.peak[k] = float(metrics[k])

    def _check_warnings(self, metrics, gpu_used, gpu_total, gpu_index):
        if metrics["ram_pct"] >= self.warn_ram_pct:
            self._maybe_warn(
                "ram",
                f"RAM usage {metrics['ram_pct']:.1f}% "
                f"({metrics['ram_used_gb']:.2f}/{metrics['ram_total_gb']:.2f} GB) "
                f">= threshold {self.warn_ram_pct:.1f}% "
                f"(proc RSS {metrics['proc_rss_gb']:.2f} GB, "
                f"shm {metrics['shm_used_gb']:.2f} GB)",
            )

        vram_pct = (ResourceMonitorSettings.percent_scale * gpu_used / gpu_total) if gpu_total > 0 else 0.0
        if vram_pct >= self.warn_vram_pct and gpu_total > 0:
            self._maybe_warn(
                "vram",
                f"GPU{gpu_index} VRAM usage {vram_pct:.1f}% "
                f"({gpu_used:.2f}/{gpu_total:.2f} GB) "
                f">= threshold {self.warn_vram_pct:.1f}%",
            )

        if metrics["swap_pct"] >= self.warn_swap_pct:
            self._maybe_warn(
                "swap",
                f"Swap usage {metrics['swap_pct']:.1f}% "
                f"({metrics['swap_used_gb']:.2f} GB) >= threshold {self.warn_swap_pct:.1f}%",
            )

        if metrics["shm_pct"] >= self.warn_shm_pct:
            self._maybe_warn(
                "shm",
                f"/dev/shm usage {metrics['shm_pct']:.1f}% ({metrics['shm_used_gb']:.2f} GB) "
                f">= threshold {self.warn_shm_pct:.1f}% "
                f"(DataLoader workers may exhaust shared memory)",
            )

    def sample(self):
        metrics = {}

        self._sample_ram_metrics(metrics)
        self._sample_swap_metrics(metrics)
        self._sample_process_memory_metrics(metrics)
        self._sample_process_stats(metrics)
        self._sample_process_children(metrics)
        self._sample_cpu_metrics(metrics)
        self._sample_shm_metrics(metrics)
        self._sample_disk_io_metrics(metrics)

        gpu_used, gpu_total, gpu_index = self._sample_gpu_nvml_metrics(metrics)

        metrics["vram_used_gb"] = gpu_used
        metrics["vram_pct"]     = (ResourceMonitorSettings.percent_scale * gpu_used / gpu_total) if gpu_total > 0 else 0.0

        self._update_peak_metrics(metrics)
        self._check_warnings(metrics, gpu_used, gpu_total, gpu_index)

        return metrics

    def _tb_metrics(self, metrics):
        allowed = set(ResourceMonitorSettings.tensorboard_scalars)
        for i in range(len(self._gpu_handles)):
            allowed.update(f"gpu{i}_{suffix}" for suffix in ResourceMonitorSettings.tensorboard_gpu_scalars)

        return {k: v for k, v in metrics.items() if k in allowed}

    def _publish(self, metrics):
        step = int(self.step_getter())
        if self.log_to_tb:
            self.tracker.log_metrics("system", self._tb_metrics(metrics), step)

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                metrics = self.sample()
                self._publish(metrics)
                self._sample_idx += 1
            except Exception as exc:
                self.logger.warning(f"[ResourceMonitor] sample failed: {exc}")
            self._stop_evt.wait(self.interval)

    def _log_startup_info(self):
        self.logger.section("[Resource Monitor]")
        self.logger.kv_table({
            "Enabled"         : True,
            "Poll interval"   : f"{self.interval:.1f} s",
            "NVML available"  : f"{self._nvml_ok} ({len(self._gpu_handles)} GPUs)",
            "TB logging"      : self.log_to_tb,
            "Warn thresholds" : f"RAM>={self.warn_ram_pct:.0f}%  VRAM>={self.warn_vram_pct:.0f}%  SWAP>={self.warn_swap_pct:.0f}%  SHM>={self.warn_shm_pct:.0f}%",
        })

    def start(self):
        if not self.enabled:
            self.logger.subsection("[ResourceMonitor] disabled by config")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._log_startup_info()

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="ResourceMonitor", daemon=True)
        self._thread.start()

    def _log_peak_metrics(self):
        peaks = {}
        for key, value in self.peak.items():
            unit                 = "%" if key.endswith("_pct") else "GB"
            peaks[f"peak {key}"] = f"{value:.2f} {unit}"
        peaks["Total samples"] = self._sample_idx

        self.logger.section("[Resource Monitor - Peaks]")
        self.logger.kv_table(peaks)

    def _stop_thread(self):
        if self._thread is None:
            return

        self._stop_evt.set()
        self._thread.join(timeout=max(self.interval * ResourceMonitorSettings.join_interval_factor, ResourceMonitorSettings.min_join_timeout_s))
        self._thread = None

    def stop(self):
        self._stop_thread()
        self._log_peak_metrics()
        self._shutdown_nvml()

    def _shutdown_nvml(self):
        if self._nvml_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ok = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
