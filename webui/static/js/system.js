"use strict";

class SystemPanel {
  constructor(element) {
    this.element = element;
    this.pollTimer = null;
  }

  enter() {
    this._poll();
    clearInterval(this.pollTimer);
    this.pollTimer = setInterval(() => this._poll(), 2000);
  }

  leave() {
    clearInterval(this.pollTimer);
    this.pollTimer = null;
  }

  async _poll() {
    const status = await window.apiGet("/api/system/status");
    if (status.error) return;
    this._render(status);
  }

  _bar(percent) {
    const safe = Math.max(0, Math.min(100, percent || 0));
    return `<div class="bar"><div class="bar__fill ${window.barClass(safe)}" style="width:${safe}%"></div></div>`;
  }

  _metric(label, value) {
    return `<div class="metric-row"><span class="metric-row__label">${label}</span><span class="metric-row__value">${value}</span></div>`;
  }

  _cpuPanel(status) {
    const cpu = status.cpu;
    const cores = (cpu.per_core || [])
      .map((percent) => {
        const safe = Math.max(0, Math.min(100, percent));
        return `<div class="core"><div class="core__fill" style="height:${safe}%"></div></div>`;
      })
      .join("");

    return (
      `<div class="panel"><div class="panel__title">CPU</div>` +
      this._metric("total", `${cpu.percent.toFixed(1)}%`) +
      this._bar(cpu.percent) +
      this._metric("cores", String(cpu.count)) +
      `<div class="cores" style="margin-top:10px">${cores}</div></div>`
    );
  }

  _memoryPanel(status) {
    const ram = status.ram;
    return (
      `<div class="panel"><div class="panel__title">Memory</div>` +
      this._metric("used", `${window.formatBytes(ram.used)} / ${window.formatBytes(ram.total)}`) +
      this._bar(ram.percent) +
      this._metric("percent", `${ram.percent.toFixed(1)}%`) +
      `</div>`
    );
  }

  _diskPanel(status) {
    const disk = status.disk;
    return (
      `<div class="panel"><div class="panel__title">Disk</div>` +
      this._metric("used", `${window.formatBytes(disk.used)} / ${window.formatBytes(disk.total)}`) +
      this._bar(disk.percent) +
      this._metric("free", window.formatBytes(disk.free)) +
      `<div class="process-item__meta">${window.escapeHtml(disk.path)}</div></div>`
    );
  }

  _gpuPanel(status) {
    if (!status.gpus || !status.gpus.length) {
      return `<div class="panel panel--wide"><div class="panel__title">GPU</div><div class="detail-empty">No NVIDIA devices detected.</div></div>`;
    }

    const cards = status.gpus
      .map((gpu) => {
        const memoryPercent = gpu.mem_total ? (100 * gpu.mem_used) / gpu.mem_total : 0;
        return (
          `<div class="gpu-card"><div class="gpu-card__name"><span>[${gpu.index}] ${window.escapeHtml(gpu.name)}</span>` +
          `<span class="gpu-card__temp">${gpu.temp}&deg;C</span></div>` +
          this._metric("utilisation", `${gpu.util}%`) +
          this._bar(gpu.util) +
          this._metric("memory", `${window.formatBytes(gpu.mem_used)} / ${window.formatBytes(gpu.mem_total)}`) +
          this._bar(memoryPercent) +
          `</div>`
        );
      })
      .join("");

    return `<div class="panel panel--wide"><div class="panel__title">GPU</div>${cards}</div>`;
  }

  _render(status) {
    this.element.innerHTML =
      `<div class="panel"><div class="panel__title">Host</div>` +
      this._metric("hostname", window.escapeHtml(status.host)) +
      `</div>` +
      this._cpuPanel(status) +
      this._memoryPanel(status) +
      this._diskPanel(status) +
      this._gpuPanel(status);
  }
}

window.SystemPanel = SystemPanel;
