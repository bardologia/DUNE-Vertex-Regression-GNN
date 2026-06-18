"use strict";

class TensorboardView {
  static POLL_MS = 5000;
  static SETTINGS_KEY = "_tb_global_settings";
  static AUTO_RELOAD_MS = 31000;
  static PAGINATION_SIZE = 500;

  constructor() {
    this._seedSettings();
    this.strip = document.getElementById("tb-strip");
    this.frame = document.getElementById("tb-frame");
    this.empty = document.getElementById("tb-empty");
    this.emptyTitle = document.getElementById("tb-empty-title");
    this.emptyHint = document.getElementById("tb-empty-hint");
    this.openBtn = document.getElementById("tb-open");
    this.stopBtn = document.getElementById("tb-stop");
    this.startBtn = document.getElementById("tb-start");

    this.instances = [];
    this.selectedId = null;
    this.loadedUrl = null;
    this.timer = null;
    this.active = false;

    this.openBtn.addEventListener("click", () => {
      const inst = this._selected();
      if (inst) window.open(inst.url, "_blank");
    });
    this.stopBtn.addEventListener("click", () => this._stop());
    this.startBtn.addEventListener("click", () => this._start());
  }

  _seedSettings() {
    let stored = {};
    try {
      stored = JSON.parse(localStorage.getItem(TensorboardView.SETTINGS_KEY) || "{}");
    } catch (e) {
      stored = {};
    }
    stored.autoReload = true;
    stored.autoReloadPeriodInMs = TensorboardView.AUTO_RELOAD_MS;
    stored.paginationSize = TensorboardView.PAGINATION_SIZE;
    localStorage.setItem(TensorboardView.SETTINGS_KEY, JSON.stringify(stored));
  }

  enter() {
    if (this.active) return;
    this.active = true;
    this.refresh();
    this.timer = setInterval(() => this.refresh(), TensorboardView.POLL_MS);
  }

  leave() {
    this.active = false;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  async refresh() {
    let data;
    try {
      data = await window.apiGet("/api/tensorboard");
    } catch (e) {
      return;
    }
    if (!this.active || !data || data.error) return;

    this.instances = (data.instances || []).filter((i) => i.status === "starting" || i.status === "running");

    if (!this.instances.some((i) => i.id === this.selectedId)) this.selectedId = null;
    if (!this.selectedId && this.instances.length) this.selectedId = this.instances[0].id;

    this._render();
  }

  _selected() {
    return this.instances.find((i) => i.id === this.selectedId) || null;
  }

  _shortdir(logdir) {
    const parts = String(logdir).replace(/\/+$/, "").split("/");
    return parts.slice(-2).join("/") || logdir;
  }

  _render() {
    this.strip.innerHTML = "";
    this.instances.forEach((inst) => {
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "tb-pill" + (inst.id === this.selectedId ? " is-active" : "") + (inst.status === "running" ? " is-running" : " is-starting");
      pill.title = inst.logdir;
      pill.innerHTML =
        `<span class="tb-pill__dot" aria-hidden="true"></span>` +
        `<span class="tb-pill__name">${this._shortdir(inst.logdir)}</span>` +
        `<span class="tb-pill__state">${inst.status}</span>`;
      pill.addEventListener("click", () => {
        this.selectedId = inst.id;
        this._render();
      });
      this.strip.appendChild(pill);
    });

    const inst = this._selected();
    const ready = inst && inst.status === "running";

    this.openBtn.disabled = !ready;
    this.stopBtn.disabled = !inst;

    if (ready) {
      this.empty.hidden = true;
      this.frame.classList.add("is-live");
      if (this.loadedUrl !== inst.url) {
        this.loadedUrl = inst.url;
        this.frame.src = inst.url;
      }
      return;
    }

    this.frame.classList.remove("is-live");
    if (this.loadedUrl !== null) {
      this.loadedUrl = null;
      this.frame.src = "about:blank";
    }

    this.empty.hidden = false;
    if (inst) {
      this.emptyTitle.textContent = "TensorBoard is starting";
      this.emptyHint.textContent = `Indexing ${inst.logdir} — the dashboard appears here as soon as it responds.`;
      this.startBtn.hidden = true;
    } else {
      this.emptyTitle.textContent = "No TensorBoard running";
      this.emptyHint.textContent = "Launch a training job and an instance starts automatically over its run directory, or start one manually over all runs.";
      this.startBtn.hidden = false;
    }
  }

  async _start() {
    this.startBtn.disabled = true;
    try {
      const res = await window.apiPost("/api/tensorboard/start", {});
      if (res && res.error) window.toast(res.error, "error");
      else window.toast("TensorBoard starting", "ok");
    } catch (e) {
      window.toast("Could not start TensorBoard", "error");
    }
    this.startBtn.disabled = false;
    this.refresh();
  }

  async _stop() {
    const inst = this._selected();
    if (!inst) return;
    try {
      const res = await window.apiPost(`/api/tensorboard/${inst.id}/stop`, {});
      if (res && res.error) window.toast(res.error, "error");
    } catch (e) {
      window.toast("Could not stop TensorBoard", "error");
    }
    this.selectedId = null;
    this.loadedUrl = null;
    this.frame.src = "about:blank";
    this.refresh();
  }
}

window.TensorboardView = TensorboardView;
