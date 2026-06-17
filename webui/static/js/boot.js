"use strict";

class Application {
  constructor() {
    this.scriptPanel = null;
    this.processPanel = null;
    this.modelPanel = null;
    this.resultsPanel = null;
    this.systemPanel = null;
    this.router = null;
  }

  async init() {
    await this._probeBackend();
    this._buildPanels();
    this._buildRouter();
  }

  async _probeBackend() {
    const status = await window.apiGet("/api/system/status");
    const wrapper = document.getElementById("nav-status");
    const text = document.getElementById("status-text");
    const live = !status.error;
    wrapper.classList.toggle("is-ok", live);
    wrapper.classList.toggle("is-down", !live);
    text.textContent = live ? `${status.host || "backend"} live` : "offline";
  }

  _buildPanels() {
    this.scriptPanel = new window.ScriptPanel({
      grid: document.getElementById("script-grid"),
      filters: document.getElementById("script-filters"),
      launch: document.getElementById("launch-panel"),
    });
    this.scriptPanel.load();
    window.scriptPanel = this.scriptPanel;

    this.processPanel = new window.ProcessPanel({
      list: document.getElementById("process-list"),
      console: document.getElementById("console"),
    });
    window.processPanel = this.processPanel;

    this.modelPanel = new window.ModelPanel({
      list: document.getElementById("model-list"),
      detail: document.getElementById("model-detail"),
    });

    this.resultsPanel = new window.ResultsPanel({
      list: document.getElementById("run-list"),
      detail: document.getElementById("run-detail"),
      bar: document.getElementById("tensorboard-bar"),
    });

    this.systemPanel = new window.SystemPanel(document.getElementById("system-board"));
  }

  _buildRouter() {
    this.router = new window.Router((page) => this._onRoute(page));
    window.router = this.router;
    this.router.start();
  }

  _onRoute(page) {
    if (page === "processes") this.processPanel.enter();
    else this.processPanel.leave();

    if (page === "system") this.systemPanel.enter();
    else this.systemPanel.leave();

    if (page === "models") this.modelPanel.enter();
    if (page === "results") this.resultsPanel.enter();
  }
}

document.addEventListener("DOMContentLoaded", () => new Application().init());
