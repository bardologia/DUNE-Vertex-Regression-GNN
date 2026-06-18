"use strict";

class Application {
  constructor() {
    this.scriptPanel = null;
    this.launchPanel = null;
    this.consolePanel = null;
    this.resultsPanel = null;
    this.tensorboardView = null;
    this.observatory = null;
    this.router = null;
  }

  async init() {
    this._buildScene();
    await this._probeBackend();
    this._buildPanels();
    this._buildRouter();
  }

  _buildScene() {
    const canvas = document.getElementById("server-anim");
    if (canvas && window.ServerScene) {
      window.serverScene = new window.ServerScene(canvas);
    }
    const scene = document.getElementById("scene");
    if (scene && window.EventScene) {
      window.eventScene = new window.EventScene(scene);
    }
  }

  async _probeBackend() {
    const status = await window.apiGet("/api/system");
    const wrapper = document.getElementById("nav-status");
    const text = document.getElementById("status-text");
    const live = !status.error;
    wrapper.classList.toggle("is-ok", live);
    wrapper.classList.toggle("is-down", !live);
    text.textContent = live ? `${status.host || "backend"} live` : "offline";

    const footer = document.getElementById("footer-root");
    if (footer && status.disk) footer.textContent = status.disk.path || "";
  }

  _buildPanels() {
    this.observatory = new window.Observatory({
      board: document.getElementById("status-board"),
      host: document.getElementById("status-host"),
      sum: document.getElementById("status-sum"),
    });

    this.scriptPanel = new window.ScriptPanel({
      grid: document.getElementById("script-grid"),
      filters: document.getElementById("script-filters"),
    });
    this.scriptPanel.load();

    this.launchPanel = new window.LaunchPanel({
      rail: document.getElementById("launch-rail"),
      config: document.getElementById("launch-config"),
      kicker: document.getElementById("launch-kicker"),
      title: document.getElementById("launch-title"),
      purpose: document.getElementById("launch-purpose"),
      facts: document.getElementById("launch-facts"),
    });

    this.consolePanel = new window.ConsolePanel({
      list: document.getElementById("job-list"),
      count: document.getElementById("job-count"),
      tiles: document.getElementById("console-tiles"),
    });
    window.consolePanel = this.consolePanel;

    this.resultsPanel = new window.ResultsPanel({
      list: document.getElementById("run-list"),
      detail: document.getElementById("run-detail"),
      bar: document.getElementById("tensorboard-bar"),
    });

    this.tensorboardView = new window.TensorboardView();
  }

  _buildRouter() {
    this.router = new window.Router((page, param) => this._onRoute(page, param));
    window.router = this.router;
    this.router.start();
  }

  _onRoute(page, param) {
    if (page === "system") this.observatory.enter();
    else this.observatory.leave();

    if (page === "console") this.consolePanel.enter();
    else this.consolePanel.leave();

    if (page === "launch") this.launchPanel.enter(param);
    else this.launchPanel.leave();

    if (page === "tensorboard") this.tensorboardView.enter();
    else this.tensorboardView.leave();

    if (page === "results") this.resultsPanel.enter();
  }
}

document.addEventListener("DOMContentLoaded", () => new Application().init());
