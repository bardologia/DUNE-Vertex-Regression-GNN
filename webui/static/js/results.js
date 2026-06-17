"use strict";

class ResultsPanel {
  constructor(refs) {
    this.listElement = refs.list;
    this.detailElement = refs.detail;
    this.barElement = refs.bar;
    this.runs = [];
    this.selectedName = null;
  }

  async enter() {
    await this.load();
    await this._refreshTensorboard();
  }

  async load() {
    const data = await window.apiGet("/api/runs");
    this.runs = data.runs || [];
    this._renderList();
    if (this.runs.length) this._select(this.runs[0].name);
    else this.detailElement.innerHTML = `<div class="detail-empty">No runs under runs/ yet.</div>`;
  }

  _renderList() {
    this.listElement.innerHTML = "";
    if (!this.runs.length) {
      this.listElement.innerHTML = `<li class="list-empty">No runs found.</li>`;
      return;
    }
    this.runs.forEach((run) => {
      const item = document.createElement("li");
      item.className = "run-item" + (run.name === this.selectedName ? " is-active" : "");
      const metric = run.best_metric ? `${run.best_metric.name} ${run.best_metric.value}` : run.model;
      item.innerHTML =
        `<div class="run-item__top"><span class="run-item__name">${window.escapeHtml(run.name)}</span></div>` +
        `<div class="run-item__meta">${window.escapeHtml(metric)}</div>` +
        `<div class="run-item__meta">${window.escapeHtml(run.timestamp.replace("T", " "))}</div>`;
      item.addEventListener("click", () => this._select(run.name));
      this.listElement.appendChild(item);
    });
  }

  _select(name) {
    this.selectedName = name;
    const run = this.runs.find((item) => item.name === name);
    this._renderList();

    const metricRow = run.best_metric ? `<dt>best metric</dt><dd>${window.escapeHtml(run.best_metric.name)} = ${window.escapeHtml(String(run.best_metric.value))}</dd>` : "";

    this.detailElement.innerHTML =
      `<div class="run-item__top"><span class="model-item__name">${window.escapeHtml(run.name)}</span>` +
      `<button class="btn btn--primary btn--mini" id="tb-launch">Launch TensorBoard</button></div>` +
      `<dl class="kv"><dt>model</dt><dd>${window.escapeHtml(run.model)}</dd>` +
      `<dt>timestamp</dt><dd>${window.escapeHtml(run.timestamp.replace("T", " "))}</dd>` +
      `<dt>path</dt><dd>${window.escapeHtml(run.path)}</dd>${metricRow}</dl>` +
      `<div class="tree">${this._treeHtml(run.tree)}</div>`;

    this.detailElement.querySelector("#tb-launch").addEventListener("click", () => this._launchTensorboard(run.name));
  }

  _treeHtml(node) {
    const files = (node.files || [])
      .map((file) => `<div class="tree__file"><span>${window.escapeHtml(file.name)}</span><span class="tree__size">${window.formatBytes(file.size)}</span></div>`)
      .join("");
    const children = (node.children || [])
      .map((child) => `<div class="tree__dir">${window.escapeHtml(child.name)}/</div><div class="tree__children">${this._treeHtml(child)}</div>`)
      .join("");
    return files + children;
  }

  async _launchTensorboard(name) {
    const result = await window.apiPost("/api/tensorboard", { run: name });
    if (!result.ok) {
      window.toast(result.error || "TensorBoard launch failed", "error");
      return;
    }
    window.toast(`TensorBoard on port ${result.port}`, "ok");
    await this._refreshTensorboard();
  }

  async _refreshTensorboard() {
    const data = await window.apiGet("/api/tensorboard");
    const instances = data.instances || [];

    if (!instances.length) {
      this.barElement.innerHTML = "";
      return;
    }

    this.barElement.innerHTML = instances
      .map((instance) => {
        const stopButton = instance.status === "running" ? `<button class="btn btn--mini btn--danger" data-stop="${instance.pid}">Stop</button>` : "";
        return (
          `<div class="tb-instance"><span class="badge badge--${instance.status === "running" ? "running" : "failed"}">${instance.status}</span>` +
          `<a href="${window.escapeHtml(instance.url)}" target="_blank">${window.escapeHtml(instance.url)}</a>` +
          `<span class="process-item__meta">${window.escapeHtml(instance.logdir)}</span>` +
          `<span class="console__spacer"></span>${stopButton}</div>`
        );
      })
      .join("");

    this.barElement.querySelectorAll("[data-stop]").forEach((button) => {
      button.addEventListener("click", async () => {
        await window.apiPost(`/api/tensorboard/${button.dataset.stop}/stop`, {});
        this._refreshTensorboard();
      });
    });
  }
}

window.ResultsPanel = ResultsPanel;
