"use strict";

class ModelPanel {
  constructor(refs) {
    this.listElement = refs.list;
    this.detailElement = refs.detail;
    this.models = [];
    this.selectedName = null;
    this.loaded = false;
  }

  async enter() {
    if (this.loaded) return;
    this.loaded = true;
    await this.load();
  }

  async load() {
    this.listElement.innerHTML = `<li class="list-empty">Loading models ...</li>`;
    const data = await window.apiGet("/api/models");

    if (!data.ok) {
      this.listElement.innerHTML = `<li class="list-empty">${window.escapeHtml(data.error || "models unavailable")}</li>`;
      this.loaded = false;
      return;
    }

    this.models = data.models || [];
    this._renderList();
    if (this.models.length) this._select(this.models[0].name);
  }

  _renderList() {
    this.listElement.innerHTML = "";
    this.models.forEach((model) => {
      const item = document.createElement("li");
      item.className = "model-item" + (model.name === this.selectedName ? " is-active" : "");
      const fieldCount = Object.keys(model.config_defaults || {}).length;
      item.innerHTML =
        `<div class="model-item__name">${window.escapeHtml(model.name)}</div>` +
        `<div class="model-item__meta">${fieldCount} config fields</div>`;
      item.addEventListener("click", () => this._select(model.name));
      this.listElement.appendChild(item);
    });
  }

  _select(name) {
    this.selectedName = name;
    const model = this.models.find((item) => item.name === name);
    this._renderList();

    this.detailElement.innerHTML =
      `<div class="model-item__name">${window.escapeHtml(model.name)}</div>` +
      `<div class="config-json">${window.escapeHtml(JSON.stringify(model.config_defaults, null, 2))}</div>`;
  }
}

window.ModelPanel = ModelPanel;
