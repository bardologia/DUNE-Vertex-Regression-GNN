"use strict";

class ScriptPanel {
  constructor(refs) {
    this.gridElement = refs.grid;
    this.filterElement = refs.filters;
    this.scripts = [];
    this.filter = "All";
    this.interpreters = [];
    this.preferred = null;
    this.loaded = false;
  }

  async load() {
    const interpreterData = await window.apiGet("/api/interpreters");
    this.interpreters = interpreterData.interpreters || [];
    this.preferred = interpreterData.preferred || null;

    const data = await window.apiGet("/api/scripts");
    this.scripts = data.scripts || [];
    this.loaded = true;

    window.scriptCatalog = this;
    this._renderFilters();
    this._renderGrid();
  }

  find(key) {
    return this.scripts.find((item) => item.key === key) || null;
  }

  _categories() {
    return ["All", ...new Set(this.scripts.map((script) => script.category))];
  }

  _renderFilters() {
    this.filterElement.innerHTML = "";
    this._categories().forEach((category) => {
      const chip = document.createElement("button");
      chip.className = "chip" + (category === this.filter ? " is-active" : "");
      chip.textContent = category;
      chip.addEventListener("click", () => {
        this.filter = category;
        [...this.filterElement.children].forEach((child) => child.classList.toggle("is-active", child.textContent === category));
        this._renderGrid();
      });
      this.filterElement.appendChild(chip);
    });
  }

  _renderGrid() {
    this.gridElement.innerHTML = "";
    const items = this.scripts.filter((script) => this.filter === "All" || script.category === this.filter);

    items.forEach((script) => {
      const card = document.createElement("button");
      card.className = "script-card";
      card.innerHTML =
        `<span class="script-card__glow" aria-hidden="true"></span>` +
        `<div class="script-card__top"><span class="script-card__cat">${window.escapeHtml(script.category)}</span>` +
        `<span class="script-card__file">${window.escapeHtml(script.file)}</span></div>` +
        `<h3 class="script-card__title">${window.escapeHtml(script.label)}</h3>` +
        `<p class="script-card__purpose">${window.escapeHtml(script.purpose)}</p>` +
        `<div class="script-card__foot"><span>${script.has_config ? window.escapeHtml(script.config_class) : "no overrides"}</span>` +
        `<span class="arrow">configure &rarr;</span></div>`;
      card.addEventListener("click", () => window.router.go(`launch/${script.key}`));
      this.gridElement.appendChild(card);
    });
  }
}

window.ScriptPanel = ScriptPanel;
