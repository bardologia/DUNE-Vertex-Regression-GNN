"use strict";

class ScriptPanel {
  constructor(refs) {
    this.gridElement = refs.grid;
    this.filterElement = refs.filters;
    this.launchElement = refs.launch;
    this.scripts = [];
    this.filter = "All";
    this.interpreters = [];
    this.preferred = null;
    this.activeKey = null;
  }

  async load() {
    const interpreterData = await window.apiGet("/api/interpreters");
    this.interpreters = interpreterData.interpreters || [];
    this.preferred = interpreterData.preferred || null;

    const data = await window.apiGet("/api/scripts");
    this.scripts = data.scripts || [];
    this._renderFilters();
    this._renderGrid();
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
      const card = document.createElement("div");
      card.className = "script-card";
      card.innerHTML =
        `<div class="script-card__top"><span class="script-card__cat">${window.escapeHtml(script.category)}</span>` +
        `<span class="script-card__file">${window.escapeHtml(script.file)}</span></div>` +
        `<h3 class="script-card__title">${window.escapeHtml(script.label)}</h3>` +
        `<p class="script-card__purpose">${window.escapeHtml(script.purpose)}</p>` +
        `<div class="script-card__foot"><span>${script.has_config ? window.escapeHtml(script.config_class) : "no overrides"}</span>` +
        `<span class="arrow">configure &rarr;</span></div>`;
      card.addEventListener("click", () => this._select(script.key));
      this.gridElement.appendChild(card);
    });
  }

  async _select(key) {
    this.activeKey = key;
    const script = this.scripts.find((item) => item.key === key);

    this.launchElement.hidden = false;
    this.launchElement.innerHTML = `<div class="detail-empty">Loading configuration for ${window.escapeHtml(script.label)} ...</div>`;
    this.launchElement.scrollIntoView({ behavior: "smooth", block: "start" });

    if (!script.has_config) {
      this._renderForm(script, []);
      return;
    }

    const schema = await window.apiGet(`/api/scripts/${key}/config`);
    if (!schema.ok) {
      this.launchElement.innerHTML =
        `<div class="launch__head"><div><h2>${window.escapeHtml(script.label)}</h2><p>${window.escapeHtml(script.file)}</p></div></div>` +
        `<div class="config-json">${window.escapeHtml(schema.error || "configuration could not be resolved")}</div>`;
      return;
    }
    this._renderForm(script, schema.leaves || []);
  }

  _groupLeaves(leaves) {
    const groups = new Map();
    leaves.forEach((leaf) => {
      const section = leaf.section || "general";
      const groupKey = `${section}#${leaf.block}`;
      if (!groups.has(groupKey)) {
        groups.set(groupKey, { section, leaves: [] });
      }
      groups.get(groupKey).leaves.push(leaf);
    });
    return [...groups.values()];
  }

  _renderForm(script, leaves) {
    const groups = this._groupLeaves(leaves);

    const interpreterOptions = this.interpreters
      .map((interpreter) => {
        const selected = interpreter.path === this.preferred ? " selected" : "";
        return `<option value="${window.escapeHtml(interpreter.path)}"${selected}>${window.escapeHtml(interpreter.label)}</option>`;
      })
      .join("");

    const groupsHtml = groups
      .map((group) => {
        const title = group.section || "general";
        const fields = group.leaves
          .map((leaf) => this._fieldHtml(leaf))
          .join("");
        return `<div class="field-group"><div class="field-group__title">${window.escapeHtml(title)}</div><div class="field-grid">${fields}</div></div>`;
      })
      .join("");

    this.launchElement.innerHTML =
      `<div class="launch__head">` +
      `<div><h2>${window.escapeHtml(script.label)}</h2><p>${window.escapeHtml(script.purpose)}</p></div>` +
      `<button class="btn btn--primary" id="launch-button">Launch</button>` +
      `</div>` +
      `<div class="launch__controls">` +
      `<span class="control-label">Interpreter</span>` +
      `<select class="control" id="launch-interpreter">${interpreterOptions}</select>` +
      `</div>` +
      (leaves.length ? groupsHtml : `<div class="detail-empty">This script takes no overrides. Launch runs it with its built-in defaults.</div>`);

    this.launchElement.querySelector("#launch-button").addEventListener("click", () => this._launch(script, leaves));
  }

  _fieldHtml(leaf) {
    const editableAttribute = leaf.editable ? "" : " disabled";
    if (leaf.type === "bool") {
      const current = leaf.value === "True";
      return (
        `<div class="field"><label>${window.escapeHtml(leaf.path)}<span class="field__type">bool</span></label>` +
        `<select data-path="${window.escapeHtml(leaf.path)}" data-kind="bool"${editableAttribute}>` +
        `<option value="true"${current ? " selected" : ""}>true</option>` +
        `<option value="false"${!current ? " selected" : ""}>false</option></select></div>`
      );
    }
    return (
      `<div class="field"><label>${window.escapeHtml(leaf.path)}<span class="field__type">${window.escapeHtml(leaf.type)}</span></label>` +
      `<input type="text" data-path="${window.escapeHtml(leaf.path)}" data-default="${window.escapeHtml(leaf.value)}" value="${window.escapeHtml(leaf.value)}"${editableAttribute} /></div>`
    );
  }

  _collectOverrides(leaves) {
    const overrides = {};
    const defaults = new Map(leaves.map((leaf) => [leaf.path, leaf.value]));

    this.launchElement.querySelectorAll("[data-path]").forEach((control) => {
      if (control.disabled) return;
      const path = control.dataset.path;
      const value = control.dataset.kind === "bool" ? control.value : control.value.trim();
      const original = control.dataset.kind === "bool" ? (defaults.get(path) === "True" ? "true" : "false") : defaults.get(path);
      if (value !== original) {
        overrides[path] = value;
      }
    });
    return overrides;
  }

  async _launch(script, leaves) {
    const interpreter = this.launchElement.querySelector("#launch-interpreter").value;
    const overrides = this._collectOverrides(leaves);

    const result = await window.apiPost("/api/launch", { script: script.key, overrides, interpreter });
    if (!result.ok) {
      window.toast(result.error || "Launch failed", "error");
      return;
    }
    window.toast(`Launched ${script.label} (pid ${result.pid})`, "ok");
    if (window.processPanel) window.processPanel.focusPid = result.pid;
    if (window.router) window.router.go("processes");
  }
}

window.ScriptPanel = ScriptPanel;
