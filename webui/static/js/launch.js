"use strict";

class LaunchPanel {
  constructor(refs) {
    this.refs = refs;
    this.script = null;
    this.leaves = [];
    this.defaults = new Map();
    this.values = new Map();
    this.bools = new Set();
    this.currentKey = null;
  }

  async enter(key) {
    if (!key) { window.router.go("scripts"); return; }
    if (key === this.currentKey && this.script) return;
    this.currentKey = key;

    await this._awaitCatalog();
    const script = window.scriptCatalog ? window.scriptCatalog.find(key) : null;
    if (!script) { window.router.go("scripts"); return; }

    this.script = script;
    this._renderHeader(script);
    this.refs.config.innerHTML = `<div class="launch-skeleton"><div class="launch-skeleton__panel"></div><div class="launch-skeleton__panel"></div></div>`;
    this.refs.rail.innerHTML = "";

    if (!script.has_config) {
      this._setup([]);
      return;
    }

    const schema = await window.apiGet(`/api/scripts/${key}/config`);
    if (this.currentKey !== key) return;
    if (!schema.ok) {
      this.refs.config.innerHTML = `<div class="launch-error"><p class="launch-error__text">${window.escapeHtml(schema.error || "configuration could not be resolved")}</p></div>`;
      this._renderRail();
      return;
    }
    this._setup(schema.leaves || []);
  }

  leave() {}

  async _awaitCatalog() {
    for (let i = 0; i < 60; i++) {
      if (window.scriptCatalog && window.scriptCatalog.loaded) return;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  _renderHeader(script) {
    this.refs.kicker.textContent = script.category;
    this.refs.title.textContent = script.label;
    this.refs.purpose.textContent = script.purpose;
    this.refs.facts.innerHTML =
      `<div><dt>entry</dt><dd>${window.escapeHtml(script.file)}</dd></div>` +
      `<div><dt>config</dt><dd>${script.has_config ? window.escapeHtml(script.config_class) : "none"}</dd></div>`;
  }

  _setup(leaves) {
    this.leaves = leaves;
    this.defaults = new Map();
    this.values = new Map();
    this.bools = new Set();

    leaves.forEach((leaf) => {
      const isBool = leaf.type === "bool";
      const value = isBool ? (leaf.value === "True" ? "true" : "false") : String(leaf.value);
      this.defaults.set(leaf.path, value);
      this.values.set(leaf.path, value);
      if (isBool) this.bools.add(leaf.path);
    });

    this._renderConfig();
    this._renderRail();
  }

  _groups() {
    const groups = new Map();
    this.leaves.forEach((leaf) => {
      const section = this._topSection(leaf);
      if (!groups.has(section)) groups.set(section, []);
      groups.get(section).push(leaf);
    });
    return [...groups.entries()];
  }

  _topSection(leaf) {
    const parts = leaf.path.split(".");
    return parts.length > 1 ? parts[0] : "general";
  }

  _subgroups(leaves) {
    const groups = new Map();
    leaves.forEach((leaf) => {
      const namespace = this._namespace(leaf);
      if (!groups.has(namespace)) groups.set(namespace, []);
      groups.get(namespace).push(leaf);
    });
    return [...groups.entries()];
  }

  _namespace(leaf) {
    return leaf.path.split(".").slice(1, -1).join(".");
  }

  _leafName(leaf) {
    const parts = leaf.path.split(".");
    return parts[parts.length - 1];
  }

  _humanize(token) {
    return token
      .split(".")
      .map((segment) => segment.replace(/_/g, " ").replace(/^./, (character) => character.toUpperCase()))
      .join(" · ");
  }

  _renderConfig() {
    if (!this.leaves.length) {
      this.refs.config.innerHTML = `<div class="launch-empty">This script takes no overrides. Launch runs it with its built-in defaults.</div>`;
      return;
    }

    const bands = this._groups().map(([section, leaves]) => {
      const body = this._subgroups(leaves).map(([namespace, groupLeaves]) => {
        const fields  = groupLeaves.map((leaf) => this._fieldHtml(leaf)).join("");
        const heading = namespace ? `<h4 class="band-subgroup__head">${window.escapeHtml(this._humanize(namespace))}</h4>` : "";
        return `<div class="band-subgroup">${heading}<div class="band-fields">${fields}</div></div>`;
      }).join("");

      const open = " is-open";
      return (
        `<section class="launch-band${open}" data-section="${window.escapeHtml(section)}">` +
        `<header class="band-head"><i class="band-head__chev" aria-hidden="true"></i>` +
        `<h3 class="band-head__name">${window.escapeHtml(this._humanize(section))}</h3>` +
        `<span class="band-head__count">${leaves.length} field${leaves.length === 1 ? "" : "s"}</span></header>` +
        `<div class="band-body">${body}</div>` +
        `</section>`
      );
    }).join("");

    this.refs.config.innerHTML = `<div class="launch-bands">${bands}</div>`;

    this.refs.config.querySelectorAll(".band-head").forEach((head) => {
      head.addEventListener("click", () => head.parentElement.classList.toggle("is-open"));
    });
    this.refs.config.querySelectorAll(".cfg-edit__input").forEach((input) => {
      input.addEventListener("input", () => this._onInput(input));
    });
    this.refs.config.querySelectorAll(".switch").forEach((sw) => {
      sw.addEventListener("click", () => this._onToggle(sw));
    });
  }

  _fieldHtml(leaf) {
    const path  = window.escapeHtml(leaf.path);
    const label = window.escapeHtml(this._leafName(leaf));
    const name  = `${label}<span>${window.escapeHtml(leaf.type)}</span>`;
    const disabled = leaf.editable ? "" : " disabled";

    if (leaf.type === "bool") {
      const on = this.values.get(leaf.path) === "true";
      return (
        `<div class="cfg-edit__row" title="${path}"><span class="cfg-edit__name">${name}</span>` +
        `<button type="button" class="switch${on ? " is-on" : ""}" data-path="${path}"${disabled ? " disabled" : ""} aria-pressed="${on}"><span class="switch__knob"></span></button></div>`
      );
    }
    const value = window.escapeHtml(this.values.get(leaf.path));
    return (
      `<div class="cfg-edit__row" title="${path}"><span class="cfg-edit__name">${name}</span>` +
      `<input class="cfg-edit__input" type="text" data-path="${path}" value="${value}"${disabled} /></div>`
    );
  }

  _onInput(input) {
    const path = input.dataset.path;
    this.values.set(path, input.value);
    input.classList.toggle("is-dirty", input.value !== this.defaults.get(path));
    this._renderRail();
  }

  _onToggle(sw) {
    if (sw.disabled) return;
    const path = sw.dataset.path;
    const on = !(this.values.get(path) === "true");
    this.values.set(path, on ? "true" : "false");
    sw.classList.toggle("is-on", on);
    sw.setAttribute("aria-pressed", String(on));
    sw.classList.toggle("is-dirty", (on ? "true" : "false") !== this.defaults.get(path));
    this._renderRail();
  }

  _overrides() {
    const out = [];
    this.values.forEach((value, path) => {
      const original = this.defaults.get(path);
      if (value !== original) out.push({ path, value, original });
    });
    return out;
  }

  _renderRail() {
    const interpreters = (window.scriptCatalog && window.scriptCatalog.interpreters) || [];
    const preferred = window.scriptCatalog && window.scriptCatalog.preferred;
    const options = interpreters
      .map((interp) => `<option value="${window.escapeHtml(interp.path)}"${interp.path === preferred ? " selected" : ""}>${window.escapeHtml(interp.label)}</option>`)
      .join("");

    const overrides = this._overrides();
    const manifest = overrides.length
      ? overrides.map((o) =>
          `<button type="button" class="rail-override" data-reset="${window.escapeHtml(o.path)}">` +
          `<span class="rail-override__path">${window.escapeHtml(o.path)}</span>` +
          `<span class="rail-override__change">${window.escapeHtml(o.original)} &rarr; <b>${window.escapeHtml(o.value)}</b></span>` +
          `<span class="rail-override__x" aria-hidden="true">&times;</span></button>`
        ).join("")
      : `<p class="rail-manifest__empty">No overrides. Launch runs the script with its configuration defaults.</p>`;

    const interpreterName = (interpreters.find((i) => i.path === preferred) || interpreters[0] || {}).label || "python";
    const commandLines = [`${interpreterName} ${this.script ? this.script.file : ""}`]
      .concat(overrides.map((o) => `  ↳ ${o.path} = ${o.value}`))
      .map((line) => window.escapeHtml(line))
      .join("\n");

    this.refs.rail.innerHTML =
      `<div class="rail-block"><span class="rail-block__label">Interpreter</span>` +
      `<select class="run-select" id="launch-interpreter">${options}</select></div>` +
      `<div class="rail-block"><span class="rail-block__label">Overrides &middot; ${overrides.length}</span>` +
      `<div class="rail-manifest">${manifest}</div></div>` +
      `<div class="rail-block"><span class="rail-block__label">Command</span>` +
      `<pre class="rail-command">${commandLines}</pre></div>` +
      `<div class="rail-block rail-block--actions">` +
      `<button class="btn btn--primary rail-launch" id="launch-button">Launch <small>${overrides.length ? `${overrides.length} override${overrides.length === 1 ? "" : "s"}` : "defaults"}</small></button></div>`;

    this.refs.rail.querySelectorAll("[data-reset]").forEach((btn) => {
      btn.addEventListener("click", () => this._reset(btn.dataset.reset));
    });
    this.refs.rail.querySelector("#launch-button").addEventListener("click", () => this._launch());
  }

  _reset(path) {
    this.values.set(path, this.defaults.get(path));
    const control = this.refs.config.querySelector(`[data-path="${CSS.escape(path)}"]`);
    if (control) {
      if (this.bools.has(path)) {
        const on = this.defaults.get(path) === "true";
        control.classList.toggle("is-on", on);
        control.classList.remove("is-dirty");
        control.setAttribute("aria-pressed", String(on));
      } else {
        control.value = this.defaults.get(path);
        control.classList.remove("is-dirty");
      }
    }
    this._renderRail();
  }

  async _launch() {
    const interpreter = this.refs.rail.querySelector("#launch-interpreter").value;
    const overrides = {};
    this._overrides().forEach((o) => { overrides[o.path] = o.value; });

    const result = await window.apiPost("/api/launch", { script: this.script.key, overrides, interpreter });
    if (!result.ok) {
      window.toast(result.error || "Launch failed", "error");
      return;
    }
    window.toast(`Launched ${this.script.label} (pid ${result.pid})`, "ok");
    if (window.consolePanel) window.consolePanel.focusPid = result.pid;
    window.router.go("console");
  }
}

window.LaunchPanel = LaunchPanel;
