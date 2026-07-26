"use strict";

class LaunchPanel extends ConfigForm {
  constructor(refs) {
    super();
    this.refs      = refs;
    this.key       = null;
    this.script    = null;
    this.cmdEl     = null;
    this.manifestEl = null;
    this.launchBtn = null;
    this.active    = false;
    this.launching = false;
    this.loadSeq   = 0;
    this._wireTabs();
    this._wireKeys();
  }

  _resetState() {
    this.dirty         = {};
    this.controls      = {};
    this.states        = [];
    this.gates         = [];
    this.sections      = [];
    this.byPath        = new Map();
    this.activeSection = null;
    this.query         = "";
    this.modelFamilies = null;
    this.layoutEl      = null;
    this.nomatchEl     = null;
    this.countEl       = null;
  }

  _wireTabs() {
    document.querySelectorAll(".launch__tab").forEach((tab) => {
      tab.addEventListener("click", () => this._setPane(tab.dataset.pane));
    });
  }

  _wireKeys() {
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !this.active) return;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
        document.activeElement.blur();
        return;
      }
      window.router.go("scripts");
    });
  }

  _setPane(pane) {
    document.querySelectorAll(".launch__tab").forEach((tab) => {
      const active = tab.dataset.pane === pane;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });

    document.getElementById("launch-config").classList.toggle("is-active", pane === "config");
    document.getElementById("launch-source").classList.toggle("is-active", pane === "source");
  }

  async _awaitCatalog() {
    for (let attempt = 0; attempt < 60; attempt++) {
      if (window.scriptCatalog && window.scriptCatalog.loaded) return;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  _renderSkeleton() {
    this.refs.kicker.textContent  = "";
    this.refs.title.textContent   = "Loading...";
    this.refs.purpose.textContent = "";
    this.refs.facts.innerHTML     = "";
    this.refs.rail.innerHTML      = "";
    this.refs.config.innerHTML    = "";

    const wall     = document.createElement("div");
    wall.className = "launch-skeleton";
    for (let index = 0; index < 4; index++) {
      const block     = document.createElement("div");
      block.className = "launch-skeleton__panel";
      wall.appendChild(block);
    }
    this.refs.config.appendChild(wall);
  }

  _renderHead(script) {
    this.refs.kicker.textContent  = `${script.category} · ${script.file}`;
    this.refs.title.textContent   = script.label;
    this.refs.purpose.textContent = script.purpose;
    this._renderFacts();
  }

  _renderFacts() {
    const host     = this.refs.facts;
    host.innerHTML = "";

    const facts = [["entry config", this.script.has_config ? this.script.config_class : "none"]];
    if (this.config) {
      facts.push(["fields", String(this.config.leaves.length)]);
      facts.push(["sections", String(this.sections.length)]);
    }

    facts.forEach(([term, value]) => {
      const dt       = document.createElement("dt");
      dt.textContent = term;
      const dd       = document.createElement("dd");
      dd.textContent = value;
      host.appendChild(dt);
      host.appendChild(dd);
    });
  }

  _renderRail() {
    const host     = this.refs.rail;
    host.innerHTML = "";

    const interpreters = (window.scriptCatalog && window.scriptCatalog.interpreters) || [];
    const preferred    = window.scriptCatalog && window.scriptCatalog.preferred;

    const interpreter     = document.createElement("div");
    interpreter.className = "rail-block";
    interpreter.innerHTML = `<span class="rail-block__label">Interpreter</span>`;

    const select     = document.createElement("select");
    select.className = "run-select";
    select.id        = "launch-interpreter";
    interpreters.forEach((entry) => {
      const option       = document.createElement("option");
      option.value       = entry.path;
      option.textContent = `${entry.label}  ·  ${entry.path}`;
      if (entry.path === preferred) option.selected = true;
      select.appendChild(option);
    });
    interpreter.appendChild(select);

    const command     = document.createElement("div");
    command.className = "rail-block";

    const commandHead     = document.createElement("div");
    commandHead.className = "rail-block__row";
    commandHead.innerHTML = `<span class="rail-block__label">Command</span>`;

    const copy       = document.createElement("button");
    copy.className   = "btn btn--mini";
    copy.textContent = "Copy";
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(this._commandText()).then(() => window.toast("Command copied", "ok"));
    });
    commandHead.appendChild(copy);

    const code     = document.createElement("code");
    code.className = "rail-command";
    this.cmdEl     = code;
    command.appendChild(commandHead);
    command.appendChild(code);

    const manifest     = document.createElement("div");
    manifest.className = "rail-block";
    manifest.innerHTML = `<span class="rail-block__label">Overrides</span>`;

    const list       = document.createElement("div");
    list.className   = "rail-manifest";
    this.manifestEl  = list;
    manifest.appendChild(list);

    const actions     = document.createElement("div");
    actions.className = "rail-block rail-block--actions";

    const launch     = document.createElement("button");
    launch.className = "btn btn--primary rail-launch";
    launch.addEventListener("click", () => this._launch());
    this.launchBtn = launch;
    actions.appendChild(launch);

    host.appendChild(interpreter);
    host.appendChild(command);
    host.appendChild(manifest);
    host.appendChild(actions);
    this._refresh();
  }

  async _renderSource(key) {
    const code = this.refs.source;
    code.textContent = "loading...";
    code.removeAttribute("data-highlighted");

    const result = await window.apiGet(`/api/scripts/${key}/source`).catch(() => ({}));
    if (key !== this.key) return;

    code.textContent = result.ok ? result.source : (result.error || "source unavailable");
    code.className   = "language-python";

    if (result.ok && window.hljs) {
      try {
        window.hljs.highlightElement(code);
      } catch (error) {
        return;
      }
    }
  }

  _renderConfigError(message) {
    const host     = this.refs.config;
    host.innerHTML = "";

    const panel     = document.createElement("div");
    panel.className = "launch-error";

    const text       = document.createElement("pre");
    text.className   = "launch-error__text";
    text.textContent = message;

    const retry       = document.createElement("button");
    retry.className   = "btn btn--ghost";
    retry.textContent = "Retry resolution";
    retry.addEventListener("click", () => this.enter(this.key));

    panel.appendChild(text);
    panel.appendChild(retry);
    host.appendChild(panel);
  }

  _usesModelPanels(cfg) {
    return cfg.layout.sections.some((section) => section.panels.some((panel) => panel.kind === "special"));
  }

  _renderConfig(cfg) {
    const host     = this.refs.config;
    host.innerHTML = "";

    if (!cfg.leaves.length) {
      host.innerHTML = `<p class="cfg-note">${cfg.config_class} exposes no configuration fields.</p>`;
      return;
    }

    this.byPath = new Map(cfg.leaves.map((leaf) => [leaf.path, leaf]));

    host.appendChild(this._buildToolbar(cfg));
    this._renderLayout(host, cfg);
    this._renderFacts();
  }

  _navigate(key) {
    window.history.replaceState(null, "", `#/launch/${this.key}/${key}`);
    this._setActiveSection(key);
  }

  _commandText() {
    if (!this.script) return "";

    const interpreters = (window.scriptCatalog && window.scriptCatalog.interpreters) || [];
    const chosen       = document.getElementById("launch-interpreter");
    const preferred    = chosen ? chosen.value : (window.scriptCatalog && window.scriptCatalog.preferred);
    const label        = (interpreters.find((entry) => entry.path === preferred) || interpreters[0] || {}).label || "python";

    let text = `${label} ${this.script.file}`;
    Object.entries(this.dirty).forEach(([path, value]) => {
      const rendered = /^[\w@%+=:,./-]+$/.test(value) ? value : `'${String(value).replace(/'/g, `'\\''`)}'`;
      text += ` --${path} ${rendered}`;
    });
    return text;
  }

  _renderManifest() {
    this.manifestEl.innerHTML = "";
    const entries = Object.entries(this.dirty);

    if (!entries.length) {
      const empty       = document.createElement("p");
      empty.className   = "rail-manifest__empty";
      empty.textContent = "No overrides. Every field launches at its default.";
      this.manifestEl.appendChild(empty);
      return;
    }

    entries.forEach(([path, value]) => {
      const control = this.controls[path];
      const item    = document.createElement("button");
      item.type     = "button";
      item.className = "rail-override";
      item.title     = "Remove override";
      item.innerHTML =
        `<span class="rail-override__path">${path}</span>` +
        `<span class="rail-override__change">${control ? control.leaf.value : ""} &rarr; <b>${value}</b></span>` +
        `<span class="rail-override__x" aria-hidden="true">&times;</span>`;
      item.addEventListener("click", () => this._resetField(path));
      this.manifestEl.appendChild(item);
    });
  }

  _refresh() {
    const count = Object.keys(this.dirty).length;

    if (this.cmdEl)   this.cmdEl.textContent = this._commandText();
    if (this.countEl) this.countEl.textContent = count ? `${count} override${count > 1 ? "s" : ""}` : "all defaults";

    if (this.launchBtn) {
      this.launchBtn.classList.toggle("is-armed", count > 0);
      this.launchBtn.innerHTML = count
        ? `&#9654;&nbsp; Launch run <small>${count} override${count > 1 ? "s" : ""}</small>`
        : `&#9654;&nbsp; Launch run <small>all defaults</small>`;
    }

    if (this.manifestEl) this._renderManifest();
    this._refreshBadges();
    this._refreshGates();
  }

  async _launch() {
    if (!this.script || this.launching) return;

    const interpreter = document.getElementById("launch-interpreter").value;

    this.launching = true;
    this.launchBtn.disabled = true;
    try {
      const result = await window.apiPost("/api/launch", { script: this.script.key, overrides: { ...this.dirty }, interpreter });
      if (!result.ok) {
        window.toast(result.error || "Launch failed", "error");
        return;
      }
      window.toast(`Launched ${this.script.label} (pid ${result.pid})`, "ok");
      if (window.consolePanel) window.consolePanel.focusPid = result.pid;
      window.router.go("console");
    } finally {
      this.launching = false;
      this.launchBtn.disabled = false;
    }
  }

  leave() {
    this.active = false;
  }

  async enter(param) {
    this.active = true;
    if (!param) {
      window.router.go("scripts");
      return;
    }

    const [key, section] = param.split("/");
    if (key === this.key && this.config) {
      if (section) this._setActiveSection(section);
      return;
    }

    const seq = ++this.loadSeq;
    this.key    = key;
    this.script = null;
    this.config = null;
    this._resetState();
    this.activeSection = section || null;
    this._renderSkeleton();

    await this._awaitCatalog();
    if (seq !== this.loadSeq) return;

    const script = window.scriptCatalog ? window.scriptCatalog.find(key) : null;
    if (!script) {
      window.toast("Could not load script", "error");
      window.router.go("scripts");
      return;
    }

    this.script = script;
    this._renderHead(script);
    this._renderRail();
    this._renderSource(key);
    this._setPane("config");

    if (!script.has_config) {
      this.refs.config.innerHTML = `<p class="cfg-note">This script takes no overrides. Launch runs it with its built-in defaults.</p>`;
      return;
    }

    const cfg = await window.apiGet(`/api/scripts/${key}/config`);
    if (seq !== this.loadSeq) return;
    if (!cfg.ok) {
      this._renderConfigError(cfg.error || "could not resolve configuration");
      return;
    }
    this.config = cfg;

    if (this._usesModelPanels(cfg)) {
      const models = await window.apiGet("/api/models").catch(() => ({}));
      if (seq !== this.loadSeq) return;
      this.modelFamilies = (models && models.families) || [];
    }

    this._renderConfig(cfg);
    this._refresh();
  }
}

window.LaunchPanel = LaunchPanel;
