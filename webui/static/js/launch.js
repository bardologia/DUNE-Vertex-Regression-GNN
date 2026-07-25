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
    this.runs = [];
    this.models = [];
    this.dynamicLoaded = false;
    this.layout = null;
    this.byPath = new Map();
    this.sections = [];
    this.sectionOf = new Map();
    this.activeSection = null;
    this.query = "";
    this.tabsWired = false;
  }

  static FIELD_SPEC = {
    run_directory      : { kind: "runpick" },
    model_name         : { kind: "model" },
    device             : { kind: "segmented", options: ["cuda", "cpu"] },
    splits             : { kind: "multi", options: ["train", "val", "test"] },
    data_term          : { kind: "segmented", options: ["mse", "huber"] },
    type               : { kind: "segmented", options: ["reduce_on_plateau", "cosine_annealing", "constant"] },
    mode               : { kind: "segmented", options: ["min", "max"] },
    warmup_mode        : { kind: "select", options: ["linear", "cosine", "exponential", "polynomial"] },
    clip_mode          : { kind: "select", options: ["fixed", "adaptive_percentile", "adaptive_mean_std", "disabled"] },
    light_noise_mode   : { kind: "segmented", options: ["multiplicative", "additive"] },

    batch_size         : { presets: [8, 16, 32, 64, 128, 256] },
    epochs             : { presets: [25, 50, 100, 200, 300] },
    k_neighbors        : { presets: [4, 8, 16, 32] },
    num_workers        : { presets: [0, 2, 4, 8] },
    seed               : { presets: [0, 42, 1234] },
    patience           : { presets: [5, 10, 15, 25] },
    edge_rbf_count     : { presets: [8, 16, 32] },
    worker_count       : { presets: [4, 8, 10, 16] },
    store_worker_count : { presets: [4, 8, 10, 16] },
  };

  async enter(key) {
    if (!key) { window.router.go("scripts"); return; }
    if (key === this.currentKey && this.script) return;
    this.currentKey = key;

    await this._awaitCatalog();
    const script = window.scriptCatalog ? window.scriptCatalog.find(key) : null;
    if (!script) { window.router.go("scripts"); return; }

    this.script = script;
    this._renderHeader(script);
    this._wireTabs();
    this._setPane("config");
    this._loadSource(key);
    this.refs.config.innerHTML = `<div class="launch-skeleton"><div class="launch-skeleton__panel"></div><div class="launch-skeleton__panel"></div></div>`;
    this.refs.rail.innerHTML = "";

    const schemaPromise = script.has_config
      ? window.apiGet(`/api/scripts/${key}/config`)
      : Promise.resolve({ ok: true, leaves: [] });

    await this._loadDynamic();
    if (this.currentKey !== key) return;

    if (!script.has_config) {
      this._setup({ leaves: [], layout: null });
      return;
    }

    const schema = await schemaPromise;
    if (this.currentKey !== key) return;
    if (!schema.ok) {
      this.refs.config.innerHTML = `<div class="launch-error"><p class="launch-error__text">${window.escapeHtml(schema.error || "configuration could not be resolved")}</p></div>`;
      this._renderRail();
      return;
    }
    this._setup(schema);
  }

  leave() {}

  async _awaitCatalog() {
    for (let i = 0; i < 60; i++) {
      if (window.scriptCatalog && window.scriptCatalog.loaded) return;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  async _loadDynamic() {
    if (this.dynamicLoaded) return;
    const [runs, models] = await Promise.all([
      window.apiGet("/api/runs").catch(() => ({})),
      window.apiGet("/api/models").catch(() => ({})),
    ]);
    this.runs = (runs.runs || []).map((run) => ({ ...run, ready: this._runReady(run) }));
    this.models = (models.models || []).map((model) => model.name);
    this.dynamicLoaded = true;
  }

  _runReady(run) {
    if (run.best_metric) return true;
    const hasCheckpoint = (node) => {
      if ((node.files || []).some((file) => /\.pt$/i.test(file.name))) return true;
      return (node.children || []).some(hasCheckpoint);
    };
    return run.tree ? hasCheckpoint(run.tree) : false;
  }

  _wireTabs() {
    if (this.tabsWired) return;
    this.tabsWired = true;

    document.querySelectorAll(".launch__tab").forEach((tab) => {
      tab.addEventListener("click", () => this._setPane(tab.dataset.pane));
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

  async _loadSource(key) {
    this.refs.source.textContent = "loading…";
    this.refs.source.removeAttribute("data-highlighted");

    const result = await window.apiGet(`/api/scripts/${key}/source`).catch(() => ({}));
    if (this.currentKey !== key) return;

    this.refs.source.textContent = result.ok ? result.source : (result.error || "source unavailable");
    this.refs.source.className = "language-python";

    if (result.ok && window.hljs) {
      try {
        window.hljs.highlightElement(this.refs.source);
      } catch (error) {
        return;
      }
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

  _setup(schema) {
    const leaves = schema.leaves || [];

    this.leaves = leaves;
    this.layout = schema.layout || null;
    this.byPath = new Map(leaves.map((leaf) => [leaf.path, leaf]));
    this.defaults = new Map();
    this.values = new Map();
    this.bools = new Set();
    this.query = "";
    this.activeSection = null;

    leaves.forEach((leaf) => {
      const isBool = leaf.type === "bool";
      const value = isBool ? (leaf.value === "True" ? "true" : "false") : String(leaf.value);
      this.defaults.set(leaf.path, value);
      this.values.set(leaf.path, value);
      if (isBool) this.bools.add(leaf.path);
    });

    this._autoSelectRun();
    this._renderConfig();
    this._renderRail();
  }

  _autoSelectRun() {
    const runLeaf = this.leaves.find((leaf) => this._spec(leaf).kind === "runpick");
    if (!runLeaf) return;
    const candidate = this.runs.find((run) => run.ready) || this.runs[0];
    if (candidate) this.values.set(runLeaf.path, candidate.path);
  }

  _spec(leaf) {
    const name = this._leafName(leaf);
    const declared = (this.layout && this.layout.widgets && this.layout.widgets[leaf.path]) || LaunchPanel.FIELD_SPEC[name] || {};
    let kind = declared.kind;
    if (!kind) {
      if (leaf.type === "bool") kind = "toggle";
      else if (leaf.type === "int") kind = "int";
      else if (leaf.type === "float") kind = "float";
      else if (leaf.type === "PosixPath" || leaf.type === "Path") kind = "path";
      else kind = "text";
    }
    return { ...declared, kind };
  }

  _leafName(leaf) {
    const parts = leaf.path.split(".");
    return parts[parts.length - 1];
  }

  _gateLabel(name) {
    if (name.startsWith("use_")) return name.slice(4);
    if (name !== "enabled" && name.endsWith("_enabled")) return name.slice(0, -"_enabled".length);
    return name;
  }

  _renderConfig() {
    if (!this.leaves.length || !this.layout) {
      this.refs.config.innerHTML = `<div class="launch-empty">This script takes no overrides. Launch runs it with its built-in defaults.</div>`;
      return;
    }

    this.sections = [];
    this.sectionOf = new Map();

    const declared = [];
    if (this.layout.essentials.length) declared.push({ key: "essentials", title: "Essentials", panels: null });
    this.layout.sections.forEach((section) => declared.push(section));

    if (!this.activeSection || !declared.some((section) => section.key === this.activeSection)) {
      this.activeSection = declared[0].key;
    }

    const single = this.layout.mode === "single";
    const body = declared.map((section) => this._sectionHtml(section)).join("");
    const nav = single ? "" : `<nav class="secnav" aria-label="Configuration sections">${declared.map((section) => this._navItemHtml(section)).join("")}</nav>`;

    this.refs.config.innerHTML =
      this._toolbarHtml() +
      `<div class="launch-layout${single ? " launch-layout--single" : ""}">` +
      `<div class="secmain">${body}<p class="cfg-note launch-nomatch" hidden>No fields match this filter.</p></div>` +
      nav +
      `</div>`;

    this._wireConfig();
    this._applyVisibility();
  }

  _toolbarHtml() {
    return (
      `<div class="cfg-toolbar">` +
      `<input class="cfg-search" id="cfg-search" type="search" spellcheck="false" placeholder="Filter ${this.leaves.length} fields&hellip;" value="${window.escapeHtml(this.query)}" />` +
      `<span class="cfg-toolbar__count" id="cfg-count"></span>` +
      `<button type="button" class="btn btn--mini" id="cfg-reset-all">Reset all</button>` +
      `</div>`
    );
  }

  _navItemHtml(section) {
    const active = section.key === this.activeSection ? " is-active" : "";
    return (
      `<button type="button" class="secnav__item${active}" data-section-nav="${window.escapeHtml(section.key)}">` +
      `<span class="secnav__name">${window.escapeHtml(section.title)}</span>` +
      `<span class="edit-badge" data-badge="${window.escapeHtml(section.key)}" hidden></span>` +
      `</button>`
    );
  }

  _sectionHtml(section) {
    this.sections.push(section);

    const body = section.panels === null
      ? this._essentialsHtml()
      : section.panels.map((panel) => this._panelHtml(panel, section.key)).join("");

    return (
      `<section class="launch-section" data-section="${window.escapeHtml(section.key)}">` +
      `<h3 class="launch-section__title">${window.escapeHtml(section.title)}</h3>` +
      `<div class="launch-section__body">${body}</div>` +
      `</section>`
    );
  }

  _essentialsHtml() {
    const fields = this.layout.essentials.map((entry) => this._entryHtml(entry, "essentials")).join("");

    return (
      `<section class="launch-pins">` +
      `<header class="launch-pins__head"><h3 class="launch-pins__name">Run essentials</h3><span class="launch-pins__hint">check these before every launch</span></header>` +
      `<div class="launch-pins__grid">${fields}</div>` +
      `</section>`
    );
  }

  _panelHtml(panel, sectionKey) {
    const head = panel.title
      ? `<header class="cfg-panel__head"><h4 class="cfg-panel__name">${window.escapeHtml(panel.title)}</h4></header>`
      : "";

    const groups = panel.groups.map((group) => {
      const title = group.title ? `<div class="field-group__title">${window.escapeHtml(group.title)}</div>` : "";
      const fields = group.fields.map((entry) => this._entryHtml(entry, sectionKey)).join("");
      return `<div class="field-group">${title}<div class="field-group__grid">${fields}</div></div>`;
    }).join("");

    return (
      `<section class="cfg-panel" data-cols="${Math.min(panel.groups.length, 4)}">${head}` +
      `<div class="cfg-panel__groups">${groups}</div></section>`
    );
  }

  _entryHtml(entry, sectionKey) {
    if (!entry.gate) return this._fieldHtml(this.byPath.get(entry.path), sectionKey);

    const gate = this.byPath.get(entry.gate);
    const rows = entry.fields
      .map((sub) => this._fieldHtml(this.byPath.get(sub.path), sectionKey, entry.gate))
      .join("");

    return `<div class="band-block">${this._fieldHtml(gate, sectionKey, null, this._gateLabel(this._leafName(gate)))}${rows}</div>`;
  }

  _fieldHtml(leaf, sectionKey, gatedBy = null, labelOverride = null) {
    this.sectionOf.set(leaf.path, sectionKey);

    const path  = window.escapeHtml(leaf.path);
    const label = window.escapeHtml(labelOverride || this._leafName(leaf));
    const spec  = this._spec(leaf);
    const dirty = this.values.get(leaf.path) !== this.defaults.get(leaf.path);
    const wide  = spec.kind === "runpick";

    const classes = ["cfg-field"];
    if (dirty) classes.push("is-dirty");
    if (wide) classes.push("cfg-field--wide");
    if (gatedBy) classes.push("cfg-field--dependent");

    const gated = gatedBy ? ` data-gated-by="${window.escapeHtml(gatedBy)}"` : "";
    const name  = `<span class="cfg-field__name" title="--${path}">${label}<span class="cfg-field__type">${window.escapeHtml(leaf.type)}</span></span>`;
    const ctrl  = `<div class="cfg-field__ctrl">${this._controlHtml(leaf, spec)}</div>`;

    return `<div class="${classes.join(" ")}" data-field="${path}"${gated}>${name}${ctrl}</div>`;
  }

  _controlHtml(leaf, spec) {
    const path     = window.escapeHtml(leaf.path);
    const value    = this.values.get(leaf.path);
    const editable = leaf.editable;
    const disabled = editable ? "" : " disabled";

    if (!editable) {
      return `<input class="cfg-input" type="text" value="${window.escapeHtml(value)}" disabled />`;
    }

    if (spec.kind === "toggle") {
      const on = value === "true";
      return `<button type="button" class="switch${on ? " is-on" : ""}" data-toggle="${path}" aria-pressed="${on}"><span class="switch__knob"></span></button>`;
    }

    if (spec.kind === "segmented") {
      const opts = spec.options.map((opt) =>
        `<button type="button" class="cfg-seg__opt${String(opt) === value ? " is-active" : ""}" data-seg="${path}" data-value="${window.escapeHtml(opt)}">${window.escapeHtml(opt)}</button>`
      ).join("");
      return `<div class="cfg-seg" role="group">${opts}</div>`;
    }

    if (spec.kind === "select" || spec.kind === "model") {
      const options = spec.kind === "model" ? this.models : spec.options;
      const list = (options.length ? options : [value]).map((opt) =>
        `<option value="${window.escapeHtml(opt)}"${String(opt) === value ? " selected" : ""}>${window.escapeHtml(opt)}</option>`
      ).join("");
      return `<div class="cfg-selectwrap"><select class="cfg-select" data-select="${path}">${list}</select></div>`;
    }

    if (spec.kind === "multi") {
      const chosen = this._parseList(value);
      const opts = spec.options.map((opt) =>
        `<button type="button" class="cfg-multi__opt${chosen.includes(opt) ? " is-active" : ""}" data-multi="${path}" data-value="${window.escapeHtml(opt)}">${window.escapeHtml(opt)}</button>`
      ).join("");
      return `<div class="cfg-multi" role="group">${opts}</div>`;
    }

    if (spec.kind === "runpick") {
      return this._runPickHtml(leaf);
    }

    if (spec.kind === "int" || spec.kind === "float") {
      const step = spec.kind === "int" ? `step="1"` : `step="any"`;
      const stepper = spec.kind === "int"
        ? `<button type="button" class="cfg-num__step" data-step="${path}" data-delta="-1" aria-label="decrease">&minus;</button>`
        : "";
      const stepperUp = spec.kind === "int"
        ? `<button type="button" class="cfg-num__step" data-step="${path}" data-delta="1" aria-label="increase">+</button>`
        : "";
      const presets = (spec.presets || []).length
        ? `<div class="cfg-presets">${spec.presets.map((preset) =>
            `<button type="button" class="cfg-preset${String(preset) === value ? " is-active" : ""}" data-preset="${path}" data-value="${preset}">${preset}</button>`
          ).join("")}</div>`
        : "";
      return (
        `<div class="cfg-num">${stepper}` +
        `<input class="cfg-num__input" type="number" ${step} data-num="${path}" value="${window.escapeHtml(value)}" />` +
        `${stepperUp}</div>${presets}`
      );
    }

    return `<input class="cfg-input" type="text" data-text="${path}" value="${window.escapeHtml(value)}"${disabled} />`;
  }

  _runPickHtml(leaf) {
    const path = window.escapeHtml(leaf.path);
    const selected = this.values.get(leaf.path);

    if (!this.runs.length) {
      return `<div class="cfg-runs-empty">No runs found under <code>runs/</code>. Train a model first.</div>`;
    }

    const cards = this.runs.map((run) => {
      const active = run.path === selected;
      const metric = run.best_metric
        ? `<span class="cfg-run__metric">${window.escapeHtml(String(run.best_metric.value))}${run.best_metric.unit ? " " + window.escapeHtml(run.best_metric.unit) : ""}</span>`
        : `<span class="cfg-run__metric cfg-run__metric--none">${run.ready ? "ready" : "no checkpoint"}</span>`;
      return (
        `<button type="button" class="cfg-run${active ? " is-active" : ""}${run.ready ? "" : " is-stale"}" data-run="${path}" data-value="${window.escapeHtml(run.path)}">` +
        `<span class="cfg-run__top"><span class="cfg-run__name">${window.escapeHtml(run.name)}</span>${metric}</span>` +
        `<span class="cfg-run__meta"><span>${window.escapeHtml(run.model)}</span><span>${window.escapeHtml(run.timestamp.replace("T", " "))}</span></span>` +
        `</button>`
      );
    }).join("");

    return `<div class="cfg-runs">${cards}</div>`;
  }

  _parseList(value) {
    try {
      const parsed = JSON.parse(value.replace(/'/g, '"'));
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch (error) {
      return value.split(",").map((token) => token.trim().replace(/['"\[\]]/g, "")).filter(Boolean);
    }
  }

  _serializeList(items) {
    return "[" + items.map((item) => `'${item}'`).join(", ") + "]";
  }

  _wireConfig() {
    const root = this.refs.config;

    const search = root.querySelector("#cfg-search");
    if (search) {
      search.addEventListener("input", () => {
        this.query = search.value.trim().toLowerCase();
        this._applyVisibility();
      });
    }

    const resetAll = root.querySelector("#cfg-reset-all");
    if (resetAll) resetAll.addEventListener("click", () => this._resetAll());

    root.querySelectorAll("[data-section-nav]").forEach((button) => {
      button.addEventListener("click", () => this._navigate(button.dataset.sectionNav));
    });

    root.addEventListener("click", (event) => {
      const seg = event.target.closest("[data-seg]");
      if (seg) { this._set(seg.dataset.seg, seg.dataset.value); return; }

      const multi = event.target.closest("[data-multi]");
      if (multi) { this._toggleMulti(multi.dataset.multi, multi.dataset.value); return; }

      const run = event.target.closest("[data-run]");
      if (run) { this._set(run.dataset.run, run.dataset.value); return; }

      const preset = event.target.closest("[data-preset]");
      if (preset) { this._set(preset.dataset.preset, preset.dataset.value); return; }

      const step = event.target.closest("[data-step]");
      if (step) { this._step(step.dataset.step, Number(step.dataset.delta)); return; }

      const toggle = event.target.closest("[data-toggle]");
      if (toggle) { this._toggleBool(toggle.dataset.toggle); return; }
    });

    root.addEventListener("input", (event) => {
      const target = event.target;
      if (target.dataset.num !== undefined) { this._set(target.dataset.num, target.value, true); return; }
      if (target.dataset.text !== undefined) { this._set(target.dataset.text, target.value, true); return; }
    });

    root.addEventListener("change", (event) => {
      const select = event.target.closest("[data-select]");
      if (select) this._set(select.dataset.select, select.value);
    });
  }

  _navigate(key) {
    this.activeSection = key;
    this.refs.config.querySelectorAll("[data-section-nav]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.sectionNav === key);
    });
    this._applyVisibility();
  }

  _applyVisibility() {
    const root = this.refs.config;
    const layoutEl = root.querySelector(".launch-layout");
    if (!layoutEl) return;

    const searching = Boolean(this.query);
    layoutEl.classList.toggle("is-searching", searching);

    let visible = 0;

    root.querySelectorAll(".cfg-field").forEach((field) => {
      const path = field.dataset.field;
      const gate = field.dataset.gatedBy;
      const open = !gate || this.values.get(gate) === "true";
      const hit  = !searching || path.toLowerCase().includes(this.query);

      field.hidden = !open || !hit;
      if (!field.hidden) visible += 1;
    });

    const single = this.layout.mode === "single";
    let anyShown = false;

    root.querySelectorAll(".launch-section").forEach((element) => {
      const key      = element.dataset.section;
      const hasRows  = Boolean(element.querySelector(".cfg-field:not([hidden])"));
      const show     = searching ? hasRows : (single || key === this.activeSection);

      element.hidden = !show;
      anyShown = anyShown || show;
    });

    const nomatch = root.querySelector(".launch-nomatch");
    if (nomatch) nomatch.hidden = !searching || anyShown;

    const count = root.querySelector("#cfg-count");
    if (count) count.textContent = searching ? `${visible} of ${this.leaves.length} fields` : `${this._overrides().length} changed`;

    this._refreshBadges();
  }

  _refreshBadges() {
    const counts = new Map();
    this._overrides().forEach((override) => {
      const key = this.sectionOf.get(override.path);
      if (key) counts.set(key, (counts.get(key) || 0) + 1);
    });

    this.refs.config.querySelectorAll("[data-badge]").forEach((badge) => {
      const total = counts.get(badge.dataset.badge) || 0;
      badge.hidden = total === 0;
      badge.textContent = total ? String(total) : "";
    });
  }

  _resetAll() {
    this.defaults.forEach((value, path) => this.values.set(path, value));
    this._renderConfig();
    this._renderRail();
  }

  _set(path, value, fromInput) {
    this.values.set(path, String(value));
    this._refreshField(path, fromInput);
    this._renderRail();
  }

  _toggleBool(path) {
    const on = !(this.values.get(path) === "true");
    this.values.set(path, on ? "true" : "false");
    this._refreshField(path, false);
    this._renderRail();
  }

  _toggleMulti(path, option) {
    const items = this._parseList(this.values.get(path));
    const index = items.indexOf(option);
    if (index >= 0) items.splice(index, 1);
    else items.push(option);
    this.values.set(path, this._serializeList(items));
    this._refreshField(path, false);
    this._renderRail();
  }

  _step(path, delta) {
    const current = Number(this.values.get(path));
    const next = (Number.isFinite(current) ? current : 0) + delta;
    this.values.set(path, String(next < 0 ? 0 : next));
    this._refreshField(path, false);
    this._renderRail();
  }

  _refreshField(path, fromInput) {
    const leaf = this.byPath.get(path);
    if (!leaf) return;

    const wrapper = this.refs.config.querySelector(`.cfg-field[data-field="${CSS.escape(path)}"]`);
    if (!wrapper) return;

    wrapper.classList.toggle("is-dirty", this.values.get(path) !== this.defaults.get(path));

    const spec = this._spec(leaf);

    if (spec.kind === "runpick") {
      const value = this.values.get(path);
      wrapper.querySelectorAll(".cfg-run").forEach((card) => card.classList.toggle("is-active", card.dataset.value === value));
    } else if (!fromInput) {
      const ctrl = wrapper.querySelector(".cfg-field__ctrl");
      ctrl.innerHTML = this._controlHtml(leaf, spec);
    } else {
      const value = this.values.get(path);
      wrapper.querySelectorAll(".cfg-preset").forEach((chip) => chip.classList.toggle("is-active", chip.dataset.value === value));
    }

    this._applyVisibility();
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
      `<div class="cfg-selectwrap"><select class="cfg-select" id="launch-interpreter">${options}</select></div></div>` +
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
    this._refreshField(path, false);
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
