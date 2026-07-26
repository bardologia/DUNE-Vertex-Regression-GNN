"use strict";

class PythonLiteral {
  static parse(text) {
    const reader = new PythonLiteral(String(text));
    const value  = reader._value();
    reader._ws();
    if (reader.pos !== reader.text.length) throw new Error(`trailing input at ${reader.pos}`);
    return value;
  }

  static render(value) {
    if (value === null || value === undefined) return "None";
    if (value === true)  return "True";
    if (value === false) return "False";
    if (typeof value === "number") return String(value);
    if (typeof value === "string") return `'${value.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
    if (Array.isArray(value)) return `[${value.map((item) => PythonLiteral.render(item)).join(", ")}]`;
    return `{${Object.entries(value).map(([key, item]) => `${PythonLiteral.render(key)}: ${PythonLiteral.render(item)}`).join(", ")}}`;
  }

  constructor(text) {
    this.text = text;
    this.pos  = 0;
  }

  _ws() {
    while (this.pos < this.text.length && /\s/.test(this.text[this.pos])) this.pos += 1;
  }

  _value() {
    this._ws();
    const ch = this.text[this.pos];

    if (ch === "{") return this._dict();
    if (ch === "[") return this._seq("]");
    if (ch === "(") return this._seq(")");
    if (ch === "'" || ch === '"') return this._string(ch);
    return this._atom();
  }

  _dict() {
    this.pos += 1;
    const out = {};

    while (true) {
      this._ws();
      if (this.pos >= this.text.length) throw new Error("unterminated dict");
      if (this.text[this.pos] === "}") { this.pos += 1; return out; }

      const key = this._value();
      this._ws();
      if (this.text[this.pos] !== ":") throw new Error(`expected ':' at ${this.pos}`);
      this.pos += 1;

      out[String(key)] = this._value();
      this._ws();
      if (this.text[this.pos] === ",") this.pos += 1;
    }
  }

  _seq(close) {
    this.pos += 1;
    const out = [];

    while (true) {
      this._ws();
      if (this.pos >= this.text.length) throw new Error("unterminated sequence");
      if (this.text[this.pos] === close) { this.pos += 1; return out; }

      out.push(this._value());
      this._ws();
      if (this.text[this.pos] === ",") this.pos += 1;
    }
  }

  _string(quote) {
    this.pos += 1;
    let out = "";

    while (this.pos < this.text.length) {
      const ch = this.text[this.pos];
      if (ch === "\\") { out += this.text[this.pos + 1]; this.pos += 2; continue; }
      if (ch === quote) { this.pos += 1; return out; }
      out += ch;
      this.pos += 1;
    }
    throw new Error("unterminated string");
  }

  _atom() {
    const start = this.pos;
    while (this.pos < this.text.length && !/[\s,\]})\:]/.test(this.text[this.pos])) this.pos += 1;
    const token = this.text.slice(start, this.pos);

    if (token === "True")  return true;
    if (token === "False") return false;
    if (token === "None")  return null;
    if (token === "") throw new Error(`empty token at ${start}`);

    const number = Number(token);
    return Number.isNaN(number) ? token : number;
  }
}


class LaunchWidgetDom {
  static mini(label, onClick) {
    const btn       = document.createElement("button");
    btn.type        = "button";
    btn.className   = "btn btn--mini";
    btn.textContent = label;
    btn.addEventListener("click", onClick);
    return btn;
  }
}


class ModelPickPanel {
  constructor(view, leaf) {
    this.view      = view;
    this.leaf      = leaf;
    this.families  = view.modelFamilies || [];
    this.cards     = new Map();
    this.currentEl = null;
  }

  build() {
    const root     = document.createElement("section");
    root.className = "model-panel";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Model</h3><span class="special-head__hint">the architecture to train</span>`;

    const current     = document.createElement("span");
    current.className = "model-panel__count";
    this.currentEl    = current;
    head.appendChild(current);
    root.appendChild(head);

    const body     = document.createElement("div");
    body.className = "model-panel__families";
    this.families.forEach((family) => body.appendChild(this._family(family)));
    root.appendChild(body);

    this.view.controls[this.leaf.path] = { leaf: this.leaf, reset: () => this._paint() };
    this._paint();
    return root;
  }

  _family(family) {
    const block     = document.createElement("div");
    block.className = "model-family";

    const name       = document.createElement("div");
    name.className   = "model-family__name";
    name.textContent = family.family;

    const grid     = document.createElement("div");
    grid.className = "model-pick__grid";
    family.models.forEach((model) => grid.appendChild(this._card(model)));

    block.appendChild(name);
    block.appendChild(grid);
    return block;
  }

  _card(model) {
    const card     = document.createElement("button");
    card.type      = "button";
    card.className = "model-pick";
    card.title     = `--${this.leaf.path} ${model.key}`;

    const badge = model.recommended ? `<span class="model-pick__badge">recommended</span>` : "";

    card.innerHTML =
      `<span class="model-pick__top"><span class="model-pick__name">${model.name}</span>${badge}<span class="model-pick__params">${model.capacity || ""}</span></span>` +
      `<span class="model-pick__meta">${model.blurb || ""}</span>`;

    card.addEventListener("click", () => this._select(model.key));
    this.cards.set(model.key, card);
    return card;
  }

  _select(key) {
    this.view._setValue(this.leaf, key);
    this._paint();
  }

  _paint() {
    const current = this.view._effective(this.leaf);
    let label     = current;

    this.cards.forEach((card, key) => {
      const on = key === current;
      card.classList.toggle("is-on", on);
      card.setAttribute("aria-pressed", String(on));
      if (on) label = card.querySelector(".model-pick__name").textContent;
    });

    this.currentEl.textContent = label;
  }
}


class ModelTogglePanel {
  constructor(view, leaf) {
    this.view     = view;
    this.leaf     = leaf;
    this.families = view.modelFamilies || [];
    this.keys     = this.families.flatMap((family) => family.models.map((model) => model.key));
    this.chips    = new Map();
    this.countEl  = null;
  }

  build() {
    const root     = document.createElement("section");
    root.className = "model-panel";

    const head     = document.createElement("header");
    head.className = "special-head";
    head.innerHTML = `<h3 class="special-head__name">Models in run</h3><span class="special-head__hint">toggled-off models are skipped</span>`;

    const count     = document.createElement("span");
    count.className = "model-panel__count";
    this.countEl    = count;

    head.appendChild(count);
    head.appendChild(LaunchWidgetDom.mini("All on",  () => this._emit(new Set())));
    head.appendChild(LaunchWidgetDom.mini("All off", () => this._emit(new Set(this.keys))));
    root.appendChild(head);

    const body     = document.createElement("div");
    body.className = "model-panel__families";
    this.families.forEach((family) => body.appendChild(this._family(family)));
    root.appendChild(body);

    this.view.controls[this.leaf.path] = { leaf: this.leaf, reset: () => this._paint() };
    this._paint();
    return root;
  }

  _family(family) {
    const block     = document.createElement("div");
    block.className = "model-family";

    const name       = document.createElement("div");
    name.className   = "model-family__name";
    name.textContent = family.family;

    const grid     = document.createElement("div");
    grid.className = "model-family__grid";
    family.models.forEach((model) => grid.appendChild(this._chip(model)));

    block.appendChild(name);
    block.appendChild(grid);
    return block;
  }

  _chip(model) {
    const chip     = document.createElement("button");
    chip.type      = "button";
    chip.className = "model-chip";
    chip.title     = `--${this.leaf.path} · ${model.key}`;
    chip.innerHTML = `<span class="model-chip__name">${model.name}</span><span class="model-chip__meta">${model.capacity || ""}</span>`;
    chip.addEventListener("click", () => this._toggle(model.key));
    this.chips.set(model.key, chip);
    return chip;
  }

  _skipped() {
    try {
      const parsed = PythonLiteral.parse(this.view._effective(this.leaf));
      return new Set(Array.isArray(parsed) ? parsed.map(String) : []);
    } catch (error) {
      return new Set();
    }
  }

  _toggle(key) {
    const skipped = this._skipped();
    if (skipped.has(key)) skipped.delete(key);
    else                  skipped.add(key);
    this._emit(skipped);
  }

  _emit(skipped) {
    const ordered = this.keys.filter((key) => skipped.has(key));
    this.view._setValue(this.leaf, PythonLiteral.render(ordered));
    this._paint();
  }

  _paint() {
    const skipped = this._skipped();

    this.chips.forEach((chip, key) => {
      const on = !skipped.has(key);
      chip.classList.toggle("is-on", on);
      chip.setAttribute("aria-pressed", String(on));
    });

    const active = this.keys.filter((key) => !skipped.has(key)).length;
    this.countEl.textContent = `${active}/${this.keys.length} active`;
  }
}


class RunPicker {
  static CUSTOM = "__custom__";

  constructor(view, leaf) {
    this.view    = view;
    this.leaf    = leaf;
    this.el      = null;
    this.select  = null;
    this.custom  = null;
    this.note    = null;
    this.runs    = [];
  }

  build() {
    this.el           = document.createElement("div");
    this.el.className = "picker picker--run";

    this.select           = document.createElement("select");
    this.select.className = "cfg-edit__input picker__select";
    this.select.addEventListener("change", () => this._onSelect());

    this.custom             = document.createElement("input");
    this.custom.className   = "cfg-edit__input picker__custom";
    this.custom.spellcheck  = false;
    this.custom.hidden      = true;
    this.custom.placeholder = "absolute path";
    this.custom.addEventListener("input", () => {
      this.custom.classList.toggle("is-dirty", this.custom.value !== this.leaf.value);
      this.view._setValue(this.leaf, this.custom.value);
    });

    this.note           = document.createElement("span");
    this.note.className = "picker__note";

    this.el.appendChild(this.select);
    this.el.appendChild(this.custom);
    this.el.appendChild(this.note);

    this._renderOptions();
    this._load();

    return { el: this.el, input: this.select, reset: () => this._reset() };
  }

  async _load() {
    this.note.textContent = "listing runs...";

    const result = await window.apiGet("/api/runs").catch(() => ({}));
    this.runs    = (result.runs || []).map((run) => ({ ...run, ready: this._ready(run) }));

    this.note.textContent = this.runs.length
      ? `${this.runs.length} run${this.runs.length > 1 ? "s" : ""} under ${result.runs_dir || "runs/"}`
      : `no runs under ${result.runs_dir || "runs/"}`;

    this._renderOptions();
  }

  _ready(run) {
    if (run.best_metric) return true;

    const checkpoint = (node) => {
      if ((node.files || []).some((file) => /\.pt$/i.test(file.name))) return true;
      return (node.children || []).some(checkpoint);
    };
    return run.tree ? checkpoint(run.tree) : false;
  }

  _label(run) {
    const metric = run.best_metric ? `  ${run.best_metric.value}${run.best_metric.unit ? " " + run.best_metric.unit : ""}` : (run.ready ? "" : "  (no checkpoint)");
    return `${run.name}  ·  ${run.model}${metric}`;
  }

  _renderOptions() {
    const current = this.view._effective(this.leaf);
    const known   = new Set(this.runs.map((run) => run.path));

    this.select.innerHTML = "";

    if (current && !known.has(current)) this.select.appendChild(this._option(current, `${this._tail(current)} (current)`));
    this.runs.forEach((run) => this.select.appendChild(this._option(run.path, this._label(run))));
    this.select.appendChild(this._option(RunPicker.CUSTOM, "Custom path..."));

    const isCustom = !this.custom.hidden && this.custom.value && !known.has(this.custom.value);
    this.select.value = isCustom ? RunPicker.CUSTOM : current;
    this.select.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }

  _onSelect() {
    if (this.select.value === RunPicker.CUSTOM) {
      this.custom.hidden = false;
      if (!this.custom.value) this.custom.value = this.view._effective(this.leaf);
      this.custom.focus();
      this.view._setValue(this.leaf, this.custom.value);
      return;
    }

    this.custom.hidden = true;
    this.select.classList.toggle("is-dirty", this.select.value !== this.leaf.value);
    this.view._setValue(this.leaf, this.select.value);
  }

  _option(value, label) {
    const option       = document.createElement("option");
    option.value       = value;
    option.textContent = label;
    return option;
  }

  _tail(path) {
    const trimmed = String(path).replace(/\/+$/, "");
    const cut     = trimmed.lastIndexOf("/");
    return cut >= 0 ? trimmed.slice(cut + 1) : trimmed;
  }

  _reset() {
    this.custom.hidden = true;
    this.custom.value  = "";
    this.custom.classList.remove("is-dirty");
    this._renderOptions();
  }
}


class NumberField {
  constructor(view, leaf, short, spec = null) {
    this.view    = view;
    this.leaf    = leaf;
    this.short   = short || leaf.path.split(".").pop();
    this.integer = leaf.type === "int";
    this.default = Number.isFinite(Number(leaf.value)) ? Number(leaf.value) : 0;
    this.log     = false;
    this.range   = this._resolve(spec);
    this.input   = null;
    this.chips   = new Map();
    this.reset   = () => this._paint();
  }

  _resolve(spec) {
    const range = spec
      ? { min: spec.min, max: spec.max, step: spec.step || 1, log: Boolean(spec.log), presets: spec.presets.slice() }
      : this._fallback();

    this.log = range.log;
    range.min = Math.min(range.min, this.default);
    range.max = Math.max(range.max, this.default);
    range.presets.push(this.default);
    range.presets = this._cleanPresets(range.presets, range);
    return range;
  }

  _fallback() {
    if (!this.integer && this.default > 0 && this.default <= 1) {
      return { min: 0, max: 1, step: 0.01, log: false, presets: [0, 0.25, 0.5, 0.75, 1] };
    }

    const base    = Math.abs(this.default) || (this.integer ? 10 : 1);
    const max     = this._nice(base * 4);
    const min     = this.default < 0 ? -max : 0;
    const span    = max - min || 1;
    const step    = this.integer ? 1 : Math.pow(10, Math.floor(Math.log10(span)) - 2) || 0.01;
    const presets = [min, min + span * 0.25, min + span * 0.5, min + span * 0.75, max].map((x) => (this.integer ? Math.round(x) : this._nice(x)));
    return { min, max, step, log: false, presets };
  }

  _nice(x) {
    if (x === 0) return 0;
    const unit = Math.pow(10, Math.floor(Math.log10(Math.abs(x)))) / 10;
    return Math.round(x / unit) * unit;
  }

  _cleanPresets(list, range) {
    const within = list.filter((x) => Number.isFinite(x) && x >= range.min - 1e-9 && x <= range.max + 1e-9);
    const seen   = new Map();
    within.forEach((x) => {
      const key = this.integer ? String(Math.round(x)) : this._fmt(x);
      if (!seen.has(key)) seen.set(key, Number(key));
    });
    return [...seen.values()].sort((a, b) => a - b).slice(0, 8);
  }

  _fmt(value) {
    if (this.integer) return String(Math.round(value));
    if (value === 0) return "0";
    return String(Number(value.toPrecision(this.log ? 2 : 6)));
  }

  _chip(value) {
    const chip       = document.createElement("button");
    chip.type        = "button";
    chip.className   = "numfield__chip";
    chip.textContent = this._fmt(value);
    chip.title       = `set ${this.short} = ${this._fmt(value)}`;
    chip.addEventListener("click", () => {
      this.input.value = this._fmt(value);
      this.view._setValue(this.leaf, value === this.default ? this.leaf.value : this._fmt(value));
      this._mark();
    });
    this.chips.set(value, chip);
    return chip;
  }

  _mark() {
    const current = Number(this.view._effective(this.leaf));
    this.input.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
    this.chips.forEach((chip, key) => {
      const tolerance = this.integer ? 0.5 : Math.max(1e-12, Math.abs(current) * 1e-6);
      chip.classList.toggle("is-active", Number.isFinite(current) && Math.abs(Number(key) - current) < tolerance);
    });
  }

  _paint() {
    const effective  = this.view._effective(this.leaf);
    this.input.value = effective === "None" ? "" : effective;
    this._mark();
  }

  build() {
    const el     = document.createElement("div");
    el.className = "numfield";

    const top     = document.createElement("div");
    top.className = "numfield__top";

    const input      = document.createElement("input");
    input.className  = "cfg-edit__input numfield__input";
    input.type       = "number";
    input.step       = this.integer ? "1" : "any";
    input.spellcheck = false;
    this.input       = input;

    input.addEventListener("input", () => {
      const raw   = input.value;
      const value = Number(raw);
      this.view._setValue(this.leaf, raw === "" ? "" : (value === this.default ? this.leaf.value : raw));
      this._mark();
    });

    const presets     = document.createElement("div");
    presets.className = "numfield__presets";
    this.range.presets.forEach((value) => presets.appendChild(this._chip(value)));

    top.appendChild(input);
    top.appendChild(presets);
    el.appendChild(top);

    this._paint();
    return { el, input, reset: this.reset };
  }
}


class MultiValueField {
  constructor(view, leaf, spec) {
    this.view  = view;
    this.leaf  = leaf;
    this.spec  = spec;
    this.el    = null;
    this.chips = null;
    this.input = null;
    this.count = null;
    this.reset = () => this._paint();
  }

  _values() {
    try {
      const parsed = PythonLiteral.parse(this.view._effective(this.leaf));
      return Array.isArray(parsed) ? parsed.slice() : [];
    } catch (error) {
      return [];
    }
  }

  _cast(token) {
    if (!this.spec.numeric) return token;
    const value = Number(token);
    if (!Number.isFinite(value)) return null;
    return this.spec.integer ? Math.trunc(value) : value;
  }

  _emit(values) {
    this.view._setValue(this.leaf, PythonLiteral.render(values));
    this._paint();
  }

  _onKey(event) {
    if (event.key !== "Enter" && event.key !== ",") return;
    event.preventDefault();
    this._commitEntry();
  }

  _commitEntry() {
    if (!this.input) return;

    const tokens = this.input.value.split(",").map((part) => part.trim()).filter(Boolean);
    if (!tokens.length) return;

    const values = this._values();
    tokens.forEach((token) => {
      const cast = this._cast(token);
      if (cast !== null && !values.some((existing) => existing === cast)) values.push(cast);
    });

    this.input.value = "";
    this._emit(values);
  }

  _toggleChoice(value) {
    const values = this._values();
    const index  = values.indexOf(value);
    if (index >= 0) values.splice(index, 1);
    else            values.push(value);

    this._emit(this.spec.choices.map((choice) => choice.value).filter((choice) => values.includes(choice)));
  }

  _removeValue(value) {
    this._emit(this._values().filter((existing) => existing !== value));
  }

  _paint() {
    const values         = this._values();
    this.chips.innerHTML = "";

    if (this.spec.choices) {
      this.spec.choices.forEach((choice) => {
        const on         = values.includes(choice.value);
        const chip       = document.createElement("button");
        chip.type        = "button";
        chip.className   = "multivalue__choice" + (on ? " is-on" : "");
        chip.textContent = choice.label;
        chip.title       = `--${this.leaf.path} · ${choice.value}`;
        chip.setAttribute("aria-pressed", String(on));
        chip.addEventListener("click", () => this._toggleChoice(choice.value));
        this.chips.appendChild(chip);
      });
    } else {
      values.forEach((value) => {
        const chip        = document.createElement("span");
        chip.className    = "multivalue__chip";
        const label       = document.createElement("span");
        label.textContent = String(value);
        const remove      = document.createElement("button");
        remove.type       = "button";
        remove.className  = "multivalue__x";
        remove.innerHTML  = "&times;";
        remove.title      = "remove";
        remove.addEventListener("click", () => this._removeValue(value));
        chip.appendChild(label);
        chip.appendChild(remove);
        this.chips.appendChild(chip);
      });
    }

    this.count.textContent = values.length
      ? `${values.length} value${values.length === 1 ? "" : "s"} · ${PythonLiteral.render(values)}`
      : (this.spec.empty || "select at least one value");
    this.count.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }

  build() {
    this.el           = document.createElement("div");
    this.el.className = "picker multivalue";

    const chips     = document.createElement("div");
    chips.className = "multivalue__chips";
    this.chips      = chips;
    this.el.appendChild(chips);

    if (!this.spec.choices) {
      const entry       = document.createElement("input");
      entry.className   = "cfg-edit__input multivalue__entry";
      entry.type        = "text";
      entry.spellcheck  = false;
      entry.placeholder = this.spec.placeholder || "add value, Enter";
      entry.addEventListener("keydown", (event) => this._onKey(event));
      entry.addEventListener("blur", () => this._commitEntry());
      this.input = entry;
      this.el.appendChild(entry);
    }

    const note     = document.createElement("p");
    note.className = "picker__note";
    this.count     = note;
    this.el.appendChild(note);

    this._paint();
    return { el: this.el, input: this.input || this.el, reset: this.reset };
  }
}


class ConfigForm {
  constructor() {
    this.dirty         = {};
    this.controls      = {};
    this.states        = [];
    this.gates         = [];
    this.sections      = [];
    this.byPath        = new Map();
    this.activeSection = null;
    this.query         = "";
    this._section      = null;
    this.config        = null;
    this.modelFamilies = null;
    this.layoutEl      = null;
    this.nomatchEl     = null;
    this.countEl       = null;
  }

  _buildToolbar(cfg) {
    const bar     = document.createElement("div");
    bar.className = "cfg-toolbar";

    const search       = document.createElement("input");
    search.className   = "cfg-search";
    search.type        = "search";
    search.placeholder = `Filter ${cfg.leaves.length} fields...`;
    search.spellcheck  = false;
    search.addEventListener("input", () => {
      this.query = search.value.trim().toLowerCase();
      this._applyVisibility();
    });

    const count     = document.createElement("span");
    count.className = "cfg-toolbar__count";
    this.countEl    = count;

    const reset       = document.createElement("button");
    reset.className   = "btn btn--mini";
    reset.textContent = "Reset all";
    reset.addEventListener("click", () => this._resetAll());

    bar.appendChild(search);
    bar.appendChild(count);
    bar.appendChild(reset);
    return bar;
  }

  _buildPins(pinned) {
    const panel     = document.createElement("section");
    panel.className = "launch-pins";

    const head     = document.createElement("header");
    head.className = "launch-pins__head";
    head.innerHTML = `<h3 class="launch-pins__name">Run essentials</h3><span class="launch-pins__hint">check these before every launch</span>`;

    const grid     = document.createElement("div");
    grid.className = "launch-pins__grid";
    pinned.forEach((leaf) => grid.appendChild(this._buildRow(leaf, "essentials", true)));

    panel.appendChild(head);
    panel.appendChild(grid);
    return panel;
  }

  _buildPanel(panel) {
    if (panel.kind === "hidden")  return null;
    if (panel.kind === "special") return this._buildSpecialPanel(panel);
    return this._buildFieldsPanel(panel);
  }

  _buildSpecialPanel(panel) {
    const leaf = this.byPath.get(panel.fields[0]);

    if (panel.panel === "model_card") {
      if (!leaf || !this.modelFamilies || !this.modelFamilies.length) return this._buildPathsPanel("Model", panel.fields);
      return new window.ModelPickPanel(this, leaf).build();
    }

    if (panel.panel === "model_toggle") {
      if (!leaf || !this.modelFamilies || !this.modelFamilies.length) return this._buildPathsPanel("Models in run", panel.fields);
      return new window.ModelTogglePanel(this, leaf).build();
    }

    return this._buildPathsPanel(panel.panel, panel.fields);
  }

  _buildPathsPanel(title, paths) {
    return this._buildFieldsPanel({ kind: "fields", title, groups: [{ title: null, fields: paths.map((path) => ({ path })) }] });
  }

  _buildFieldsPanel(panel) {
    const el       = document.createElement("section");
    el.className   = "cfg-panel";
    el.dataset.cols = String(Math.min(panel.groups.length, 4));

    if (panel.title) {
      const head     = document.createElement("header");
      head.className = "cfg-panel__head";
      head.innerHTML = `<h4 class="cfg-panel__name">${panel.title}</h4>`;
      el.appendChild(head);
    }

    el.appendChild(this._buildGroups(panel.groups));
    return el;
  }

  _buildGroups(groups) {
    const body     = document.createElement("div");
    body.className = "cfg-panel__groups";

    groups.forEach((group) => {
      const groupEl     = document.createElement("div");
      groupEl.className = "field-group";

      if (group.title) {
        const heading       = document.createElement("div");
        heading.className   = "field-group__title";
        heading.textContent = group.title;
        groupEl.appendChild(heading);
      }

      const inner     = document.createElement("div");
      inner.className = "field-group__grid";
      group.fields.forEach((entry) => this._buildEntry(entry, inner));

      groupEl.appendChild(inner);
      body.appendChild(groupEl);
    });

    return body;
  }

  _buildEntry(entry, host) {
    if (!entry.gate) {
      host.appendChild(this._buildRow(this.byPath.get(entry.path), this._section));
      return;
    }

    const lead     = this.byPath.get(entry.gate);
    const cell     = document.createElement("div");
    cell.className = "band-block";

    cell.appendChild(this._buildGateRow(lead, this._gateLabel(this._shortName(lead))));

    const gatedRows = [];
    entry.fields.forEach((sub) => {
      const row = this._buildRow(this.byPath.get(sub.path), this._section);
      row.classList.add("cfg-edit__row--dependent");
      cell.appendChild(row);
      gatedRows.push(this.states[this.states.length - 1]);
    });

    this.gates.push({ leaf: lead, states: gatedRows });
    host.appendChild(cell);
  }

  _gateLabel(short) {
    if (short.startsWith("use_")) return short.slice(4);
    if (short !== "enabled" && short.endsWith("_enabled")) return short.slice(0, -"_enabled".length);
    return short;
  }

  _shortName(leaf) {
    return leaf.section ? leaf.path.slice(leaf.section.length + 1) : leaf.path;
  }

  _buildRow(leaf, sectionKey, pinned = false) {
    const short = this._shortName(leaf);

    const row     = document.createElement("div");
    row.className = "cfg-edit__row";
    row.title     = `--${leaf.path}`;

    const label       = document.createElement("div");
    label.className   = "cfg-edit__name";
    label.textContent = short;
    label.title       = `${leaf.type} · --${leaf.path}`;
    row.appendChild(label);

    let control;
    const spec    = leaf.editable ? this._widgetSpec(leaf) : null;
    const kind    = spec ? spec.kind : null;
    const choices = kind === "choice" ? spec.options : null;

    if (kind === "run" && window.RunPicker) {
      control = new window.RunPicker(this, leaf).build();
      row.classList.add("cfg-edit__row--wide");
    } else if (kind === "multi" && window.MultiValueField) {
      control = new window.MultiValueField(this, leaf, spec).build();
      row.classList.add("cfg-edit__row--board");
    } else if (choices) {
      control = this._choiceControl(leaf, choices);
      row.classList.add("cfg-edit__row--choice");
    } else if (!leaf.editable) {
      control = this._textControl(leaf);
      control.input.disabled = true;
      control.input.classList.add("is-locked");
      control.input.title = "not overridable from the command line";
    } else if (leaf.type === "bool") {
      control = this._switchControl(leaf);
      row.classList.add("cfg-edit__row--bool");
    } else if (leaf.type === "int" || leaf.type === "float") {
      control = new window.NumberField(this, leaf, short, kind === "number" ? spec : null).build();
      row.classList.add("cfg-edit__row--num");
    } else {
      control = this._textControl(leaf);
    }

    row.appendChild(control.el);
    this.controls[leaf.path] = { leaf, reset: control.reset, input: control.input };
    this.states.push({ leaf, row, sectionKey: sectionKey !== undefined ? sectionKey : this._section, pinned });
    return row;
  }

  _buildGateRow(lead, label) {
    const row     = document.createElement("div");
    row.className = "cfg-edit__row cfg-edit__row--bool cfg-edit__row--gate";
    row.title     = `--${lead.path}`;

    const name       = document.createElement("div");
    name.className   = "cfg-edit__name";
    name.textContent = label;
    row.appendChild(name);

    const toggle = this._switchControl(lead);
    row.appendChild(toggle.el);

    this.controls[lead.path] = { leaf: lead, reset: toggle.reset, input: toggle.input };
    this.states.push({ leaf: lead, row, sectionKey: this._section });
    return row;
  }

  _widgetSpec(leaf) {
    if (!this.config || !this.config.layout) return null;
    return this.config.layout.widgets[leaf.path] || null;
  }

  _effective(leaf) {
    return this.dirty[leaf.path] !== undefined ? this.dirty[leaf.path] : leaf.value;
  }

  _choiceControl(leaf, choices) {
    const select     = document.createElement("select");
    select.className = "cfg-edit__input picker__select";

    const current   = String(leaf.value);
    const effective = String(this._effective(leaf));
    const options   = [...new Set([current, effective, ...choices])];

    options.forEach((value) => {
      const option       = document.createElement("option");
      option.value       = value;
      option.textContent = value;
      select.appendChild(option);
    });

    select.value = effective;
    select.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);

    select.addEventListener("change", () => {
      select.classList.toggle("is-dirty", select.value !== leaf.value);
      this._setValue(leaf, select.value);
    });

    const reset = () => {
      select.value = String(this._effective(leaf));
      select.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
    };
    return { el: select, input: select, reset };
  }

  _textControl(leaf) {
    const input      = document.createElement("input");
    input.className  = "cfg-edit__input";
    input.value      = this._effective(leaf);
    input.spellcheck = false;
    input.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);

    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value);
      this._setValue(leaf, input.value);
    });

    const reset = () => {
      input.value = this._effective(leaf);
      input.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
    };
    return { el: input, input, reset };
  }

  _switchControl(leaf) {
    const toggle     = document.createElement("button");
    toggle.type      = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.innerHTML = `<span class="switch__knob"></span>`;

    const paint = () => {
      const on = this._effective(leaf) === "True";
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
      toggle.setAttribute("aria-checked", String(on));
    };

    toggle.addEventListener("click", () => {
      this._setValue(leaf, this._effective(leaf) === "True" ? "False" : "True");
      paint();
    });
    paint();

    return { el: toggle, input: toggle, reset: () => paint() };
  }

  _setValue(leaf, value) {
    if (value !== leaf.value) this.dirty[leaf.path] = value;
    else                      delete this.dirty[leaf.path];
    this._refresh();
  }

  _resetField(path) {
    const control = this.controls[path];
    delete this.dirty[path];
    if (control) control.reset();
    this._refresh();
  }

  _resetAll() {
    this.dirty = {};
    Object.values(this.controls).forEach((control) => control.reset());
    this._refresh();
  }

  _navigate(key) {
    this._setActiveSection(key);
  }

  _setActiveSection(key) {
    const target = this.sections.find((section) => section.key === key);
    this.activeSection = (target || this.sections[0]).key;

    this.sections.forEach((section) => {
      if (section.navBtn) section.navBtn.classList.toggle("is-active", section.key === this.activeSection);
    });
    this._applyVisibility();
  }

  _refreshGates() {
    this.states.forEach(({ row }) => {
      delete row.dataset.gated;
    });

    this.gates.forEach((gate) => {
      if (this._effective(gate.leaf) !== "True") gate.states.forEach(({ row }) => (row.dataset.gated = "1"));
    });

    this._applyVisibility();
  }

  _applyVisibility() {
    const searching = Boolean(this.query);
    if (this.layoutEl) this.layoutEl.classList.toggle("is-searching", searching);

    this.states.forEach(({ leaf, row }) => {
      const matches = !searching || leaf.path.toLowerCase().includes(this.query);
      row.hidden = !matches || row.dataset.gated === "1";
    });

    let anyVisible = false;
    this.sections.forEach((section) => {
      const hasRows = this.states.some(({ row, sectionKey }) => sectionKey === section.key && !row.hidden);
      const single  = this.config && this.config.layout && this.config.layout.mode === "single";
      const show    = searching ? hasRows : (single || section.key === this.activeSection);

      section.el.hidden = !show;
      anyVisible = anyVisible || (show && (!searching || hasRows));
    });

    if (this.nomatchEl) this.nomatchEl.hidden = !searching || anyVisible;
  }

  _refreshBadges() {
    const counts = new Map();
    this.states.forEach(({ leaf, sectionKey }) => {
      if (this.dirty[leaf.path] !== undefined) counts.set(sectionKey, (counts.get(sectionKey) || 0) + 1);
    });

    this.sections.forEach((section) => {
      if (!section.badge) return;
      const total = counts.get(section.key) || 0;
      section.badge.hidden = total === 0;
      section.badge.textContent = total ? `${total}` : "";
    });
  }

  _renderLayout(host, cfg) {
    const layout = cfg.layout;
    this.byPath  = new Map(cfg.leaves.map((leaf) => [leaf.path, leaf]));

    const wrap     = document.createElement("div");
    wrap.className = "launch-layout";
    if (layout.mode === "single") wrap.classList.add("launch-layout--single");
    this.layoutEl  = wrap;

    const nav     = document.createElement("nav");
    nav.className = "secnav";
    nav.setAttribute("aria-label", "Configuration sections");

    const main     = document.createElement("div");
    main.className = "secmain";

    const declared = [];
    if (layout.essentials.length) declared.push({ key: "essentials", title: "Essentials", panels: null });
    layout.sections.forEach((section) => declared.push(section));

    declared.forEach((section) => {
      this._section = section.key;

      const el         = document.createElement("section");
      el.className     = "launch-section";
      el.dataset.section = section.key;

      const title       = document.createElement("h3");
      title.className   = "launch-section__title";
      title.textContent = section.title;
      el.appendChild(title);

      const body     = document.createElement("div");
      body.className = "launch-section__body";
      el.appendChild(body);

      if (section.panels === null) {
        body.appendChild(this._buildPins(layout.essentials.map((entry) => this.byPath.get(entry.path))));
      } else {
        section.panels.forEach((panel) => {
          const built = this._buildPanel(panel);
          if (built) body.appendChild(built);
        });
      }

      const record = { key: section.key, title: section.title, el, navBtn: null, badge: null };

      if (layout.mode === "sections") {
        const button     = document.createElement("button");
        button.type      = "button";
        button.className = "secnav__item";

        const badge     = document.createElement("span");
        badge.className = "edit-badge";
        badge.hidden    = true;

        button.innerHTML = `<span class="secnav__name">${section.title}</span>`;
        button.appendChild(badge);
        button.addEventListener("click", () => this._navigate(record.key));

        record.navBtn = button;
        record.badge  = badge;
        nav.appendChild(button);
      }

      main.appendChild(el);
      this.sections.push(record);
    });

    wrap.appendChild(main);
    if (layout.mode === "sections") wrap.appendChild(nav);

    const empty       = document.createElement("p");
    empty.className   = "cfg-note launch-nomatch";
    empty.textContent = "No fields match this filter.";
    empty.hidden      = true;
    this.nomatchEl    = empty;
    main.appendChild(empty);

    host.appendChild(wrap);
    this._setActiveSection(this.activeSection || this.sections[0].key);
  }
}

window.PythonLiteral    = PythonLiteral;
window.LaunchWidgetDom  = LaunchWidgetDom;
window.ModelPickPanel   = ModelPickPanel;
window.ModelTogglePanel = ModelTogglePanel;
window.RunPicker        = RunPicker;
window.NumberField      = NumberField;
window.MultiValueField  = MultiValueField;
window.ConfigForm       = ConfigForm;
