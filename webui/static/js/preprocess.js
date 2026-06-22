"use strict";

const PP_FIELD_RANGES = {
  scale_factor: [0, 2, 0.01],
  detection_efficiency: [0, 0.2, 0.001],
  active_fraction: [0.5, 1, 0.001],
  median_factor: [1, 100, 0.5],
  neighbor_count: [1, 16, 1],
  min_events: [1, 500, 1],
  sensor_dropout_probability: [0, 0.5, 0.005],
  spurious_activation_probability: [0, 0.2, 0.002],
  spurious_activation_max_light: [0, 5, 0.05],
  light_noise_sigma: [0, 1, 0.01],
  photon_thinning_survival: [0, 1, 0.01],
  gain_jitter_sigma: [0, 0.5, 0.005],
};

const PP_INT_FIELDS = new Set(["efficiency_seed", "seed", "neighbor_count", "min_events"]);

const PP_GROUPS = [
  { key: "scaling", title: "Light scaling" },
  { key: "efficiency", title: "Binomial efficiency" },
  { key: "outlier", title: "Outlier cleaning" },
  { key: "augmentation", title: "Sensor augmentation" },
];

const PP_LABELS = {
  scale_factor: "scale factor",
  detection_efficiency: "detection efficiency",
  efficiency_seed: "efficiency seed",
  enabled: "enabled",
  active_fraction: "active fraction",
  median_factor: "median factor",
  neighbor_count: "neighbour count",
  min_events: "min events",
  seed: "seed",
  sensor_dropout_enabled: "sensor dropout",
  sensor_dropout_probability: "dropout probability",
  spurious_activation_enabled: "spurious activation",
  spurious_activation_probability: "spurious probability",
  spurious_activation_max_light: "spurious max light",
  light_noise_enabled: "light noise",
  light_noise_sigma: "noise sigma",
  light_noise_mode: "noise mode",
  photon_thinning_enabled: "photon thinning",
  photon_thinning_survival: "thinning survival",
  gain_jitter_enabled: "gain jitter",
  gain_jitter_sigma: "gain sigma",
};

class StageViewer {
  constructor(canvas, shared) {
    this.canvas = canvas;
    this.shared = shared;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1e6);

    this.points = null;
    this.box = null;

    this.full = null;
    this.pointSize = 1;
    this.detectorExt = [1, 1, 1];
    this.wallFlags = WALL_DEFS.map(() => true);

    this._wireControls();
  }

  _wireControls() {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    this.canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      this.shared.theta -= (event.clientX - lastX) * 0.008;
      this.shared.phi -= (event.clientY - lastY) * 0.008;
      this.shared.phi = Math.max(0.05, Math.min(Math.PI - 0.05, this.shared.phi));
      lastX = event.clientX;
      lastY = event.clientY;
    });
    const release = (event) => {
      dragging = false;
      try { this.canvas.releasePointerCapture(event.pointerId); } catch (e) {}
    };
    this.canvas.addEventListener("pointerup", release);
    this.canvas.addEventListener("pointercancel", release);
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.shared.radius *= 1 + Math.sign(event.deltaY) * 0.12;
      this.shared.radius = Math.max(this.shared.minRadius, Math.min(this.shared.maxRadius, this.shared.radius));
    }, { passive: false });
  }

  setStage(stage, extent) {
    const positions = stage.positions;
    const light = stage.light;

    let lightMin = Infinity;
    let lightMax = -Infinity;
    light.forEach((value) => { if (value < lightMin) lightMin = value; if (value > lightMax) lightMax = value; });

    const positionArray = new Float32Array(positions.length * 3);
    const colorArray = new Float32Array(positions.length * 3);
    const wallArray = new Int8Array(positions.length);
    positions.forEach((point, i) => {
      positionArray[i * 3] = point[0];
      positionArray[i * 3 + 1] = point[1];
      positionArray[i * 3 + 2] = point[2];
      const rgb = window.colorForLight(light[i], lightMin, lightMax);
      colorArray[i * 3] = rgb[0];
      colorArray[i * 3 + 1] = rgb[1];
      colorArray[i * 3 + 2] = rgb[2];
      wallArray[i] = window.classifyWall(point, this.detectorExt);
    });

    this.full = { positions: positionArray, colors: colorArray, wall: wallArray, count: positions.length };
    this.pointSize = extent * 0.02;
    this._rebuildPoints();
  }

  setDetectorBounds(min, max) {
    this.detectorExt = window.wallExtents(min, max);
  }

  setWallFlags(flags) {
    this.wallFlags = flags;
    this._rebuildPoints();
  }

  _rebuildPoints() {
    if (!this.full) return;

    const flags = this.wallFlags;
    const count = this.full.count;

    let visible = 0;
    for (let i = 0; i < count; i++) if (flags[this.full.wall[i]]) visible++;

    const positionArray = new Float32Array(visible * 3);
    const colorArray = new Float32Array(visible * 3);
    let cursor = 0;
    for (let i = 0; i < count; i++) {
      if (!flags[this.full.wall[i]]) continue;
      positionArray[cursor * 3]     = this.full.positions[i * 3];
      positionArray[cursor * 3 + 1] = this.full.positions[i * 3 + 1];
      positionArray[cursor * 3 + 2] = this.full.positions[i * 3 + 2];
      colorArray[cursor * 3]     = this.full.colors[i * 3];
      colorArray[cursor * 3 + 1] = this.full.colors[i * 3 + 1];
      colorArray[cursor * 3 + 2] = this.full.colors[i * 3 + 2];
      cursor++;
    }

    if (this.points) {
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positionArray, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
    const material = new THREE.PointsMaterial({ size: this.pointSize, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.92 });
    this.points = new THREE.Points(geometry, material);
    this.scene.add(this.points);
  }

  setBox(min, max) {
    if (this.box) {
      this.scene.remove(this.box);
      this.box.geometry.dispose();
      this.box.material.dispose();
    }
    const size = new THREE.Vector3(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const center = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const geometry = new THREE.BoxGeometry(Math.max(size.x, 1e-3), Math.max(size.y, 1e-3), Math.max(size.z, 1e-3));
    const edges = new THREE.EdgesGeometry(geometry);
    this.box = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xbcc5cf, transparent: true, opacity: 0.28 }));
    this.box.position.copy(center);
    geometry.dispose();
    this.scene.add(this.box);
  }

  _resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    if (this._lastW === width && this._lastH === height) return;
    this._lastW = width;
    this._lastH = height;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this._resize();
    const shared = this.shared;
    const sinPhi = Math.sin(shared.phi);
    this.camera.position.set(
      shared.target.x + shared.radius * sinPhi * Math.sin(shared.theta),
      shared.target.y + shared.radius * Math.cos(shared.phi),
      shared.target.z + shared.radius * sinPhi * Math.cos(shared.theta)
    );
    this.camera.lookAt(shared.target);
    this.renderer.render(this.scene, this.camera);
  }
}

class PreprocessingPanel {
  constructor(refs) {
    this.refs = refs;
    this.entered = false;
    this.polling = false;
    this.viewers = [];
    this.config = null;
    this.gt = [];
    this.bounds = null;
    this.selectedIndex = null;
    this.pendingIndex = null;
    this.fetching = false;
    this.running = false;

    this.shared = {
      theta: Math.PI * 0.25,
      phi: Math.PI * 0.35,
      radius: 10,
      minRadius: 0.01,
      maxRadius: 1e6,
      target: new THREE.Vector3(0, 0, 0),
    };

    this._loop = this._loop.bind(this);

    this.wallToggle = new window.WallToggle(this.refs.walls, (flags) => this.viewers.forEach((viewer) => viewer.setWallFlags(flags)));
  }

  async enter() {
    if (!this.viewers.length) this._buildViewers();
    this._start();
    if (!this.entered) {
      this.entered = true;
      await this._load();
    }
  }

  leave() {
    this.running = false;
  }

  _buildViewers() {
    this.refs.stages.querySelectorAll("canvas.pp-canvas").forEach((canvas) => {
      this.viewers.push(new StageViewer(canvas, this.shared));
    });
  }

  _start() {
    if (this.running) return;
    this.running = true;
    requestAnimationFrame(this._loop);
  }

  _loop() {
    if (!this.running) return;
    this.viewers.forEach((viewer) => viewer.render());
    requestAnimationFrame(this._loop);
  }

  async _load() {
    this.refs.hint.hidden = true;
    this.refs.progress.hidden = false;
    this.refs.progressLabel.textContent = "requesting raw-light source";

    const result = await window.apiPost("/api/preprocess/load", {});
    if (!result.ok) {
      this.refs.progress.hidden = true;
      this.refs.hint.hidden = false;
      this.refs.hint.textContent = result.error || "Load failed.";
      return;
    }
    await this._poll();
  }

  async _poll() {
    this.polling = true;
    while (true) {
      let status;
      try {
        status = await window.apiGet("/api/preprocess/status");
      } catch (e) {
        this._failLoad("Backend unreachable.");
        break;
      }

      if (status.state === "loading") {
        this.refs.progressLabel.textContent = status.stage || "loading";
        await new Promise((resolve) => setTimeout(resolve, 700));
        continue;
      }
      if (status.state === "ready") {
        this.refs.progress.hidden = true;
        await this._ready();
        break;
      }
      this._failLoad(status.error || "Load failed.");
      break;
    }
    this.polling = false;
  }

  _failLoad(message) {
    this.refs.progress.hidden = true;
    this.refs.hint.hidden = false;
    this.refs.hint.textContent = message;
  }

  async _ready() {
    const data = await window.apiGet("/api/preprocess/events");
    if (!data || !data.ok) { this._failLoad((data && data.error) || "Could not read events."); return; }

    this.gt = data.gt;
    this.bounds = { min: data.bounds_min, max: data.bounds_max };
    this.config = data.defaults;

    this.viewers.forEach((viewer) => {
      viewer.setDetectorBounds(data.detector_min, data.detector_max);
      viewer.setWallFlags(this.wallToggle.flags);
    });

    this.refs.stage.hidden = false;
    this._buildControls();
    this._buildTargetSliders();

    const eventCard = this.refs.target.closest(".ev-card");
    if (eventCard) this._wireCollapse(eventCard, eventCard.querySelector(".ev-card__cap"));

    const wallsCard = this.refs.walls.closest(".ev-card");
    if (wallsCard) this._wireCollapse(wallsCard, wallsCard.querySelector(".ev-card__cap"));

    const center = [
      (this.bounds.min[0] + this.bounds.max[0]) / 2,
      (this.bounds.min[1] + this.bounds.max[1]) / 2,
      (this.bounds.min[2] + this.bounds.max[2]) / 2,
    ];
    this._setTargetValues(center);
    this._snap(center);
  }

  _buildControls() {
    this.refs.controls.innerHTML = "";

    PP_GROUPS.forEach((group) => {
      const values = this.config[group.key];
      const card = document.createElement("div");
      card.className = "pp-card";

      const cap = document.createElement("div");
      cap.className = "pp-card__cap";
      cap.textContent = group.title;
      card.appendChild(cap);

      const body = document.createElement("div");
      body.className = "pp-card__body";
      Object.keys(values).forEach((field) => {
        body.appendChild(this._buildControl(group.key, field, values[field]));
      });
      card.appendChild(body);

      this._wireCollapse(card, cap);
      this.refs.controls.appendChild(card);
    });
  }

  _buildControl(groupKey, field, value) {
    const row = document.createElement("div");
    row.className = "pp-control";

    const label = document.createElement("label");
    label.textContent = PP_LABELS[field] || field;
    row.appendChild(label);

    if (typeof value === "boolean") {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = value;
      input.addEventListener("change", () => { this.config[groupKey][field] = input.checked; this._queuePreview(); });
      row.classList.add("pp-control--toggle");
      row.appendChild(input);
      return row;
    }

    if (field === "light_noise_mode") {
      const select = document.createElement("select");
      ["multiplicative", "additive"].forEach((mode) => {
        const option = document.createElement("option");
        option.value = mode;
        option.textContent = mode;
        option.selected = mode === value;
        select.appendChild(option);
      });
      select.addEventListener("change", () => { this.config[groupKey][field] = select.value; this._queuePreview(); });
      row.appendChild(select);
      return row;
    }

    if (PP_INT_FIELDS.has(field) && !PP_FIELD_RANGES[field]) {
      const input = document.createElement("input");
      input.type = "number";
      input.value = value;
      input.step = 1;
      input.addEventListener("change", () => { this.config[groupKey][field] = parseInt(input.value, 10) || 0; this._queuePreview(); });
      row.appendChild(input);
      return row;
    }

    const range = PP_FIELD_RANGES[field] || [0, 1, 0.01];
    const input = document.createElement("input");
    input.type = "range";
    input.min = range[0];
    input.max = range[1];
    input.step = range[2];
    input.value = value;
    const output = document.createElement("output");
    output.textContent = this._fmt(value);
    input.addEventListener("input", () => {
      const parsed = PP_INT_FIELDS.has(field) ? parseInt(input.value, 10) : parseFloat(input.value);
      this.config[groupKey][field] = parsed;
      output.textContent = this._fmt(parsed);
      this._queuePreview();
    });
    row.appendChild(output);
    row.appendChild(input);
    return row;
  }

  _wireCollapse(card, cap) {
    cap.addEventListener("click", () => card.classList.toggle("is-collapsed"));
  }

  _buildTargetSliders() {
    const axes = ["x", "y", "z"];
    this.refs.target.innerHTML = "";
    this.targetInputs = {};

    axes.forEach((axis, index) => {
      const min = this.bounds.min[index];
      const max = this.bounds.max[index];
      const step = (max - min) / 400 || 0.001;

      const row = document.createElement("div");
      row.className = "ev-slider";
      row.innerHTML =
        `<label>${axis}</label>` +
        `<input type="range" min="${min}" max="${max}" step="${step}" value="${(min + max) / 2}" data-axis="${index}" />` +
        `<output data-axis="${index}">0</output>`;
      this.refs.target.appendChild(row);

      const input = row.querySelector("input");
      this.targetInputs[index] = input;
      input.addEventListener("input", () => this._onTarget());
    });
  }

  _setTargetValues(point) {
    for (let index = 0; index < 3; index++) {
      this.targetInputs[index].value = point[index];
      this.refs.target.querySelector(`output[data-axis="${index}"]`).textContent = this._fmt(point[index]);
    }
  }

  _onTarget() {
    const point = [0, 1, 2].map((index) => {
      const value = parseFloat(this.targetInputs[index].value);
      this.refs.target.querySelector(`output[data-axis="${index}"]`).textContent = this._fmt(value);
      return value;
    });
    this._snap(point);
  }

  _snap(point) {
    if (!this.gt.length) return;

    let best = 0;
    let bestDistance = Infinity;
    for (let i = 0; i < this.gt.length; i++) {
      const dx = this.gt[i][0] - point[0];
      const dy = this.gt[i][1] - point[1];
      const dz = this.gt[i][2] - point[2];
      const distance = dx * dx + dy * dy + dz * dz;
      if (distance < bestDistance) { bestDistance = distance; best = i; }
    }

    this.refs.nearest.textContent = `· snapped ${this._fmt(Math.sqrt(bestDistance))} cm away`;
    if (best !== this.selectedIndex) {
      this.selectedIndex = best;
      this.recenter = true;
      this._queuePreview();
    }
  }

  _queuePreview() {
    this.pendingIndex = this.selectedIndex;
    clearTimeout(this.debounce);
    this.debounce = setTimeout(() => this._pump(), 120);
  }

  async _pump() {
    if (this.fetching || this.selectedIndex === null) return;
    this.fetching = true;
    while (this.pendingIndex !== null) {
      const index = this.pendingIndex;
      this.pendingIndex = null;
      await this._fetchPreview(index);
    }
    this.fetching = false;
  }

  async _fetchPreview(index) {
    const data = await window.apiPost("/api/preprocess/preview", { index, config: this.config });
    if (!data || !data.ok) { if (data && data.error) window.toast(data.error, "warn"); return; }
    this._render(data);
  }

  _render(data) {
    const min = data.bounds_min;
    const max = data.bounds_max;
    const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1e-3);

    if (this.recenter) {
      this.recenter = false;
      this.shared.target.set((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
      this.shared.radius = extent * 0.95;
      this.shared.minRadius = extent * 0.05;
      this.shared.maxRadius = extent * 12;
    }

    data.stages.forEach((stage, i) => {
      const viewer = this.viewers[i];
      if (!viewer) return;
      viewer.setStage(stage, extent);
      viewer.setBox(min, max);
      const readout = this.refs.stages.querySelectorAll(".pp-stage__meta")[i];
      if (readout) readout.textContent = `${stage.n_active} active · ${this._fmt(stage.total_light)} light`;
    });

    this.refs.readout.textContent = `octant ${data.octant} · base event ${data.base_event_id} · vertex ${data.gt.map((v) => this._fmt(v)).join(" · ")}`;
  }

  _fmt(value) {
    if (!Number.isFinite(value)) return "–";
    const abs = Math.abs(value);
    if (abs >= 1000) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(1);
    if (abs >= 0.01 || abs === 0) return value.toFixed(2);
    return value.toExponential(1);
  }
}

window.StageViewer = StageViewer;
window.PreprocessingPanel = PreprocessingPanel;
