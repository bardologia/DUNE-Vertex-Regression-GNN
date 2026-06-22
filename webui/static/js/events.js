"use strict";

function colorForLight(value, min, max) {
  const t = max > min ? (value - min) / (max - min) : 0.5;
  const stops = [
    [0.18, 0.30, 0.78],
    [0.16, 0.72, 0.86],
    [0.62, 0.84, 0.30],
    [0.98, 0.86, 0.24],
  ];
  const scaled = t * (stops.length - 1);
  const index = Math.max(0, Math.min(stops.length - 2, Math.floor(scaled)));
  const frac = scaled - index;
  const a = stops[index];
  const b = stops[index + 1];
  return [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac, a[2] + (b[2] - a[2]) * frac];
}

class EventViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1e6);

    this.target = new THREE.Vector3(0, 0, 0);
    this.radius = 10;
    this.theta = Math.PI * 0.25;
    this.phi = Math.PI * 0.35;
    this.minRadius = 0.01;
    this.maxRadius = 1e6;

    this.targetGoal = new THREE.Vector3(0, 0, 0);
    this.radiusGoal = 10;
    this.frameEase = 0.14;

    this.gtGoal = new THREE.Vector3(0, 0, 0);
    this.predGoal = new THREE.Vector3(0, 0, 0);
    this.pointsFade = 0;

    this.points = null;
    this.box = null;
    this.axes = null;
    this.running = false;

    this.full = null;
    this.pointSize = 1;
    this.detectorExt = [1, 1, 1];
    this.wallFlags = WALL_DEFS.map(() => true);

    this._buildMarkers();
    this._wireControls();
    this._loop = this._loop.bind(this);
  }

  _buildMarkers() {
    const makeDot = (color) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
      const material = new THREE.PointsMaterial({ color, size: 1, sizeAttenuation: true, transparent: true, opacity: 0.95, depthWrite: false });
      const dot = new THREE.Points(geometry, material);
      dot.visible = false;
      this.scene.add(dot);
      return dot;
    };

    this.gtMarker = makeDot(0x18c08f);
    this.predMarker = makeDot(0xff7a3c);

    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    this.errorLine = new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color: 0xff7a3c, transparent: true, opacity: 0.45 }));
    this.errorLine.visible = false;
    this.scene.add(this.errorLine);
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
      this.theta -= (event.clientX - lastX) * 0.008;
      this.phi -= (event.clientY - lastY) * 0.008;
      this.phi = Math.max(0.05, Math.min(Math.PI - 0.05, this.phi));
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
      this.radius *= 1 + Math.sign(event.deltaY) * 0.12;
      this.radius = Math.max(this.minRadius, Math.min(this.maxRadius, this.radius));
    }, { passive: false });
  }

  start() {
    if (this.running) return;
    this.running = true;
    requestAnimationFrame(this._loop);
  }

  stop() {
    this.running = false;
  }

  _resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    if (this.canvas.width === width && this.canvas.height === height && this._lastW === width && this._lastH === height) return;

    this._lastW = width;
    this._lastH = height;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  _colorForLight(value, min, max) {
    return colorForLight(value, min, max);
  }

  setEvent(detail, hasPrediction) {
    const firstEvent = this.points === null;
    const positions = detail.sensors.positions;
    const light = detail.sensors.light;
    const gt = detail.gt;
    const pred = hasPrediction ? detail.pred : null;

    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    const include = (point) => {
      for (let axis = 0; axis < 3; axis++) {
        if (point[axis] < min[axis]) min[axis] = point[axis];
        if (point[axis] > max[axis]) max[axis] = point[axis];
      }
    };
    positions.forEach(include);
    include(gt);
    if (hasPrediction) include(pred);
    if (!positions.length) { include([0, 0, 0]); }

    const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1e-3);

    this.targetGoal.set(center[0], center[1], center[2]);
    this.radiusGoal = extent * 1.9;
    this.minRadius = extent * 0.05;
    this.maxRadius = extent * 12;
    if (firstEvent) {
      this.target.copy(this.targetGoal);
      this.radius = this.radiusGoal;
    }

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
      const rgb = this._colorForLight(light[i], lightMin, lightMax);
      colorArray[i * 3] = rgb[0];
      colorArray[i * 3 + 1] = rgb[1];
      colorArray[i * 3 + 2] = rgb[2];
      wallArray[i] = window.classifyWall(point, this.detectorExt);
    });

    this.full = { positions: positionArray, colors: colorArray, wall: wallArray, count: positions.length };
    this.pointSize = extent * 0.018;
    this._rebuildPoints();
    this.pointsFade = firstEvent ? 1 : 0;

    const markerSize = extent * 0.05;
    this.gtMarker.material.size = markerSize;
    this.predMarker.material.size = markerSize;

    this.gtGoal.set(gt[0], gt[1], gt[2]);
    this.gtMarker.visible = true;
    if (firstEvent) this.gtMarker.position.copy(this.gtGoal);

    if (hasPrediction) {
      this.predGoal.set(pred[0], pred[1], pred[2]);
      this.predMarker.visible = true;
      if (firstEvent) this.predMarker.position.copy(this.predGoal);
      this.errorLine.visible = true;
    } else {
      this.predMarker.visible = false;
      this.errorLine.visible = false;
    }

    this._buildBox(min, max);
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

  _buildBox(min, max) {
    if (this.box) {
      this.scene.remove(this.box);
      this.box.geometry.dispose();
      this.box.material.dispose();
    }
    if (this.axes) this.scene.remove(this.axes);

    const size = new THREE.Vector3(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const center = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    const geometry = new THREE.BoxGeometry(Math.max(size.x, 1e-3), Math.max(size.y, 1e-3), Math.max(size.z, 1e-3));
    const edges = new THREE.EdgesGeometry(geometry);
    this.box = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xbcc5cf, transparent: true, opacity: 0.4 }));
    this.box.position.copy(center);
    geometry.dispose();
    this.scene.add(this.box);

    this.axes = new THREE.AxesHelper(Math.max(size.x, size.y, size.z) * 0.55);
    this.axes.position.set(min[0], min[1], min[2]);
    this.scene.add(this.axes);
  }

  _tween() {
    const ease = this.frameEase;
    this.target.lerp(this.targetGoal, ease);
    this.radius += (this.radiusGoal - this.radius) * ease;

    if (this.gtMarker.visible) this.gtMarker.position.lerp(this.gtGoal, ease);
    if (this.predMarker.visible) this.predMarker.position.lerp(this.predGoal, ease);

    if (this.errorLine.visible) {
      const linePositions = this.errorLine.geometry.getAttribute("position");
      linePositions.setXYZ(0, this.gtMarker.position.x, this.gtMarker.position.y, this.gtMarker.position.z);
      linePositions.setXYZ(1, this.predMarker.position.x, this.predMarker.position.y, this.predMarker.position.z);
      linePositions.needsUpdate = true;
    }

    if (this.points) {
      this.pointsFade += (1 - this.pointsFade) * ease;
      this.points.material.opacity = 0.92 * this.pointsFade;
    }
  }

  _loop() {
    if (!this.running) return;
    this._resize();
    this._tween();

    const sinPhi = Math.sin(this.phi);
    this.camera.position.set(
      this.target.x + this.radius * sinPhi * Math.sin(this.theta),
      this.target.y + this.radius * Math.cos(this.phi),
      this.target.z + this.radius * sinPhi * Math.cos(this.theta)
    );
    this.camera.lookAt(this.target);
    this.renderer.render(this.scene, this.camera);

    requestAnimationFrame(this._loop);
  }
}

class EventExplorerPanel {
  constructor(refs) {
    this.refs = refs;
    this.datasets = [];
    this.runs = [];
    this.currentKind = null;
    this.currentName = null;
    this.currentSplit = "test";
    this.hasPrediction = false;
    this.meta = null;
    this.gt = [];
    this.error = [];
    this.nActive = [];
    this.selectedIndex = null;
    this.entered = false;
    this.polling = false;
    this.pendingIndex = null;
    this.fetching = false;
    this.viewer = null;

    this.refs.splits.querySelectorAll(".ev-split").forEach((button) => {
      button.addEventListener("click", () => this._setSplit(button.dataset.split));
    });

    this.wallToggle = new window.WallToggle(this.refs.walls, (flags) => { if (this.viewer) this.viewer.setWallFlags(flags); });
  }

  async enter() {
    if (!this.viewer) {
      this.viewer = new EventViewer(this.refs.canvas);
      this.viewer.setWallFlags(this.wallToggle.flags);
    }
    this.viewer.start();

    if (!this.entered) {
      this.entered = true;
      await this._loadSources();
    }
  }

  leave() {
    if (this.viewer) this.viewer.stop();
  }

  async _loadSources() {
    const data = await window.apiGet("/api/events/sources");
    this.datasets = (data && data.datasets) || [];
    this.runs = (data && data.runs) || [];
    this._renderSources();
  }

  _renderSources() {
    this.refs.sources.innerHTML = "";
    this.datasets.forEach((dataset) => {
      const isActive = this.currentKind === "dataset" && this.currentName === dataset.name;
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "ev-source" + (isActive ? " is-active" : "");
      pill.innerHTML =
        `<span class="ev-run__name">${window.escapeHtml(dataset.label)}</span>` +
        `<span class="ev-run__model">${dataset.cached ? "cached" : "build on open"}</span>`;
      pill.addEventListener("click", () => this._selectDataset(dataset.name));
      this.refs.sources.appendChild(pill);
    });

    this.refs.runs.innerHTML = "";
    this.runs.forEach((run) => {
      const isActive = this.currentKind === "run" && this.currentName === run.run;
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "ev-run" + (isActive ? " is-active" : "");
      pill.innerHTML =
        `<span class="ev-run__name">${window.escapeHtml(run.run)}</span>` +
        `<span class="ev-run__model">${window.escapeHtml(run.model)}</span>`;
      pill.addEventListener("click", () => this._selectRun(run.run));
      this.refs.runs.appendChild(pill);
    });
  }

  _selectDataset(name) {
    if (this.polling) { window.toast("A source is still loading.", "warn"); return; }
    this.currentKind = "dataset";
    this.currentName = name;
    this.currentSplit = "all";
    this.refs.splits.hidden = true;
    this._renderSources();
    this._load();
  }

  _selectRun(run) {
    if (this.polling) { window.toast("A source is still loading.", "warn"); return; }
    this.currentKind = "run";
    this.currentName = run;
    if (this.currentSplit === "all") this.currentSplit = "test";
    this.refs.splits.hidden = false;
    this._renderSources();
    this._load();
  }

  _setSplit(split) {
    if (this.currentKind !== "run" || split === this.currentSplit) return;
    this.currentSplit = split;
    this.refs.splits.querySelectorAll(".ev-split").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.split === split);
    });
    if (!this.polling) this._load();
  }

  async _load() {
    if (!this.currentName) return;

    this.refs.stage.hidden = true;
    this.refs.hint.hidden = true;
    this.refs.progress.hidden = false;
    this._setProgress("requesting load");

    const result = await window.apiPost("/api/events/load", { kind: this.currentKind, name: this.currentName, split: this.currentSplit });
    if (!result.ok) {
      this.refs.progress.hidden = true;
      this.refs.hint.hidden = false;
      this.refs.hint.textContent = result.error || "Load failed.";
      return;
    }
    await this._poll();
  }

  _setProgress(stage) {
    this.refs.progressFill.classList.add("is-indeterminate");
    this.refs.progressLabel.textContent = stage;
  }

  async _poll() {
    this.polling = true;
    const kind = this.currentKind;
    const name = this.currentName;
    const split = this.currentSplit;

    while (true) {
      let status;
      try {
        status = await window.apiGet("/api/events/status");
      } catch (e) {
        this._failLoad("Backend unreachable.");
        break;
      }

      if (status.kind !== kind || status.name !== name || status.split !== split) {
        this.refs.progress.hidden = true;
        break;
      }

      if (status.state === "loading") {
        this._setProgress(status.stage || "loading");
        await new Promise((resolve) => setTimeout(resolve, 600));
        continue;
      }

      if (status.state === "ready") {
        this.refs.progress.hidden = true;
        await this._fetchList(kind, name, split);
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

  async _fetchList(kind, name, split) {
    const data = await window.apiGet(`/api/events/list?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}&split=${encodeURIComponent(split)}`);
    if (!data || !data.ok) { this._failLoad((data && data.error) || "Could not read events."); return; }

    this.meta = data.meta;
    if (this.viewer) this.viewer.setDetectorBounds(this.meta.detector_min, this.meta.detector_max);
    this.gt = data.gt;
    this.error = data.error || [];
    this.nActive = data.n_active;
    this.hasPrediction = !!data.has_prediction;
    if (this.refs.legendPred) this.refs.legendPred.hidden = !this.hasPrediction;

    this.refs.stage.hidden = false;
    this._buildSliders();

    const center = [
      (this.meta.bounds_min[0] + this.meta.bounds_max[0]) / 2,
      (this.meta.bounds_min[1] + this.meta.bounds_max[1]) / 2,
      (this.meta.bounds_min[2] + this.meta.bounds_max[2]) / 2,
    ];
    this._setSliderValues(center);
    this._snap(center);
  }

  _buildSliders() {
    const axes = ["x", "y", "z"];
    this.refs.sliders.innerHTML = "";
    this.sliderInputs = {};

    axes.forEach((axis, index) => {
      const min = this.meta.bounds_min[index];
      const max = this.meta.bounds_max[index];
      const step = (max - min) / 400 || 0.001;

      const row = document.createElement("div");
      row.className = "ev-slider";
      row.innerHTML =
        `<label>${axis}</label>` +
        `<input type="range" min="${min}" max="${max}" step="${step}" value="${(min + max) / 2}" data-axis="${index}" />` +
        `<output data-axis="${index}">0</output>`;
      this.refs.sliders.appendChild(row);

      const input = row.querySelector("input");
      this.sliderInputs[index] = input;
      input.addEventListener("input", () => this._onSlider());
    });
  }

  _setSliderValues(point) {
    for (let index = 0; index < 3; index++) {
      this.sliderInputs[index].value = point[index];
      this.refs.sliders.querySelector(`output[data-axis="${index}"]`).textContent = this._fmt(point[index]);
    }
  }

  _onSlider() {
    const point = [0, 1, 2].map((index) => {
      const value = parseFloat(this.sliderInputs[index].value);
      this.refs.sliders.querySelector(`output[data-axis="${index}"]`).textContent = this._fmt(value);
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

    if (best !== this.selectedIndex) this._queueDetail(best);
  }

  _queueDetail(index) {
    this.selectedIndex = index;
    this.pendingIndex = index;
    this._pump();
  }

  async _pump() {
    if (this.fetching) return;
    this.fetching = true;
    while (this.pendingIndex !== null) {
      const index = this.pendingIndex;
      this.pendingIndex = null;
      await this._fetchDetail(index);
    }
    this.fetching = false;
  }

  async _fetchDetail(index) {
    const kind = this.currentKind;
    const name = this.currentName;
    const split = this.currentSplit;
    const data = await window.apiGet(`/api/events/detail?kind=${encodeURIComponent(kind)}&name=${encodeURIComponent(name)}&split=${encodeURIComponent(split)}&index=${index}`);
    if (!data || !data.ok) return;
    if (kind !== this.currentKind || name !== this.currentName || split !== this.currentSplit) return;

    if (this.viewer) this.viewer.setEvent(data, data.has_prediction);
    this._renderStats(data);
  }

  _renderStats(detail) {
    const vector = (point) => `<span class="ev-vec">${point.map((value) => this._fmt(value)).join(" &middot; ")}</span>`;
    const octant = detail.signs.map((sign) => (sign < 0 ? "−" : "+")).join("");

    let rows;
    if (detail.has_prediction) {
      rows = [
        ["true vertex (x y z)", vector(detail.gt)],
        ["prediction (x y z)", vector(detail.pred)],
        ["abs error (x y z)", vector(detail.error_xyz)],
        ["3D error", `<span class="ev-strong">${this._fmt(detail.error)} cm</span>`],
        ["error percentile", `${(detail.error_rank * 100).toFixed(1)} % of split below`],
        ["active sensors", String(detail.n_active)],
        ["total light", this._fmt(detail.total_light)],
        ["base event id", String(detail.base_event_id)],
        ["octant", octant],
        ["split error mean / median", `${this._fmt(this.meta.error_mean)} / ${this._fmt(this.meta.error_median)} cm`],
      ];
    } else {
      rows = [
        ["true vertex (x y z)", vector(detail.gt)],
        ["active sensors", String(detail.n_active)],
        ["total light", this._fmt(detail.total_light)],
        ["base event id", String(detail.base_event_id)],
        ["octant", octant],
      ];
    }

    this.refs.stats.innerHTML = rows
      .map(([key, value]) => `<tr><th>${window.escapeHtml(key)}</th><td>${value}</td></tr>`)
      .join("");

    const label = this.currentKind === "dataset" ? this.currentName : `${this.currentName} · ${this.currentSplit}`;
    this.refs.readout.textContent = `event ${detail.index + 1} / ${this.meta.count} · ${label}`;
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

window.colorForLight = colorForLight;
window.EventViewer = EventViewer;
window.EventExplorerPanel = EventExplorerPanel;
