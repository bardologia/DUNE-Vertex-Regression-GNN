"use strict";

class Observatory {
  constructor(els) {
    this.els = els;
    this.built = false;
    this.gpuEls = [];
    this.coreEls = [];
    this.histMax = 120;
    this.hist = { cpu: [], ram: [], gpus: [] };
    this.sysTimer = null;
    this.jobTimer = null;
  }

  enter() {
    this._poll();
    this._pollJobs();
    clearInterval(this.sysTimer);
    clearInterval(this.jobTimer);
    this.sysTimer = setInterval(() => { if (!document.hidden) this._poll(); }, 2500);
    this.jobTimer = setInterval(() => { if (!document.hidden) this._pollJobs(); }, 4000);
  }

  leave() {
    clearInterval(this.sysTimer);
    clearInterval(this.jobTimer);
    this.sysTimer = null;
    this.jobTimer = null;
  }

  async _poll() {
    let sys;
    try { sys = await window.apiGet("/api/system"); } catch (e) { return; }
    if (!sys || sys.error) return;
    if (!this.built) this._build(sys);
    this._update(sys);
  }

  async _pollJobs() {
    let data;
    try { data = await window.apiGet("/api/processes"); } catch (e) { return; }
    this._updateJobs(data.processes || []);
  }

  _build(sys) {
    this.built = true;
    const gpus  = sys.gpus || [];
    const cores = (sys.cpu && sys.cpu.cores) || [];
    const wd    = sys.watchdog || {};
    this.hist.gpus = gpus.map(() => ({ u: [], m: [] }));

    const gpuCards = gpus.length
      ? gpus.map((g, i) =>
          `<article class="gcard" data-gpu="${i}">` +
          `<header class="gcard__head"><span class="gcard__idx">gpu ${g.index != null ? g.index : i}</span><span class="gcard__name">${this._esc(g.name || "unknown")}</span><span class="gcard__temp">--</span></header>` +
          `<div class="gcard__row"><span class="gcard__pct">--</span><span class="gcard__unit">% util</span><span class="gcard__vram">--</span></div>` +
          `<canvas class="gcard__graph"></canvas>` +
          `<footer class="gcard__foot"><span class="gcard__mem">--</span><span class="gcard__legend"><i class="lg lg--util"></i>util<i class="lg lg--vram"></i>vram</span></footer>` +
          `</article>`
        ).join("")
      : `<div class="sboard__empty">no CUDA devices visible to the backend</div>`;

    const coreCells = cores.map((_, i) => `<i class="cpu__cell" data-core="${i}" title="core ${i}"></i>`).join("");

    const limitCells = [
      [wd.gpu_count != null ? String(wd.gpu_count) : "0", "devices"],
      [wd.interval != null ? `${Math.round(wd.interval)} s` : "--", "poll interval"],
    ].map(([v, k]) => `<div><dt>${v}</dt><dd>${k}</dd></div>`).join("");

    this.els.board.innerHTML =
      `<section class="sboard sboard--wd" aria-label="GPU watchdog">` +
      `<div class="wd__state"><i class="wd__light" id="sb-wd-light" aria-hidden="true"></i><span class="wd__label">gpu watchdog</span><span class="wd__mode" id="sb-wd-mode">--</span></div>` +
      `<span class="wd__status" id="sb-wd-status">monitoring</span>` +
      `<dl class="wd__limits">${limitCells}</dl>` +
      `</section>` +

      `<section class="sboard sboard--gpus" aria-label="CUDA devices">` +
      `<header class="sboard__cap"><span>cuda devices</span><span class="sboard__n">${gpus.length}</span></header>` +
      `<div class="sboard__gpugrid">${gpuCards}</div>` +
      `</section>` +

      `<section class="sboard sboard--cpu" aria-label="Processor">` +
      `<header class="sboard__cap"><span>processor</span><span class="sboard__n">${sys.cpu ? sys.cpu.count : 0} cores</span></header>` +
      `<div class="cpu__top">` +
      `<div class="cpu__big"><span class="cpu__pct" id="sb-cpu-pct">--</span><span class="cpu__unit">% busy</span></div>` +
      `<dl class="cpu__load"><div><dt id="sb-load1">--</dt><dd>load 1m</dd></div><div><dt id="sb-load5">--</dt><dd>5m</dd></div><div><dt id="sb-load15">--</dt><dd>15m</dd></div></dl>` +
      `</div>` +
      `<canvas class="sboard__graph" id="sb-cpu-graph"></canvas>` +
      `<div class="sboard__metric"><span>avg usage</span><span id="sb-cpu-avg">--</span></div>` +
      `<div class="bar"><i class="bar__fill" id="sb-cpu-bar"></i></div>` +
      `<div class="sboard__metric"><span>active cores</span><span id="sb-cpu-active">--</span></div>` +
      `<div class="bar"><i class="bar__fill bar__fill--cores" id="sb-cores-bar"></i></div>` +
      `<div class="cpu__grid" id="sb-cores">${coreCells}</div>` +
      `</section>` +

      `<section class="sboard sboard--mem" aria-label="Memory">` +
      `<header class="sboard__cap"><span>memory</span><span class="sboard__n" id="sb-mem-total"></span></header>` +
      `<div class="sboard__metric"><span>ram</span><span id="sb-ram-txt">--</span></div>` +
      `<div class="bar"><i class="bar__fill" id="sb-ram-bar"></i></div>` +
      `<div class="sboard__metric"><span>swap</span><span id="sb-swap-txt">--</span></div>` +
      `<div class="bar"><i class="bar__fill bar__fill--swap" id="sb-swap-bar"></i></div>` +
      `<canvas class="sboard__graph sboard__graph--mem" id="sb-mem-graph"></canvas>` +
      `</section>` +

      `<section class="sboard sboard--disk" aria-label="Storage">` +
      `<header class="sboard__cap"><span>storage</span><span class="sboard__n" id="sb-disk-total"></span></header>` +
      `<div class="sboard__metric"><span class="sboard__path" id="sb-disk-path"></span><span id="sb-disk-txt">--</span></div>` +
      `<div class="bar"><i class="bar__fill" id="sb-disk-bar"></i></div>` +
      `<div class="sboard__metric"><span>free</span><span id="sb-disk-free">--</span></div>` +
      `</section>` +

      `<section class="sboard sboard--procs" aria-label="Processes">` +
      `<header class="sboard__cap"><span>processes &middot; ${this._esc(sys.user || "user")}</span><span class="sboard__n" id="sb-proc-n"></span></header>` +
      `<div class="ptable">` +
      `<div class="ptable__row ptable__row--head"><span>pid</span><span>cpu%</span><span>mem</span><span>s</span><span>command</span></div>` +
      `<div class="ptable__body" id="sb-procs"></div>` +
      `</div>` +
      `</section>` +

      `<section class="sboard sboard--jobs" aria-label="Jobs">` +
      `<header class="sboard__cap"><span>launches</span><span class="sboard__n" id="sb-jobs-n">0</span></header>` +
      `<ul class="sboard__jobs" id="sb-jobs"><li class="sboard__empty">no runs yet</li></ul>` +
      `</section>`;

    this.gpuEls = [...this.els.board.querySelectorAll(".gcard")].map((card) => ({
      pct: card.querySelector(".gcard__pct"),
      vramTxt: card.querySelector(".gcard__vram"),
      temp: card.querySelector(".gcard__temp"),
      mem: card.querySelector(".gcard__mem"),
      graph: card.querySelector(".gcard__graph"),
    }));
    this.coreEls = [...this.els.board.querySelectorAll(".cpu__cell")];
  }

  _update(sys) {
    if (window.serverScene && window.serverScene.feed) window.serverScene.feed(sys);

    const cpu  = sys.cpu || {};
    const mem  = sys.mem || {};
    const disk = sys.disk || {};
    const gpus = sys.gpus || [];
    const wd   = sys.watchdog || {};

    if (this.els.host) this.els.host.textContent = sys.host || "server";
    if (this.els.sum) {
      const bits = [];
      if (sys.uptime) bits.push(`up ${this._uptime(sys.uptime)}`);
      if (cpu.count) bits.push(`${cpu.count} cores`);
      bits.push(`${gpus.length} CUDA device${gpus.length === 1 ? "" : "s"}`);
      if (mem.total) bits.push(`${this._tb(mem.total)} ram`);
      this.els.sum.textContent = bits.join(" · ");
    }

    this._renderWatchdog(wd);

    this._push(this.hist.cpu, cpu.total || 0);
    if (mem.total) this._push(this.hist.ram, (100 * (mem.total - mem.available)) / mem.total);

    gpus.forEach((g, i) => {
      const el = this.gpuEls[i];
      const h  = this.hist.gpus[i];
      if (!el || !h) return;
      const util   = g.util != null ? g.util : 0;
      const memPct = g.mem_total ? (100 * g.mem_used) / g.mem_total : 0;
      this._push(h.u, util);
      this._push(h.m, memPct);

      el.pct.textContent = Math.round(util);
      el.vramTxt.innerHTML = `<b>${this._gb(g.mem_used)}</b> / ${this._gb(g.mem_total)} GB`;
      el.temp.textContent = g.temp != null ? `${Math.round(g.temp)}°C` : "--";
      el.temp.className = "gcard__temp" + (g.temp >= 85 ? " is-danger" : g.temp >= 70 ? " is-warn" : "");
      el.mem.textContent = `${Math.round(memPct)}% vram`;
      this._spark(el.graph, [
        { data: h.m, color: this.teal, fill: 0.1 },
        { data: h.u, color: this.blue, fill: 0.16 },
      ]);
    });

    const pctEl = document.getElementById("sb-cpu-pct");
    if (pctEl) pctEl.textContent = Math.round(cpu.total || 0);
    const load = cpu.load || [];
    ["sb-load1", "sb-load5", "sb-load15"].forEach((id, i) => {
      const el = document.getElementById(id);
      if (el && load[i] != null) el.textContent = load[i].toFixed(1);
    });
    const cores = cpu.cores || [];
    cores.forEach((u, i) => {
      const cell = this.coreEls[i];
      if (!cell) return;
      const a = 0.05 + Math.min(1, u / 100) * 0.85;
      cell.style.background = `rgba(${this.blue}, ${a.toFixed(3)})`;
      cell.title = `core ${i} · ${Math.round(u)}%`;
    });
    if (cores.length) {
      const avg    = cores.reduce((s, u) => s + u, 0) / cores.length;
      const active = cores.filter((u) => u >= 50).length;
      this._bar("sb-cpu-bar", avg);
      this._bar("sb-cores-bar", (100 * active) / cores.length);
      this._txt("sb-cpu-avg", `<b>${avg.toFixed(1)}</b> %`);
      this._txt("sb-cpu-active", `<b>${active}</b> / ${cores.length} dispatched`);
    }
    this._spark(document.getElementById("sb-cpu-graph"), [{ data: this.hist.cpu, color: this.blue, fill: 0.16 }]);

    if (mem.total) {
      const used = mem.total - mem.available;
      this._bar("sb-ram-bar", (100 * used) / mem.total);
      this._txt("sb-ram-txt", `<b>${this._gb(used)}</b> / ${this._gb(mem.total)} GB`);
      this._txt("sb-mem-total", this._tb(mem.total));
      const swapUsed = (mem.swap_total || 0) - (mem.swap_free || 0);
      this._bar("sb-swap-bar", mem.swap_total ? (100 * swapUsed) / mem.swap_total : 0);
      this._txt("sb-swap-txt", mem.swap_total ? `<b>${this._gb(swapUsed)}</b> / ${this._gb(mem.swap_total)} GB` : "none");
      this._spark(document.getElementById("sb-mem-graph"), [{ data: this.hist.ram, color: this.teal, fill: 0.12 }]);
    }

    if (disk.total) {
      this._bar("sb-disk-bar", (100 * disk.used) / disk.total);
      this._txt("sb-disk-txt", `<b>${this._tb(disk.used)}</b> / ${this._tb(disk.total)}`);
      this._txt("sb-disk-total", this._tb(disk.total));
      this._txt("sb-disk-free", `<b>${this._tb(disk.free)}</b>`);
      const path = document.getElementById("sb-disk-path");
      if (path) { path.textContent = disk.path || ""; path.title = disk.path || ""; }
    }

    this._renderProcs(sys.procs || []);
  }

  _renderWatchdog(wd) {
    const light  = document.getElementById("sb-wd-light");
    const mode   = document.getElementById("sb-wd-mode");
    const status = document.getElementById("sb-wd-status");
    if (!light || !mode || !status) return;
    const armed = !!wd.armed;
    light.classList.toggle("is-armed", armed);
    mode.textContent = armed ? "armed" : "idle";
    mode.classList.toggle("is-off", !armed);
    status.textContent = armed ? "monitoring devices" : "no nvml devices";
  }

  _renderProcs(procs) {
    const body = document.getElementById("sb-procs");
    const n    = document.getElementById("sb-proc-n");
    if (!body) return;
    if (n) n.textContent = String(procs.length);

    if (!procs.length) {
      body.innerHTML = `<div class="sboard__empty">no processes</div>`;
      return;
    }
    body.innerHTML = procs.map((p) => {
      const cls = p.cpu >= 100 ? "is-hot" : p.cpu >= 25 ? "is-mid" : "";
      const run = p.state === "R" ? " is-run" : "";
      const cmd = (p.cmd || "").split("/").slice(-2).join("/") || p.cmd;
      return (
        `<div class="ptable__row${run}">` +
        `<span class="ptable__pid">${p.pid}</span>` +
        `<span class="ptable__cpu ${cls}">${p.cpu.toFixed(1)}</span>` +
        `<span>${this._mb(p.rss)}</span>` +
        `<span class="ptable__state">${this._esc(p.state)}</span>` +
        `<span class="ptable__cmd" title="${this._esc(p.cmd)}">${this._esc(cmd)}</span>` +
        `</div>`
      );
    }).join("");
  }

  _updateJobs(jobs) {
    const list = document.getElementById("sb-jobs");
    const n    = document.getElementById("sb-jobs-n");
    if (!list) return;
    const running = jobs.filter((j) => j.status === "running").length;
    if (n) n.textContent = running > 0 ? `${running} running` : String(jobs.length);

    if (!jobs.length) {
      list.innerHTML = `<li class="sboard__empty">no runs yet</li>`;
      return;
    }
    list.innerHTML = jobs.slice(0, 9).map((j) => {
      const cls =
        j.status === "running" ? "is-run" :
        j.status === "failed" ? "is-fail" : "is-done";
      return (
        `<li class="sboard__job ${cls}">` +
        `<span class="sboard__jdot" aria-hidden="true"></span>` +
        `<span class="sboard__jname">${this._esc(j.script)}</span>` +
        `<span class="sboard__jstate">${this._esc(j.status)}</span></li>`
      );
    }).join("");
  }

  _push(arr, v) {
    arr.push(Math.max(0, Math.min(100, v)));
    if (arr.length > this.histMax) arr.shift();
  }

  _spark(cv, series) {
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(140, 160, 180, 0.08)";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((f) => {
      const y = Math.round(h * f) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    });

    const step = w / (this.histMax - 1);
    series.forEach((s) => {
      const d = s.data;
      if (d.length < 2) return;
      const x0 = w - (d.length - 1) * step;
      const py = (v) => h - 1.5 - (v / 100) * (h - 3);

      ctx.beginPath();
      d.forEach((v, i) => {
        const x = x0 + i * step;
        if (i === 0) ctx.moveTo(x, py(v));
        else ctx.lineTo(x, py(v));
      });
      ctx.strokeStyle = `rgba(${s.color}, 0.95)`;
      ctx.lineWidth = 1.4;
      ctx.stroke();

      ctx.lineTo(w, h);
      ctx.lineTo(x0, h);
      ctx.closePath();
      ctx.fillStyle = `rgba(${s.color}, ${s.fill})`;
      ctx.fill();
    });
  }

  _bar(id, pct) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    el.classList.toggle("is-hot", pct >= 90);
  }

  _txt(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  }

  _gb(bytes) { return (bytes / 1073741824).toFixed(1); }

  _mb(bytes) {
    const gb = bytes / 1073741824;
    if (gb >= 1) return `${gb.toFixed(1)}G`;
    return `${Math.round(bytes / 1048576)}M`;
  }

  _tb(bytes) {
    const tb = bytes / 1099511627776;
    return tb >= 1 ? `${tb.toFixed(2)} TB` : `${this._gb(bytes)} GB`;
  }

  _uptime(sec) {
    const d = Math.floor(sec / 86400);
    const h = Math.floor((sec % 86400) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    if (d > 0) return `${d}d ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
    return `${h}:${String(m).padStart(2, "0")}`;
  }

  _esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
}

Observatory.prototype.blue = "111, 155, 255";
Observatory.prototype.teal = "45, 212, 191";

window.Observatory = Observatory;
