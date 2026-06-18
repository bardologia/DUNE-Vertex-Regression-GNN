"use strict";

class ConsoleTile {
  constructor(process, manager, host) {
    this.process = process;
    this.manager = manager;
    this.pid = process.pid;
    this.source = null;
    this.opened = false;

    this.root = document.createElement("div");
    this.root.className = "console-tile";

    const bar = document.createElement("div");
    bar.className = "console-tile__bar";

    const dots = document.createElement("span");
    dots.className = "console-tile__dots";
    dots.setAttribute("aria-hidden", "true");
    dots.innerHTML = "<i></i><i></i><i></i>";

    this.nameEl = document.createElement("span");
    this.nameEl.className = "console-tile__name";
    this.nameEl.textContent = process.script;
    this.nameEl.title = process.command || process.script;

    this.metaEl = document.createElement("span");
    this.metaEl.className = "console-tile__meta";
    this.metaEl.textContent = `pid ${process.pid}`;

    this.badgeEl = document.createElement("span");
    this.badgeEl.className = `badge badge--${process.status}`;
    this.badgeEl.textContent = process.status;

    this.stopBtn = document.createElement("button");
    this.stopBtn.className = "btn btn--mini btn--danger";
    this.stopBtn.textContent = "Stop";
    this.stopBtn.addEventListener("click", () => this.manager.stop(this.pid));

    this.closeBtn = document.createElement("button");
    this.closeBtn.className = "btn btn--mini";
    this.closeBtn.textContent = "Close";
    this.closeBtn.addEventListener("click", () => this.manager.close(this.pid));

    bar.append(dots, this.nameEl, this.metaEl, this.badgeEl, this.stopBtn, this.closeBtn);

    this.outEl = document.createElement("div");
    this.outEl.className = "console-tile__out";

    this.root.append(bar, this.outEl);
    host.appendChild(this.root);

    this.term = new Terminal({
      cols: 120,
      rows: 24,
      convertEol: true,
      disableStdin: true,
      cursorBlink: false,
      cursorInactiveStyle: "none",
      scrollback: 10000,
      fontFamily: '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace',
      fontSize: 12,
      lineHeight: 1.3,
      theme: {
        background: "#0b0e14",
        foreground: "#d6dce4",
        cursor: "#2f6fed",
        selectionBackground: "#27405f",
      },
    });
    this.fitAddon = new FitAddon.FitAddon();
    this.term.loadAddon(this.fitAddon);

    this.setStatus(process.status);
    this._connect();
  }

  _connect() {
    this.source = new EventSource(`/api/processes/${this.pid}/stream`);
    this.source.onmessage = (event) => this._onEvent(event);
    this.source.onerror = () => this._disconnect();
  }

  _disconnect() {
    if (this.source) {
      this.source.close();
      this.source = null;
    }
  }

  _onEvent(event) {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (error) {
      return;
    }

    if (data.type === "chunk") {
      this.term.write(data.data);
    } else if (data.type === "status") {
      if (data.status === "running") {
        this.metaEl.textContent = `pid ${data.pid}`;
        this.setStatus("running");
      } else {
        this._note(`process ${data.status} (exit ${data.code})`, data.code === 0 ? "36" : "31");
        this.setStatus(data.status);
        this.manager.refresh();
      }
    } else if (data.type === "end") {
      this._disconnect();
    }
  }

  _note(text, color) {
    this.term.write(`\r\n\x1b[2;${color}m-- ${text} --\x1b[0m\r\n`);
  }

  setStatus(status) {
    this.process.status = status;
    this.badgeEl.className = `badge badge--${status}`;
    this.badgeEl.textContent = status;
    this.stopBtn.disabled = status !== "running";
  }

  fit() {
    const width = this.outEl.clientWidth;
    const height = this.outEl.clientHeight;
    if (!width || !height) return;

    if (!this.opened) {
      this.term.open(this.outEl);
      this.opened = true;
    }

    let font = Math.max(8, Math.min(16, (width - 16) / 72));
    this.term.options.fontSize = font;

    let dims = this.fitAddon.proposeDimensions();
    if (dims && dims.cols && dims.cols < 120) {
      font = Math.max(7, (font * dims.cols) / 120);
      this.term.options.fontSize = font;
      dims = this.fitAddon.proposeDimensions();
    }
    if (!dims || !dims.cols || !dims.rows) return;

    this.term.resize(Math.min(120, dims.cols), Math.max(2, dims.rows));
    this.term.scrollToBottom();
  }

  dispose() {
    this._disconnect();
    this.term.dispose();
    this.root.remove();
  }
}

class ConsolePanel {
  constructor(refs) {
    this.listElement = refs.list;
    this.countElement = refs.count;
    this.tilesElement = refs.tiles;
    this.hintElement = document.getElementById("console-hint");
    this.processes = [];
    this.tiles = new Map();
    this.dismissed = new Set();
    this.focusPid = null;
    this.listTimer = null;
    this._fitTimer = null;

    window.addEventListener("resize", () => this._queueFit());
  }

  enter() {
    this.refresh();
    clearInterval(this.listTimer);
    this.listTimer = setInterval(() => this.refresh(), 3000);
    this._queueFit();
  }

  leave() {
    clearInterval(this.listTimer);
    this.listTimer = null;
  }

  async refresh() {
    const data = await window.apiGet("/api/processes");
    if (data.error) return;
    this.processes = data.processes || [];

    if (this.focusPid !== null) {
      this.open(this.focusPid);
      this.focusPid = null;
    }

    this.processes
      .filter((process) => process.status === "running" && !this.dismissed.has(process.pid) && !this.tiles.has(process.pid))
      .forEach((process) => this.open(process.pid));

    this._renderList();
  }

  open(pid) {
    if (this.tiles.has(pid)) return;
    const process = this.processes.find((item) => item.pid === pid);
    if (!process) return;
    this.dismissed.delete(pid);
    this.tiles.set(pid, new ConsoleTile(process, this, this.tilesElement));
    this._layout();
    this._renderList();
  }

  close(pid) {
    const tile = this.tiles.get(pid);
    if (!tile) return;
    tile.dispose();
    this.tiles.delete(pid);
    this.dismissed.add(pid);
    this._layout();
    this._renderList();
  }

  toggle(pid) {
    if (this.tiles.has(pid)) this.close(pid);
    else this.open(pid);
  }

  async stop(pid) {
    const result = await window.apiPost(`/api/processes/${pid}/kill`, {});
    if (result.ok) window.toast(`Stop signal sent to pid ${pid}`, "ok");
    else window.toast(result.error || "Could not stop", "error");
  }

  _renderList() {
    this.countElement.textContent = this.processes.length ? `· ${this.processes.length}` : "";
    this.listElement.innerHTML = "";

    if (!this.processes.length) {
      this.listElement.innerHTML = `<li class="job-list__empty">No launches yet.<br />Start one from the control panel.</li>`;
      return;
    }

    this.processes.forEach((process) => {
      const item = document.createElement("li");
      item.className = "job-item" + (this.tiles.has(process.pid) ? " is-active" : "");
      const kill = process.status === "running"
        ? `<button class="btn btn--mini btn--danger job-item__kill" data-kill="${process.pid}">Stop</button>`
        : "";
      item.innerHTML =
        `<div class="job-item__top"><span class="job-item__name">${window.escapeHtml(process.script)}</span>` +
        `<span class="badge badge--${process.status}">${process.status}</span></div>` +
        `<div class="job-item__meta">pid ${process.pid} &middot; ${window.escapeHtml(String(process.started || "").replace("T", " "))}</div>` +
        kill;
      item.addEventListener("click", () => this.toggle(process.pid));
      const killBtn = item.querySelector("[data-kill]");
      if (killBtn) killBtn.addEventListener("click", (event) => { event.stopPropagation(); this.stop(process.pid); });
      this.listElement.appendChild(item);
    });
  }

  _layout() {
    const count = this.tiles.size;
    if (this.hintElement) this.hintElement.style.display = count ? "none" : "flex";
    const columns = count <= 1 ? 1 : count <= 4 ? 2 : 3;
    const rows = Math.max(1, Math.ceil(count / columns));
    this.tilesElement.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
    this.tilesElement.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    this._queueFit();
  }

  _queueFit() {
    clearTimeout(this._fitTimer);
    this._fitTimer = setTimeout(() => this.tiles.forEach((tile) => tile.fit()), 120);
  }
}

window.ConsolePanel = ConsolePanel;
