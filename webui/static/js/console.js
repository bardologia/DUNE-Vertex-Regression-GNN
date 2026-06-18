"use strict";

class ConsolePanel {
  constructor(refs) {
    this.listElement = refs.list;
    this.countElement = refs.count;
    this.tilesElement = refs.tiles;
    this.processes = [];
    this.selectedPid = null;
    this.focusPid = null;
    this.listTimer = null;
    this.logTimer = null;
    this.outElement = null;
  }

  enter() {
    this.refresh();
    clearInterval(this.listTimer);
    this.listTimer = setInterval(() => this.refresh(), 3000);
  }

  leave() {
    clearInterval(this.listTimer);
    clearInterval(this.logTimer);
    this.listTimer = null;
    this.logTimer = null;
  }

  async refresh() {
    const data = await window.apiGet("/api/processes");
    if (data.error) return;
    this.processes = data.processes || [];

    if (this.focusPid !== null) {
      this.selectedPid = this.focusPid;
      this.focusPid = null;
      this._attach(this.selectedPid);
    } else if (this.selectedPid === null && this.processes.length) {
      this._select(this.processes[0].pid);
      return;
    }
    this._renderList();
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
      item.className = "job-item" + (process.pid === this.selectedPid ? " is-active" : "");
      const kill = process.status === "running"
        ? `<button class="btn btn--mini btn--danger job-item__kill" data-kill="${process.pid}">Stop</button>`
        : "";
      item.innerHTML =
        `<div class="job-item__top"><span class="job-item__name">${window.escapeHtml(process.script)}</span>` +
        `<span class="badge badge--${process.status}">${process.status}</span></div>` +
        `<div class="job-item__meta">pid ${process.pid} &middot; ${window.escapeHtml(String(process.started || "").replace("T", " "))}</div>` +
        kill;
      item.addEventListener("click", () => this._select(process.pid));
      const killBtn = item.querySelector("[data-kill]");
      if (killBtn) killBtn.addEventListener("click", (event) => this._kill(process.pid, event));
      this.listElement.appendChild(item);
    });
  }

  _select(pid) {
    this.selectedPid = pid;
    this._attach(pid);
    this._renderList();
  }

  async _kill(pid, event) {
    event.stopPropagation();
    const result = await window.apiPost(`/api/processes/${pid}/kill`, {});
    if (result.ok) window.toast(`Stop signal sent to pid ${pid}`, "ok");
    else window.toast(result.error || "Could not stop", "error");
    this.refresh();
  }

  _attach(pid) {
    const process = this.processes.find((p) => p.pid === pid) || { script: `pid ${pid}`, pid };
    this.tilesElement.innerHTML =
      `<div class="console-tile">` +
      `<div class="console-tile__bar">` +
      `<span class="console-tile__dots" aria-hidden="true"><i></i><i></i><i></i></span>` +
      `<span class="console-tile__name">${window.escapeHtml(process.script)}</span>` +
      `<span class="console-tile__meta" id="console-meta">pid ${pid}</span>` +
      `<span class="badge badge--connecting" id="console-status">connecting</span>` +
      `</div>` +
      `<div class="console-tile__out" id="console-out"></div></div>`;

    this.outElement = this.tilesElement.querySelector("#console-out");
    this.statusElement = this.tilesElement.querySelector("#console-status");

    clearInterval(this.logTimer);
    this._poll();
    this.logTimer = setInterval(() => this._poll(), 1500);
  }

  async _poll() {
    if (this.selectedPid === null) return;
    const data = await window.apiGet(`/api/processes/${this.selectedPid}/log?lines=800`);
    if (!this.outElement) return;

    if (!data.ok) {
      this.statusElement.textContent = "error";
      this.statusElement.className = "badge badge--failed";
      this.outElement.textContent = data.error || "log unavailable";
      clearInterval(this.logTimer);
      return;
    }

    const atBottom = this.outElement.scrollHeight - this.outElement.scrollTop - this.outElement.clientHeight < 60;
    this.outElement.innerHTML = (data.lines || []).map((line) => this._line(line)).join("");
    if (atBottom) this.outElement.scrollTop = this.outElement.scrollHeight;

    this.statusElement.textContent = data.status;
    this.statusElement.className = `badge badge--${data.status}`;

    if (data.status !== "running") clearInterval(this.logTimer);
  }

  _line(text) {
    const safe = window.escapeHtml(text);
    const lower = text.toLowerCase();
    let cls = "";
    if (/\b(error|traceback|failed|exception)\b/.test(lower)) cls = "ln-err";
    else if (/\b(warn|warning)\b/.test(lower)) cls = "ln-warn";
    else if (/\b(done|complete|saved|finished|success)\b/.test(lower)) cls = "ln-ok";
    return cls ? `<span class="${cls}">${safe}</span>\n` : `${safe}\n`;
  }
}

window.ConsolePanel = ConsolePanel;
