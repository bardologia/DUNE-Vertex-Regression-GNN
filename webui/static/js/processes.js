"use strict";

class ProcessPanel {
  constructor(refs) {
    this.listElement = refs.list;
    this.console = new window.ProcessConsole(refs.console);
    this.processes = [];
    this.selectedPid = null;
    this.focusPid = null;
    this.refreshTimer = null;
  }

  enter() {
    this.refresh();
    clearInterval(this.refreshTimer);
    this.refreshTimer = setInterval(() => this.refresh(), 3000);
  }

  leave() {
    clearInterval(this.refreshTimer);
    this.refreshTimer = null;
    this.console.stop();
  }

  async refresh() {
    const data = await window.apiGet("/api/processes");
    this.processes = data.processes || [];

    if (this.focusPid !== null) {
      this.selectedPid = this.focusPid;
      this.focusPid = null;
      this.console.attach(this.selectedPid);
    }

    this._renderList();
  }

  _select(pid) {
    this.selectedPid = pid;
    this.console.attach(pid);
    this._renderList();
  }

  async _kill(pid, event) {
    event.stopPropagation();
    const result = await window.apiPost(`/api/processes/${pid}/kill`, {});
    if (result.ok) window.toast(`Stop signal sent to pid ${pid}`, "ok");
    else window.toast(result.error || "Could not stop", "error");
    this.refresh();
  }

  _renderList() {
    this.listElement.innerHTML = "";

    if (!this.processes.length) {
      const empty = document.createElement("li");
      empty.className = "list-empty";
      empty.textContent = "No processes launched yet.";
      this.listElement.appendChild(empty);
      return;
    }

    this.processes.forEach((process) => {
      const item = document.createElement("li");
      item.className = "process-item" + (process.pid === this.selectedPid ? " is-active" : "");

      const killButton = process.status === "running" ? `<button class="btn btn--mini btn--danger" data-kill="${process.pid}">Stop</button>` : "";

      item.innerHTML =
        `<div class="process-item__top"><span class="process-item__name">${window.escapeHtml(process.script)}</span>` +
        `<span class="badge badge--${process.status}">${process.status}</span></div>` +
        `<div class="process-item__meta">pid ${process.pid} &middot; ${window.escapeHtml(process.started.replace("T", " "))}</div>` +
        `<div class="process-item__meta">${killButton}</div>`;

      item.addEventListener("click", () => this._select(process.pid));
      const kill = item.querySelector("[data-kill]");
      if (kill) kill.addEventListener("click", (event) => this._kill(process.pid, event));

      this.listElement.appendChild(item);
    });
  }
}

window.ProcessPanel = ProcessPanel;
