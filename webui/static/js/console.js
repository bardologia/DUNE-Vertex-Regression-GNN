"use strict";

class ProcessConsole {
  constructor(element) {
    this.element = element;
    this.pid = null;
    this.pollTimer = null;
    this._renderEmpty();
  }

  _renderEmpty() {
    this.element.innerHTML = `<div class="console__hint">Select a process to follow its log.</div>`;
  }

  attach(pid) {
    this.pid = pid;
    this.element.innerHTML =
      `<div class="console__bar">` +
      `<span class="console__title">pid ${pid}</span>` +
      `<span class="badge" id="console-status">connecting</span>` +
      `<span class="console__spacer"></span>` +
      `</div>` +
      `<div class="console__out" id="console-out"></div>`;

    this.outElement = this.element.querySelector("#console-out");
    this.statusElement = this.element.querySelector("#console-status");

    clearInterval(this.pollTimer);
    this._poll();
    this.pollTimer = setInterval(() => this._poll(), 1500);
  }

  stop() {
    clearInterval(this.pollTimer);
    this.pollTimer = null;
  }

  async _poll() {
    if (this.pid === null) return;
    const data = await window.apiGet(`/api/processes/${this.pid}/log?lines=600`);

    if (!data.ok) {
      this.statusElement.textContent = "error";
      this.statusElement.className = "badge badge--failed";
      this.outElement.textContent = data.error || "log unavailable";
      this.stop();
      return;
    }

    const atBottom = this.outElement.scrollHeight - this.outElement.scrollTop - this.outElement.clientHeight < 40;
    this.outElement.textContent = (data.lines || []).join("\n");
    if (atBottom) this.outElement.scrollTop = this.outElement.scrollHeight;

    this.statusElement.textContent = data.status;
    this.statusElement.className = `badge badge--${data.status}`;

    if (data.status !== "running") {
      this.stop();
    }
  }
}

window.ProcessConsole = ProcessConsole;
