"use strict";

class ResultsPanel {
  constructor(refs) {
    this.listElement = refs.list;
    this.detailElement = refs.detail;
    this.barElement = refs.bar;
    this.runs = [];
    this.selectedName = null;
    this.entered = false;
    this._wireLightbox();
  }

  async enter() {
    if (!this.entered) {
      await this.load();
      this.entered = true;
    }
    await this._refreshTensorboard();
  }

  async load() {
    const data = await window.apiGet("/api/runs");
    this.runs = data.runs || [];
    this._renderList();
    if (this.runs.length) this._select(this.runs[0].name);
    else this.detailElement.innerHTML = `<div class="list-empty">No runs under runs/ yet.</div>`;
  }

  _renderList() {
    this.listElement.innerHTML = "";
    if (!this.runs.length) {
      this.listElement.innerHTML = `<div class="list-empty">No runs found.</div>`;
      return;
    }
    this.runs.forEach((run) => {
      const item = document.createElement("button");
      item.className = "run-item" + (run.name === this.selectedName ? " is-active" : "");
      const metric = run.best_metric ? `${run.best_metric.value}` : run.model;
      item.innerHTML =
        `<div class="run-item__name">${window.escapeHtml(run.name)}</div>` +
        `<div class="run-item__meta"><span>${window.escapeHtml(run.model)}</span><span class="run-item__metric">${window.escapeHtml(String(metric))}</span></div>` +
        `<div class="run-item__meta"><span>${window.escapeHtml(run.timestamp.replace("T", " "))}</span></div>`;
      item.addEventListener("click", () => this._select(run.name));
      this.listElement.appendChild(item);
    });
  }

  _select(name) {
    this.selectedName = name;
    const run = this.runs.find((item) => item.name === name);
    this._renderList();

    const metricPill = run.best_metric
      ? `<span class="rdetail__metric">${window.escapeHtml(run.best_metric.name)} = ${window.escapeHtml(String(run.best_metric.value))}${run.best_metric.unit ? " " + window.escapeHtml(run.best_metric.unit) : ""}</span>`
      : "";

    const figures = this._collectFigures(run);
    const gallery = figures.length
      ? `<section><div class="res-section__cap">figures <span>${figures.length}</span></div><div class="gallery">` +
        figures.map((fig) =>
          `<figure class="figcard" data-src="${window.escapeHtml(fig.url)}" data-cap="${window.escapeHtml(fig.rel)}">` +
          `<div class="figcard__media"><img loading="lazy" src="${window.escapeHtml(fig.url)}" alt="${window.escapeHtml(fig.name)}" /></div>` +
          `<figcaption class="figcard__cap">${window.escapeHtml(fig.name)}</figcaption></figure>`
        ).join("") + `</div></section>`
      : "";

    this.detailElement.className = "master__detail master__detail--results is-swap";
    this.detailElement.innerHTML =
      `<div class="rdetail__head"><div>` +
      `<p class="rdetail__model">${window.escapeHtml(run.model)}</p>` +
      `<h2 class="rdetail__name">${window.escapeHtml(run.name)} ${metricPill}</h2></div>` +
      `<button class="btn btn--mini" id="tb-launch">Launch TensorBoard</button></div>` +
      `<div class="rdetail__facts">` +
      `<div class="spec"><span class="spec__k">model</span><span class="spec__v">${window.escapeHtml(run.model)}</span></div>` +
      `<div class="spec"><span class="spec__k">timestamp</span><span class="spec__v">${window.escapeHtml(run.timestamp.replace("T", " "))}</span></div>` +
      (run.best_metric ? `<div class="spec"><span class="spec__k">${window.escapeHtml(run.best_metric.name)}</span><span class="spec__v is-accent">${window.escapeHtml(String(run.best_metric.value))}</span></div>` : "") +
      `<div class="spec"><span class="spec__k">path</span><span class="spec__v">${window.escapeHtml(run.path)}</span></div>` +
      `</div>` +
      gallery +
      `<section><div class="res-section__cap">files</div><div class="tree">${this._treeHtml(run, run.tree, "")}</div></section>`;

    this.detailElement.querySelector("#tb-launch").addEventListener("click", () => this._launchTensorboard(run.name));
    this.detailElement.querySelectorAll(".figcard").forEach((card) => {
      card.addEventListener("click", () => this._openLightbox(card.dataset.src, card.dataset.cap));
    });
    this.detailElement.querySelectorAll(".tree__file--img").forEach((el) => {
      el.addEventListener("click", () => this._openLightbox(el.dataset.src, el.dataset.cap));
    });
  }

  _isImage(name) {
    return /\.(png|jpe?g|gif|svg|webp)$/i.test(name);
  }

  _fileUrl(run, rel) {
    return `/api/runs/file?path=${encodeURIComponent(run.name + "/" + rel)}`;
  }

  _collectFigures(run) {
    const figures = [];
    const walk = (node, prefix) => {
      (node.files || []).forEach((file) => {
        if (this._isImage(file.name)) {
          const rel = prefix + file.name;
          figures.push({ name: file.name, rel, url: this._fileUrl(run, rel) });
        }
      });
      (node.children || []).forEach((child) => walk(child, `${prefix}${child.name}/`));
    };
    walk(run.tree, "");
    return figures.slice(0, 80);
  }

  _treeHtml(run, node, prefix) {
    const files = (node.files || [])
      .map((file) => {
        const rel = prefix + file.name;
        if (this._isImage(file.name)) {
          return `<div class="tree__file tree__file--img" data-src="${window.escapeHtml(this._fileUrl(run, rel))}" data-cap="${window.escapeHtml(rel)}"><span>${window.escapeHtml(file.name)}</span><span class="tree__size">${window.formatBytes(file.size)}</span></div>`;
        }
        return `<div class="tree__file"><span>${window.escapeHtml(file.name)}</span><span class="tree__size">${window.formatBytes(file.size)}</span></div>`;
      })
      .join("");
    const children = (node.children || [])
      .map((child) => `<div class="tree__dir">${window.escapeHtml(child.name)}/</div><div class="tree__children">${this._treeHtml(run, child, `${prefix}${child.name}/`)}</div>`)
      .join("");
    return files + children;
  }

  _wireLightbox() {
    this.lightbox = document.getElementById("lightbox");
    this.lightboxImg = document.getElementById("lightbox-img");
    this.lightboxCap = document.getElementById("lightbox-cap");
    const close = document.getElementById("lightbox-close");
    if (close) close.addEventListener("click", () => this._closeLightbox());
    if (this.lightbox) {
      this.lightbox.addEventListener("click", (event) => { if (event.target === this.lightbox) this._closeLightbox(); });
    }
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") this._closeLightbox(); });
  }

  _openLightbox(src, cap) {
    if (!this.lightbox) return;
    this.lightboxImg.src = src;
    this.lightboxCap.textContent = cap || "";
    this.lightbox.hidden = false;
  }

  _closeLightbox() {
    if (this.lightbox) this.lightbox.hidden = true;
  }

  async _launchTensorboard(name) {
    const result = await window.apiPost("/api/tensorboard", { run: name });
    if (!result.ok) { window.toast(result.error || "TensorBoard launch failed", "error"); return; }
    window.toast(`TensorBoard on port ${result.port}`, "ok");
    await this._refreshTensorboard();
  }

  async _refreshTensorboard() {
    const data = await window.apiGet("/api/tensorboard");
    const instances = data.instances || [];
    if (!instances.length) { this.barElement.innerHTML = ""; return; }

    this.barElement.innerHTML = instances
      .map((instance) => {
        const stop = instance.status === "running" ? `<button class="btn btn--mini btn--danger" data-stop="${instance.pid}">Stop</button>` : "";
        return (
          `<div class="tb-instance"><span class="badge badge--${instance.status === "running" ? "running" : "failed"}">${instance.status}</span>` +
          `<a href="${window.escapeHtml(instance.url)}" target="_blank">${window.escapeHtml(instance.url)}</a>` +
          `<span class="tb-instance__path">${window.escapeHtml(instance.logdir)}</span>` +
          `<span class="tb-spacer"></span>${stop}</div>`
        );
      })
      .join("");

    this.barElement.querySelectorAll("[data-stop]").forEach((button) => {
      button.addEventListener("click", async () => {
        await window.apiPost(`/api/tensorboard/${button.dataset.stop}/stop`, {});
        this._refreshTensorboard();
      });
    });
  }
}

window.ResultsPanel = ResultsPanel;
