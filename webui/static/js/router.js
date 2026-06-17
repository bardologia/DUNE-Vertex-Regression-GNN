"use strict";

class Router {
  constructor(onChange) {
    this.onChange = onChange;
    this.pages = {};
    document.querySelectorAll(".page").forEach((page) => {
      this.pages[page.dataset.page] = page;
    });
    this.links = [...document.querySelectorAll("[data-route]")];
    this.current = null;

    window.addEventListener("hashchange", () => this._sync());
  }

  start() {
    this._sync();
  }

  go(route) {
    window.location.hash = `#/${route}`;
  }

  _parse() {
    const raw = (window.location.hash || "").replace(/^#\/?/, "").trim();
    const [page, ...rest] = raw.split("/");
    if (this.pages[page]) return { page, param: rest.join("/") || null };
    return { page: "scripts", param: null };
  }

  _sync() {
    const { page, param } = this._parse();
    const key = `${page}/${param || ""}`;
    if (key === this.current) return;
    this.current = key;

    Object.entries(this.pages).forEach(([id, element]) => {
      element.classList.toggle("is-active", id === page);
    });

    this.links.forEach((link) => link.classList.toggle("is-current", link.dataset.route === page));
    window.scrollTo({ top: 0 });

    if (this.onChange) this.onChange(page, param);
  }
}

window.Router = Router;
