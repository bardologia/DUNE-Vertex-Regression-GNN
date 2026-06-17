"use strict";

window.apiGet = async function (url) {
  const response = await fetch(url);
  if (!response.ok && response.status >= 500) {
    return { ok: false, error: `server ${response.status}` };
  }
  return response.json();
};

window.apiPost = async function (url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return response.json();
};

let toastTimer = null;
window.toast = function (message, kind) {
  const element = document.getElementById("toast");
  element.textContent = message;
  element.className = "toast is-show" + (kind ? ` is-${kind}` : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    element.className = "toast";
  }, 3200);
};

window.formatBytes = function (value) {
  if (!value && value !== 0) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(amount >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`;
};

window.barClass = function (percent) {
  if (percent >= 90) return "bar__fill--danger";
  if (percent >= 70) return "bar__fill--warn";
  return "";
};

window.escapeHtml = function (value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
};
