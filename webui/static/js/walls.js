"use strict";

const WALL_DEFS = [
  { axis: 0, sign: -1, label: "−X" },
  { axis: 0, sign:  1, label: "+X" },
  { axis: 1, sign: -1, label: "−Y" },
  { axis: 1, sign:  1, label: "+Y" },
  { axis: 2, sign: -1, label: "−Z" },
  { axis: 2, sign:  1, label: "+Z" },
];

function wallExtents(detectorMin, detectorMax) {
  const ext = [1, 1, 1];
  for (let axis = 0; axis < 3; axis++) {
    ext[axis] = Math.max(Math.abs(detectorMin[axis]), Math.abs(detectorMax[axis]), 1e-6);
  }
  return ext;
}

function classifyWall(point, ext) {
  let bestAxis = 0;
  let bestScore = -Infinity;
  for (let axis = 0; axis < 3; axis++) {
    const score = Math.abs(point[axis]) / ext[axis];
    if (score > bestScore) { bestScore = score; bestAxis = axis; }
  }
  return bestAxis * 2 + (point[bestAxis] < 0 ? 0 : 1);
}

class WallToggle {

  constructor(container, onChange) {
    this.onChange = onChange;
    this.flags = WALL_DEFS.map(() => true);
    this.buttons = [];

    container.innerHTML = "";
    WALL_DEFS.forEach((wall, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ev-wall is-active";
      button.textContent = wall.label;
      button.title = `toggle ${wall.label} detector wall`;
      button.addEventListener("click", () => this._toggle(index));
      container.appendChild(button);
      this.buttons.push(button);
    });
  }

  _toggle(index) {
    this.flags[index] = !this.flags[index];
    this.buttons[index].classList.toggle("is-active", this.flags[index]);
    this.onChange(this.flags);
  }
}

window.WALL_DEFS = WALL_DEFS;
window.wallExtents = wallExtents;
window.classifyWall = classifyWall;
window.WallToggle = WallToggle;
