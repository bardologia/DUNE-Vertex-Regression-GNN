"use strict";

class EventScene extends CanvasBase {

  constructor(canvas) {
    super(canvas);
    this.t          = 0;
    this.nextBurst  = 0.6;
    this.photons    = [];
    this.tracks     = [];
    this.flashes    = [];
    this.ambient    = [];
    this.palette    = ["111, 155, 255", "45, 212, 191", "150, 140, 255", "190, 214, 255"];

    this._seedAmbient();

    if (REDUCED_MOTION) {
      this._burst(this.w * 0.7, this.h * 0.4);
      this._step(0);
    } else {
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }
  }

  onResize() {
    if (this.ambient && this.ambient.length) this._seedAmbient();
  }

  _seedAmbient() {
    const count = Math.max(20, Math.min(54, Math.round((this.w * this.h) / 46000)));
    this.ambient = Array.from({ length: count }, () => this._makeAmbient(true));
  }

  _makeAmbient(anywhere) {
    const angle = Math.random() * Math.PI * 2;
    const speed = 0.15 + Math.random() * 0.7;
    return {
      x     : anywhere ? Math.random() * this.w : -10,
      y     : Math.random() * this.h,
      vx    : Math.cos(angle) * speed,
      vy    : Math.sin(angle) * speed,
      size  : 0.6 + Math.random() * 1.1,
      col   : this.palette[(Math.random() * this.palette.length) | 0],
      phase : Math.random() * Math.PI * 2,
      tw    : 0.6 + Math.random() * 1.6,
    };
  }

  _burst(bx, by) {
    const x = bx != null ? bx : this.w * (0.12 + Math.random() * 0.8);
    const y = by != null ? by : this.h * (0.12 + Math.random() * 0.76);
    const core = this.palette[(Math.random() * 2) | 0];

    this.flashes.push({ x, y, r: 0, life: 0.55, maxLife: 0.55, col: "200, 222, 255" });

    const photonCount = 54 + ((Math.random() * 40) | 0);
    for (let i = 0; i < photonCount; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 5.5 + Math.random() * 10;
      const life  = 0.4 + Math.random() * 0.55;
      this.photons.push({
        x, y, px: x, py: y,
        vx   : Math.cos(angle) * speed,
        vy   : Math.sin(angle) * speed,
        life, maxLife: life,
        size : 0.8 + Math.random() * 1.0,
        col  : Math.random() < 0.7 ? core : this.palette[(Math.random() * this.palette.length) | 0],
      });
    }

    const trackCount = 4 + ((Math.random() * 5) | 0);
    for (let i = 0; i < trackCount; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 1.8 + Math.random() * 2.6;
      const life  = 1.4 + Math.random() * 1.8;
      this.tracks.push({
        x, y, angle, speed,
        angVel : (Math.random() - 0.5) * 0.06,
        life, maxLife: life,
        width  : 1.3 + Math.random() * 1.0,
        col    : Math.random() < 0.6 ? core : this.palette[(Math.random() * this.palette.length) | 0],
        hist   : [[x, y]],
      });
    }
  }

  _step(dt) {
    const ctx = this.ctx;
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.globalCompositeOperation = "lighter";

    this._drawAmbient(ctx, dt);
    this._drawFlashes(ctx, dt);
    this._drawTracks(ctx, dt);
    this._drawPhotons(ctx, dt);

    ctx.globalCompositeOperation = "source-over";
  }

  _drawAmbient(ctx, dt) {
    this.ambient.forEach((p) => {
      p.x += p.vx;
      p.y += p.vy;
      p.phase += p.tw * dt;
      if (p.x < -12) p.x = this.w + 12;
      if (p.x > this.w + 12) p.x = -12;
      if (p.y < -12) p.y = this.h + 12;
      if (p.y > this.h + 12) p.y = -12;

      const a = 0.12 + 0.1 * (0.5 + 0.5 * Math.sin(p.phase));
      ctx.beginPath();
      ctx.moveTo(p.x - p.vx * 5, p.y - p.vy * 5);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = `rgba(${p.col}, ${a * 0.7})`;
      ctx.lineWidth = p.size * 0.7;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.col}, ${a})`;
      ctx.fill();
    });
  }

  _drawFlashes(ctx, dt) {
    this.flashes = this.flashes.filter((f) => (f.life -= dt) > 0);
    this.flashes.forEach((f) => {
      const k = f.life / f.maxLife;
      const r = 6 + (1 - k) * 46;
      const grd = ctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, r);
      grd.addColorStop(0, `rgba(${f.col}, ${0.5 * k})`);
      grd.addColorStop(0.4, `rgba(${f.col}, ${0.18 * k})`);
      grd.addColorStop(1, `rgba(${f.col}, 0)`);
      ctx.beginPath();
      ctx.arc(f.x, f.y, r, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();
    });
  }

  _drawTracks(ctx, dt) {
    this.tracks = this.tracks.filter((t) => (t.life -= dt) > 0);
    this.tracks.forEach((t) => {
      t.angle += t.angVel;
      t.x += Math.cos(t.angle) * t.speed;
      t.y += Math.sin(t.angle) * t.speed;
      t.hist.push([t.x, t.y]);
      if (t.hist.length > 16) t.hist.shift();

      const k = t.life / t.maxLife;
      for (let i = 1; i < t.hist.length; i++) {
        const a = (i / t.hist.length) * k * 0.55;
        ctx.beginPath();
        ctx.moveTo(t.hist[i - 1][0], t.hist[i - 1][1]);
        ctx.lineTo(t.hist[i][0], t.hist[i][1]);
        ctx.strokeStyle = `rgba(${t.col}, ${a})`;
        ctx.lineWidth = t.width * (i / t.hist.length);
        ctx.lineCap = "round";
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(t.x, t.y, t.width * 0.9, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${t.col}, ${0.8 * k})`;
      ctx.fill();
    });
  }

  _drawPhotons(ctx, dt) {
    this.photons = this.photons.filter((p) => (p.life -= dt) > 0);
    this.photons.forEach((p) => {
      p.px = p.x;
      p.py = p.y;
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.985;
      p.vy *= 0.985;

      const a = (p.life / p.maxLife);
      ctx.beginPath();
      ctx.moveTo(p.px, p.py);
      ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = `rgba(${p.col}, ${a * 0.9})`;
      ctx.lineWidth = p.size;
      ctx.lineCap = "round";
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * 0.9, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.col}, ${a})`;
      ctx.fill();
    });
  }

  _loop() {
    this.t += 1 / 60;
    if (this.t >= this.nextBurst) {
      this._burst();
      this.nextBurst = this.t + 1.4 + Math.random() * 2.4;
    }
    this._step(1 / 60);
    requestAnimationFrame(this._loop);
  }
}

window.EventScene = EventScene;
