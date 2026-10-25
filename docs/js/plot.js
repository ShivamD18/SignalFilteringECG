// Small canvas plotting helpers: ECG-paper strip charts, log-frequency spectra, beat overlays.

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function robustRange(arrays, i0, i1) {
  let lo = Infinity, hi = -Infinity;
  for (const y of arrays) for (let i = Math.max(0, i0); i < Math.min(y.length, i1); i++) {
    if (y[i] < lo) lo = y[i];
    if (y[i] > hi) hi = y[i];
  }
  if (!Number.isFinite(lo)) return [-1, 1];
  const pad = Math.max((hi - lo) * 0.12, 0.05);
  return [lo - pad, hi + pad];
}

/**
 * Strip chart.
 * spec = { t0, t1, fs, grid: "ecg" | "plain", yRange?, traces: [{y, color, width, alpha, dash}],
 *          markers: [{t, y?, color, label, shape: "dot"|"tick"|"tickDown"|"x", row}], bands: [{t0, t1, color}],
 *          hlines: [{y, color, dash, label}], stepTrace?, yLabel?, selected? }
 */
export function drawStrip(canvas, spec) {
  const { ctx, w, h } = setup(canvas);
  const L = 44, R = 8, T = spec.top ?? 18, B = 22;
  const pw = w - L - R, ph = h - T - B;
  const fs = spec.fs, t0 = spec.t0, t1 = spec.t1;
  const i0 = Math.floor(t0 * fs), i1 = Math.ceil(t1 * fs) + 1;
  const [ylo, yhi] = spec.yRange || robustRange(spec.traces.filter((t) => !t.ignoreRange).map((t) => t.y), i0, i1);
  const X = (t) => L + ((t - t0) / (t1 - t0)) * pw;
  const Y = (v) => T + (1 - (v - ylo) / (yhi - ylo)) * ph;

  ctx.fillStyle = css(spec.grid === "ecg" ? "--paper" : "--plot-bg");
  ctx.fillRect(L, T, pw, ph);

  for (const b of spec.bands || []) {
    ctx.fillStyle = b.color;
    ctx.fillRect(X(Math.max(b.t0, t0)), T, X(Math.min(b.t1, t1)) - X(Math.max(b.t0, t0)), ph);
  }

  // Grid: ECG paper uses 40 ms x 0.1 mV small squares, bold every 5.
  ctx.lineWidth = 1;
  const minor = css(spec.grid === "ecg" ? "--paper-minor" : "--grid");
  const major = css(spec.grid === "ecg" ? "--paper-major" : "--grid-strong");
  const pxPerS = pw / (t1 - t0);
  const tStep = spec.grid === "ecg" ? 0.04 : pxPerS > 60 ? 0.2 : 1;
  if (spec.grid === "ecg" && pxPerS * 0.04 >= 3) {
    for (let k = Math.ceil(t0 / 0.04); k * 0.04 <= t1; k++) {
      ctx.strokeStyle = k % 5 === 0 ? major : minor;
      const x = Math.round(X(k * 0.04)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke();
    }
    const pxPerMv = ph / (yhi - ylo);
    if (pxPerMv * 0.1 >= 3) for (let k = Math.ceil(ylo / 0.1); k * 0.1 <= yhi; k++) {
      ctx.strokeStyle = k % 5 === 0 ? major : minor;
      const y = Math.round(Y(k * 0.1)) + 0.5;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
    }
  } else {
    ctx.strokeStyle = minor;
    const step = spec.grid === "ecg" ? 0.2 : tStep;
    for (let k = Math.ceil(t0 / step); k * step <= t1; k++) {
      const x = Math.round(X(k * step)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke();
    }
  }

  // Axes labels
  ctx.fillStyle = css("--muted");
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  const span = t1 - t0;
  const labelStep = span > 30 ? 10 : span > 12 ? 2 : span > 3 ? 1 : 0.1;
  const offset = spec.relMs ?? 0; // relMs: label time in ms relative to this point (e.g. the R peak)
  for (let k = Math.ceil((t0 - offset) / labelStep - 1e-9); offset + k * labelStep <= t1 + 1e-9; k++) {
    const s = offset + k * labelStep;
    ctx.fillText(spec.relMs !== undefined ? `${Math.round(k * labelStep * 1000)} ms` : `${+s.toFixed(1)} s`, X(s), h - 6);
  }
  ctx.textAlign = "right";
  const yTicks = niceTicks(ylo, yhi, 4);
  for (const v of yTicks) ctx.fillText(fmtTick(v), L - 6, Y(v) + 4);
  if (spec.yLabel) {
    ctx.save(); ctx.translate(11, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
    ctx.fillText(spec.yLabel, 0, 0); ctx.restore();
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(L, T, pw, ph); ctx.clip();
  for (const hl of spec.hlines || []) {
    ctx.strokeStyle = hl.color; ctx.setLineDash(hl.dash || [4, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(L, Y(hl.y)); ctx.lineTo(L + pw, Y(hl.y)); ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const tr of spec.traces) {
    ctx.strokeStyle = tr.color; ctx.lineWidth = tr.width || 1.3; ctx.globalAlpha = tr.alpha ?? 1;
    ctx.setLineDash(tr.dash || []);
    ctx.beginPath();
    const a = Math.max(0, i0), b = Math.min(tr.y.length, i1);
    // Min/max decimation per pixel column keeps QRS spikes visible when zoomed out.
    const perPx = (b - a) / pw;
    if (perPx > 2) {
      for (let px = 0; px < pw; px++) {
        const s = a + Math.floor(px * perPx), e = Math.min(b, a + Math.floor((px + 1) * perPx));
        let mn = Infinity, mx = -Infinity;
        for (let i = s; i < e; i++) { if (tr.y[i] < mn) mn = tr.y[i]; if (tr.y[i] > mx) mx = tr.y[i]; }
        if (px === 0) ctx.moveTo(L + px, Y(mn));
        ctx.lineTo(L + px, Y(mn)); ctx.lineTo(L + px, Y(mx));
      }
    } else {
      for (let i = a; i < b; i++) {
        const x = X(i / fs), y = Y(tr.y[i]);
        if (i === a) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  }
  ctx.font = "bold 11px system-ui, sans-serif";
  ctx.textAlign = "center";
  for (const m of spec.markers || []) {
    if (m.t < t0 || m.t > t1) continue;
    const x = X(m.t);
    ctx.fillStyle = m.color; ctx.strokeStyle = m.color; ctx.lineWidth = 1.5;
    if (m.shape === "dot" && m.y !== undefined) {
      ctx.beginPath(); ctx.arc(x, Y(m.y), m.selected ? 5.5 : 3.5, 0, 2 * Math.PI); ctx.fill();
      if (m.selected) { ctx.lineWidth = 2; ctx.strokeStyle = css("--fg"); ctx.stroke(); }
    } else if (m.shape === "tick") {
      ctx.beginPath(); ctx.moveTo(x, T + ph - 10); ctx.lineTo(x, T + ph); ctx.stroke();
    } else if (m.shape === "x" && m.y !== undefined) {
      const y = Y(m.y);
      ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.stroke();
    }
    if (m.label) {
      const row = m.row || 0;
      ctx.fillText(m.label, x, row >= 0 ? T + 12 + row * 13 : T + ph + (row + 1) * 13 - 3);
    }
  }
  ctx.restore();

  ctx.strokeStyle = css("--grid-strong"); ctx.lineWidth = 1;
  ctx.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);
  return { X, Y, L, T, pw, ph, t0, t1, toTime: (px) => t0 + ((px - L) / pw) * (t1 - t0) };
}

function niceTicks(lo, hi, n) {
  const span = hi - lo, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(+v.toPrecision(10));
  return out;
}
const fmtTick = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2));

/**
 * Log-frequency spectrum. spec = { traces: [{f, db, color, width, alpha, label}], response?: {f, db},
 *   vlines: [{f, color, label}], fmin, fmax, dbRange }
 */
export function drawSpectrum(canvas, spec) {
  const { ctx, w, h } = setup(canvas);
  const L = 44, R = spec.response ? 44 : 10, T = 14, B = 34;
  const pw = w - L - R, ph = h - T - B;
  const fmin = spec.fmin || 0.1, fmax = spec.fmax || 180;
  const [dlo, dhi] = spec.dbRange;
  const X = (f) => L + (Math.log10(f / fmin) / Math.log10(fmax / fmin)) * pw;
  const Y = (d) => T + (1 - (d - dlo) / (dhi - dlo)) * ph;
  const rlo = -80, rhi = 5;
  const Yr = (d) => T + (1 - (Math.max(d, rlo) - rlo) / (rhi - rlo)) * ph;

  ctx.fillStyle = css("--plot-bg"); ctx.fillRect(L, T, pw, ph);
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center";
  for (const f of [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]) {
    if (f < fmin || f > fmax) continue;
    const x = Math.round(X(f)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + ph); ctx.stroke();
    ctx.fillText(`${f}`, x, T + ph + 14);
  }
  ctx.fillText("Frequency (Hz, log scale)", L + pw / 2, h - 4);
  ctx.textAlign = "right";
  for (let d = Math.ceil(dlo / 20) * 20; d <= dhi; d += 20) {
    const y = Math.round(Y(d)) + 0.5;
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
    ctx.fillText(`${d}`, L - 6, y + 4);
  }
  ctx.save(); ctx.translate(11, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
  ctx.fillText("Power (dB)", 0, 0); ctx.restore();

  ctx.save(); ctx.beginPath(); ctx.rect(L, T, pw, ph); ctx.clip();
  for (const b of spec.bands || []) {
    ctx.fillStyle = b.color;
    ctx.fillRect(X(Math.max(b.f0, fmin)), T, X(Math.min(b.f1, fmax)) - X(Math.max(b.f0, fmin)), ph);
  }
  const line = (f, vals, map) => {
    ctx.beginPath();
    let started = false;
    for (let k = 0; k < f.length; k++) {
      if (f[k] < fmin || f[k] > fmax) continue;
      const x = X(f[k]), y = map(vals[k]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  for (const tr of spec.traces) {
    ctx.strokeStyle = tr.color; ctx.lineWidth = tr.width || 1.3; ctx.globalAlpha = tr.alpha ?? 1;
    line(tr.f, tr.db, Y);
    ctx.globalAlpha = 1;
  }
  if (spec.response) {
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2.2; ctx.setLineDash([6, 3]);
    line(spec.response.f, spec.response.db, Yr);
    ctx.setLineDash([]);
  }
  for (const v of spec.vlines || []) {
    ctx.strokeStyle = v.color; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(X(v.f), T); ctx.lineTo(X(v.f), T + ph); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.restore();
  if (spec.response) {
    ctx.fillStyle = css("--accent"); ctx.textAlign = "left";
    for (const d of [0, -20, -40, -60, -80]) ctx.fillText(`${d}`, L + pw + 6, Yr(d) + 4);
    ctx.save(); ctx.translate(w - 6, T + ph / 2); ctx.rotate(Math.PI / 2); ctx.textAlign = "center";
    ctx.fillText("Filter gain (dB)", 0, 0); ctx.restore();
  }
  ctx.strokeStyle = css("--grid-strong"); ctx.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);
}

/** Beat vs template overlay. */
export function drawBeat(canvas, { beat, template, fs, pre, color }) {
  const { ctx, w, h } = setup(canvas);
  const L = 36, R = 8, T = 10, B = 22, pw = w - L - R, ph = h - T - B;
  const all = [...beat, ...template];
  const lo = Math.min(...all) - 0.1, hi = Math.max(...all) + 0.1;
  const X = (i) => L + (i / (beat.length - 1)) * pw;
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * ph;
  ctx.fillStyle = css("--paper"); ctx.fillRect(L, T, pw, ph);
  ctx.strokeStyle = css("--paper-minor"); ctx.lineWidth = 1;
  for (let k = 0; k <= beat.length; k += Math.round(0.04 * fs)) {
    ctx.beginPath(); ctx.moveTo(X(k), T); ctx.lineTo(X(k), T + ph); ctx.stroke();
  }
  ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center";
  for (const ms of [-200, 0, 200, 400]) ctx.fillText(`${ms} ms`, X(pre + (ms / 1000) * fs), h - 6);
  const trace = (y, c, wd, dash) => {
    ctx.strokeStyle = c; ctx.lineWidth = wd; ctx.setLineDash(dash);
    ctx.beginPath(); y.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
    ctx.setLineDash([]);
  };
  trace(template, css("--muted"), 2, [5, 3]);
  trace(beat, color, 2, []);
  ctx.strokeStyle = css("--grid-strong"); ctx.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);
}
