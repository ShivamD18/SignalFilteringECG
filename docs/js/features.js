// Beat feature extraction, mirroring ecglab/features.py exactly (fs must be 360 Hz).

export const FS = 360;
const PRE = Math.round(0.25 * FS);   // 90 samples before R
const POST = Math.round(0.45 * FS);  // 162 samples after R
const QH = Math.round(0.1 * FS);     // QRS half-width
const WH = Math.round(0.15 * FS);    // width-measurement half-window
const LOCAL = 10, LONG = 100, TEMPLATE = 30, STEP = 10;
export const WINDOW = { pre: PRE, post: POST, qrsHalf: QH };

function median(a) {
  const s = Float64Array.from(a).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function ptp(a, lo, hi) {
  let mn = Infinity, mx = -Infinity;
  for (let i = lo; i < hi; i++) { if (a[i] < mn) mn = a[i]; if (a[i] > mx) mx = a[i]; }
  return mx - mn;
}

function corr(a, b, lo, hi) {
  const n = hi - lo;
  let ma = 0, mb = 0;
  for (let i = lo; i < hi; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = lo; i < hi; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    sab += x * y; saa += x * x; sbb += y * y;
  }
  const den = Math.sqrt(saa * sbb);
  return den > 0 ? sab / den : 0;
}

function energyWidth(w, lo, hi) {
  const c = [];
  let acc = 0;
  for (let i = lo + 1; i < hi; i++) { acc += (w[i] - w[i - 1]) ** 2; c.push(acc); }
  if (!(acc > 0)) return 0;
  const first = (v) => c.findIndex((x) => x >= v);
  return (first(0.9 * acc) - first(0.1 * acc)) / FS;
}

/** Fixed windows around each beat, minus their median. Returns {idx, windows}. */
export function beatWindows(ecg, r) {
  const idx = [], windows = [];
  r.forEach((p, i) => {
    if (p - PRE < 0 || p + POST >= ecg.length) return;
    const w = Float64Array.from(ecg.subarray ? ecg.subarray(p - PRE, p + POST + 1) : ecg.slice(p - PRE, p + POST + 1));
    const m = median(w);
    for (let k = 0; k < w.length; k++) w[k] -= m;
    idx.push(i);
    windows.push(w);
  });
  return { idx, windows };
}

/** Sample-wise median of the 61 surrounding beats: the patient's dominant beat. */
export function beatTemplates(windows) {
  const m = windows.length, L = windows.length ? windows[0].length : 0;
  return windows.map((_, i) => {
    const lo = Math.max(0, i - TEMPLATE), hi = Math.min(m, i + TEMPLATE + 1);
    const t = new Float64Array(L), col = new Float64Array(hi - lo);
    for (let k = 0; k < L; k++) {
      for (let j = lo; j < hi; j++) col[j - lo] = windows[j][k];
      t[k] = median(col);
    }
    return t;
  });
}

export const FEATURE_COUNT = 10 + 10 + Math.floor((PRE + POST) / STEP) + 1;

/** Returns {X: Float64Array[] rows, beatIdx: indices into r, windows, templates}. */
export function extractFeatures(ecg, r) {
  const n = r.length;
  const { idx: winIdx, windows } = beatWindows(ecg, r);
  if (!winIdx.length || n < 3) return { X: [], beatIdx: [], windows: [], templates: [] };
  const allT = beatTemplates(windows);
  const rr = [];
  for (let i = 1; i < n; i++) rr.push((r[i] - r[i - 1]) / FS);

  const X = [], beatIdx = [], W = [], T = [];
  winIdx.forEach((i, k) => {
    if (i < 1 || i > n - 2) return;
    const w = windows[k], t = allT[k];
    const pre = rr[i - 1], post = rr[i];
    const local = median(rr.slice(Math.max(0, i - LOCAL), Math.min(n - 1, i + LOCAL)));
    const long = median(rr.slice(Math.max(0, i - LONG), Math.min(n - 1, i + LONG)));
    const prev = i >= 2 ? rr[i - 2] : local;

    const q0 = PRE - QH, q1 = PRE + QH + 1, t1 = PRE + POST + 1;
    const ampT = Math.max(ptp(t, q0, q1), 1e-3);
    const ampW = ptp(w, q0, q1);
    const widthW = energyWidth(w, PRE - WH, PRE + WH + 1);
    const widthT = energyWidth(t, PRE - WH, PRE + WH + 1);
    let se = 0, ew = 0, et = 0;
    for (let j = q0; j < q1; j++) { se += (w[j] - t[j]) ** 2; ew += w[j] ** 2; et += t[j] ** 2; }

    const row = [
      pre, post, local, long, pre / local, post / local, post / pre, local / long, pre / prev, pre - local,
      corr(w, t, q0, q1), corr(w, t, 0, t1), corr(w, t, PRE + QH, t1),
      ampW / ampT, Math.sqrt(se / (q1 - q0)) / ampT, ew / Math.max(et, 1e-9), w[PRE] / ampT,
      widthW, widthW / Math.max(widthT, 1 / FS), ampW,
    ];
    for (let j = 0; j < w.length; j += STEP) row.push(w[j] / ampT);
    X.push(row);
    beatIdx.push(i);
    W.push(w);
    T.push(t);
  });
  return { X, beatIdx, windows: W, templates: T };
}
