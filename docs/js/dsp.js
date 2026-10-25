// Digital signal processing for the browser, mirroring ecglab/filters.py.
// Filters are cascades of second-order sections [b0, b1, b2, a0, a1, a2] (a0 = 1),
// the same representation SciPy uses, so results match scipy.signal to ~1e-10.

/** Butterworth high-pass or low-pass as second-order sections (bilinear transform, pre-warped). */
export function butterSos(kind, cutoff, fs, order = 2) {
  if (kind !== "highpass" && kind !== "lowpass") throw new Error("kind must be highpass or lowpass");
  if (!(cutoff > 0 && cutoff < fs / 2)) throw new Error(`cutoff ${cutoff} Hz must be between 0 and ${fs / 2} Hz`);
  const K = 2 * fs;
  const wc = K * Math.tan((Math.PI * cutoff) / fs); // pre-warped analog cutoff (rad/s)
  const hp = kind === "highpass";
  const sections = [];
  // Analog prototype poles -exp(j*pi*m/(2N)), m = -N+1, -N+3, ..., N-1; take one of each conjugate pair.
  for (let m = -order + 1; m < 0; m += 2) {
    const c = Math.cos((Math.PI * m) / (2 * order));
    // H(s) = {wc^2 | s^2} / (s^2 + 2 c wc s + wc^2)
    const num = hp ? [1, 0, 0] : [0, 0, wc * wc]; // [s^2, s, 1] coefficients
    const den = [1, 2 * c * wc, wc * wc];
    sections.push(bilinear2(num, den, K));
  }
  if (order % 2 === 1) {
    // Real pole at -wc: H(s) = {s | wc} / (s + wc)
    const num = hp ? [1, 0] : [0, wc];
    sections.push(bilinear1(num, [1, wc], K));
  }
  return sections;
}

function bilinear2([b2, b1, b0], [a2, a1, a0], K) {
  const K2 = K * K;
  const n = [b2 * K2 + b1 * K + b0, 2 * (b0 - b2 * K2), b2 * K2 - b1 * K + b0];
  const d = [a2 * K2 + a1 * K + a0, 2 * (a0 - a2 * K2), a2 * K2 - a1 * K + a0];
  return [n[0] / d[0], n[1] / d[0], n[2] / d[0], 1, d[1] / d[0], d[2] / d[0]];
}

function bilinear1([b1, b0], [a1, a0], K) {
  const d0 = a1 * K + a0;
  return [(b1 * K + b0) / d0, (b0 - b1 * K) / d0, 0, 1, (a0 - a1 * K) / d0, 0];
}

/** Second-order IIR notch, identical to scipy.signal.iirnotch. */
export function notchSos(f0, fs, q = 30) {
  const w0 = (2 * Math.PI * f0) / fs;
  const bw = w0 / q;
  const beta = Math.tan(bw / 2); // gain at band edges = 1/sqrt(2)
  const gain = 1 / (1 + beta);
  const cw = Math.cos(w0);
  return [[gain, -2 * gain * cw, gain, 1, -2 * gain * cw, 2 * gain - 1]];
}

/** Build the full conditioning chain (high-pass -> low-pass -> notch), like FilterConfig.sos(). */
export function chainSos(cfg, fs) {
  const s = [];
  if (cfg.highpass_hz) s.push(...butterSos("highpass", cfg.highpass_hz, fs, cfg.highpass_order));
  if (cfg.lowpass_hz && cfg.lowpass_hz < fs / 2) s.push(...butterSos("lowpass", cfg.lowpass_hz, fs, cfg.lowpass_order));
  if (cfg.notch_hz && cfg.notch_hz < fs / 2) s.push(...notchSos(cfg.notch_hz, fs, cfg.notch_q));
  return s.length ? s : [[1, 0, 0, 1, 0, 0]];
}

/** Direct-form II transposed cascade (scipy.signal.sosfilt). `zi` is modified in place if given. */
export function sosfilt(sos, x, zi = null) {
  const y = Float64Array.from(x);
  for (let s = 0; s < sos.length; s++) {
    const [b0, b1, b2, , a1, a2] = sos[s];
    let z0 = zi ? zi[s][0] : 0;
    let z1 = zi ? zi[s][1] : 0;
    for (let n = 0; n < y.length; n++) {
      const xn = y[n];
      const yn = b0 * xn + z0;
      z0 = b1 * xn - a1 * yn + z1;
      z1 = b2 * xn - a2 * yn;
      y[n] = yn;
    }
    if (zi) zi[s] = [z0, z1];
  }
  return y;
}

/** Initial conditions for a step response steady state (scipy.signal.sosfilt_zi). */
export function sosfiltZi(sos) {
  let scale = 1;
  return sos.map(([b0, b1, b2, , a1, a2]) => {
    const B0 = b1 - a1 * b0;
    const B1 = b2 - a2 * b0;
    const z0 = (B0 + B1) / (1 + a1 + a2);
    const zi = [scale * z0, scale * (B1 - a2 * z0)];
    scale *= (b0 + b1 + b2) / (1 + a1 + a2);
    return zi;
  });
}

const scaled = (zi, v) => zi.map(([a, b]) => [a * v, b * v]);

/** Forward-backward zero-phase filtering with odd-extension padding (scipy.signal.sosfiltfilt). */
export function sosfiltfilt(sos, x) {
  const n = x.length;
  let ntaps = 2 * sos.length + 1;
  ntaps -= Math.min(sos.filter((s) => s[2] === 0).length, sos.filter((s) => s[5] === 0).length);
  const edge = Math.min(3 * ntaps, n - 1);
  const ext = new Float64Array(n + 2 * edge);
  for (let i = 0; i < edge; i++) ext[i] = 2 * x[0] - x[edge - i];
  for (let i = 0; i < n; i++) ext[edge + i] = x[i];
  for (let i = 0; i < edge; i++) ext[edge + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  const zi = sosfiltZi(sos);
  let y = sosfilt(sos, ext, scaled(zi, ext[0]));
  y.reverse();
  y = sosfilt(sos, y, scaled(zi, y[0]));
  y.reverse();
  return y.slice(edge, edge + n);
}

/** Causal (monitor-style) filtering starting from the steady state of x[0]. */
export function sosfiltCausal(sos, x) {
  return sosfilt(sos, x, scaled(sosfiltZi(sos), x[0]));
}

export function applyChain(x, cfg, fs) {
  const sos = chainSos(cfg, fs);
  return cfg.zero_phase ? sosfiltfilt(sos, x) : sosfiltCausal(sos, x);
}

/** Magnitude response in dB at the given frequencies; squared for zero-phase filtering. */
export function freqzDb(sos, fs, freqs, zeroPhase = true) {
  return freqs.map((f) => {
    const w = (2 * Math.PI * f) / fs;
    const c1 = Math.cos(w), s1 = Math.sin(w), c2 = Math.cos(2 * w), s2 = Math.sin(2 * w);
    let mag = 1;
    for (const [b0, b1, b2, a0, a1, a2] of sos) {
      const nr = b0 + b1 * c1 + b2 * c2, ni = -(b1 * s1 + b2 * s2);
      const dr = a0 + a1 * c1 + a2 * c2, di = -(a1 * s1 + a2 * s2);
      mag *= Math.sqrt((nr * nr + ni * ni) / (dr * dr + di * di));
    }
    if (zeroPhase) mag *= mag;
    return 20 * Math.log10(Math.max(mag, 1e-12));
  });
}

/** Two cascaded median filters (200 ms, 600 ms) estimating baseline wander (de Chazal 2004). */
export function medianBaseline(x, fs, windows = [0.2, 0.6]) {
  let base = Float64Array.from(x);
  for (const w of windows) base = medianFilter(base, Math.round(w * fs) | 1);
  return base;
}

function medianFilter(x, size) {
  // Sliding median with 'nearest' edge handling (scipy.ndimage.median_filter mode="nearest").
  const n = x.length, h = (size - 1) >> 1, out = new Float64Array(n);
  const win = [];
  const at = (i) => x[Math.min(n - 1, Math.max(0, i))];
  for (let i = -h; i <= h; i++) insertSorted(win, at(i));
  for (let i = 0; i < n; i++) {
    out[i] = win[h];
    removeSorted(win, at(i - h));
    insertSorted(win, at(i + h + 1));
  }
  return out;
}

function lowerBound(a, v) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
  return lo;
}
const insertSorted = (a, v) => a.splice(lowerBound(a, v), 0, v);
const removeSorted = (a, v) => a.splice(lowerBound(a, v), 1);

/** In-place iterative radix-2 FFT. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        [cr, ci] = [cr * wr - ci * wi, cr * wi + ci * wr];
      }
    }
  }
}

/** Welch PSD (Hann window, 50 % overlap, mean-detrended segments), returns {f, p} in mV^2/Hz. */
export function welch(x, fs, nperseg = 4096) {
  nperseg = Math.min(nperseg, 1 << Math.floor(Math.log2(x.length)));
  const step = nperseg / 2, win = new Float64Array(nperseg);
  let wss = 0;
  for (let i = 0; i < nperseg; i++) { win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nperseg); wss += win[i] ** 2; }
  const p = new Float64Array(nperseg / 2 + 1);
  let segs = 0;
  for (let s = 0; s + nperseg <= x.length; s += step) {
    let mean = 0;
    for (let i = 0; i < nperseg; i++) mean += x[s + i];
    mean /= nperseg;
    const re = new Float64Array(nperseg), im = new Float64Array(nperseg);
    for (let i = 0; i < nperseg; i++) re[i] = (x[s + i] - mean) * win[i];
    fft(re, im);
    for (let k = 0; k < p.length; k++) p[k] += re[k] * re[k] + im[k] * im[k];
    segs++;
  }
  const f = new Float64Array(p.length);
  for (let k = 0; k < p.length; k++) {
    const onesided = k === 0 || k === nperseg / 2 ? 1 : 2;
    p[k] = (onesided * p[k]) / (fs * wss * segs);
    f[k] = (k * fs) / nperseg;
  }
  return { f, p };
}

export function snrDb(clean, test) {
  const mc = mean(clean), mt = mean(test);
  let ps = 0, pe = 0;
  for (let i = 0; i < clean.length; i++) {
    const c = clean[i] - mc;
    ps += c * c;
    pe += (test[i] - mt - c) ** 2;
  }
  return 10 * Math.log10(ps / Math.max(pe, 1e-20));
}

export function mean(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
}

/** Linear-interpolation resampling (low-pass first when downsampling). */
export function resampleLinear(x, fsIn, fsOut) {
  if (fsIn === fsOut) return Float64Array.from(x);
  const nOut = Math.floor(((x.length - 1) * fsOut) / fsIn) + 1;
  const y = new Float64Array(nOut);
  for (let i = 0; i < nOut; i++) {
    const t = (i * fsIn) / fsOut, k = Math.floor(t), fr = t - k;
    y[i] = k + 1 < x.length ? x[k] * (1 - fr) + x[k + 1] * fr : x[x.length - 1];
  }
  return y;
}
