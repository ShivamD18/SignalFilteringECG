// ECG Signal Lab: page controller. Pipeline: record -> + noise -> filter -> Pan-Tompkins -> features -> model -> report.
import { applyChain, butterSos, chainSos, freqzDb, medianBaseline, resampleLinear, snrDb, sosfiltfilt, welch } from "./dsp.js";
import { matchBeats, panTompkins, refinePeaks } from "./qrs.js";
import { FS, WINDOW, beatWindows, extractFeatures } from "./features.js";
import { predictProba } from "./model.js";
import { findings, rhythmMetrics, verdict } from "./report.js";
import { drawBeat, drawSpectrum, drawStrip } from "./plot.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const CLASSES = ["N", "S", "V", "F"];
const CLASS_TEXT = {
  N: "Normal, including bundle branch block beats",
  S: "Supraventricular ectopic, e.g. a premature atrial contraction",
  V: "Ventricular ectopic, e.g. a premature ventricular contraction (PVC)",
  F: "Fusion of a ventricular and a normal beat",
  Q: "Paced or unclassifiable",
  "?": "Not classified (first or last beat)",
};
const colorOf = (c) => cssVar(`--${c === "?" ? "Q" : c}`);
const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "-" : `${(100 * v).toFixed(d)}%`);
const fmt = (v, d = 0, unit = "") => (v === null || v === undefined || Number.isNaN(v) ? "-" : `${v.toFixed(d)}${unit}`);
const MODEL_NAMES = {
  gradient_boosting: "Gradient-boosted trees",
  random_forest: "Random forest",
  logistic_regression: "Logistic regression",
  mlp: "Neural network (MLP)",
};

const TRAINING_FILTER = { method: "butter", highpass_hz: 0.5, highpass_order: 2, lowpass_on: true, lowpass_hz: 40,
  lowpass_order: 4, notch_on: false, notch_hz: 60, notch_q: 30, zero_phase: true };
const FILTER_PRESETS = {
  training: { ...TRAINING_FILTER },
  diagnostic: { ...TRAINING_FILTER, highpass_hz: 0.05, highpass_order: 1, lowpass_hz: 150, lowpass_order: 2, notch_on: true },
  monitor: { ...TRAINING_FILTER, zero_phase: false },
  harsh: { ...TRAINING_FILTER, highpass_hz: 5, lowpass_hz: 15 },
  median: { ...TRAINING_FILTER, method: "median" },
};
const NOISE_PRESETS = {
  clean: { bw: 0, ma: 0, em: 0, pl: 0 },
  breathing: { bw: 1.5, ma: 0, em: 0, pl: 0 },
  muscle: { bw: 0, ma: 1, em: 0, pl: 0 },
  motion: { bw: 0, ma: 0, em: 1, pl: 0 },
  mains: { bw: 0, ma: 0, em: 0, pl: 0.25 },
  all: { bw: 1, ma: 0.6, em: 0.5, pl: 0.15 },
};
const REF_SOS = [...butterSos("highpass", 0.5, FS, 2), ...butterSos("lowpass", 100, FS, 4)];
const STAGE = { NOISE: 0, FILTER: 1 };

const state = {
  index: [], model: null, metrics: null, noiseRec: {},
  rec: null,
  noise: { bw: 0, ma: 0, em: 0, pl: 0, plHz: 60 },
  filt: { ...TRAINING_FILTER },
  view: { t0: 0, span: 10 },
  selected: null,
  d: {},
};

// ---------- helpers ----------
const hpToSlider = (hz) => 50 * Math.log10(hz / 0.05);
const sliderToHp = (v) => +(0.05 * 100 ** (v / 100)).toPrecision(2);
const rms = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s / a.length); };
function median(a) { const s = Float64Array.from(a).sort(), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
const toDb = (p) => Array.from(p, (v) => 10 * Math.log10(Math.max(v, 1e-14)));

async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

function medianBeat(x, peaks) {
  const { windows } = beatWindows(x, peaks);
  if (!windows.length) return null;
  const L = windows[0].length, out = new Float64Array(L), col = new Float64Array(windows.length);
  for (let k = 0; k < L; k++) {
    for (let j = 0; j < windows.length; j++) col[j] = windows[j][k];
    out[k] = median(col);
  }
  return out;
}

function qrsPtp(x, p) {
  let mn = Infinity, mx = -Infinity;
  for (let i = Math.max(0, p - 22); i < Math.min(x.length, p + 23); i++) { mn = Math.min(mn, x[i]); mx = Math.max(mx, x[i]); }
  return mx - mn;
}

// ---------- pipeline ----------
function filterConfig() {
  const f = state.filt;
  return {
    highpass_hz: f.method === "butter" ? f.highpass_hz : null, highpass_order: f.highpass_order,
    lowpass_hz: f.lowpass_on ? f.lowpass_hz : null, lowpass_order: f.lowpass_order,
    notch_hz: f.notch_on ? f.notch_hz : null, notch_q: f.notch_q, zero_phase: f.zero_phase,
  };
}

function filterSignal(x) {
  let y = x;
  if (state.filt.method === "median") {
    const base = medianBaseline(y, FS);
    y = y.map((v, i) => v - base[i]);
  }
  return applyChain(y, filterConfig(), FS);
}

function setRecord(rec) {
  rec.ref = sosfiltfilt(REF_SOS, rec.clean);
  rec.refSamples = rec.beats.map((b) => b[0]);
  rec.refPeaks = rec.beats.length ? refinePeaks(rec.ref, rec.refSamples, FS) : panTompkins(rec.ref, FS).rPeaks;
  const normals = rec.beats.length ? rec.refPeaks.filter((_, i) => rec.beats[i][2] === "N") : rec.refPeaks;
  rec.avgPeaks = normals.length >= 3 ? normals : rec.refPeaks;
  rec.refTemplate = medianBeat(rec.ref, rec.avgPeaks);
  rec.psdRef = welch(rec.ref, FS);
  state.rec = rec;
  state.selected = null;
  state.view.t0 = 0;
  syncViewControls();
  syncRecordUI();
  compute(STAGE.NOISE);
}

function compute(stage) {
  const { rec, d } = state;
  const n = rec.clean.length;
  if (stage <= STAGE.NOISE) {
    const g = state.noise, noise = new Float64Array(n);
    for (const k of ["bw", "ma", "em"]) {
      if (!g[k]) continue;
      const src = state.noiseRec[k];
      for (let i = 0; i < n; i++) noise[i] += g[k] * src[i % src.length];
    }
    if (g.pl) for (let i = 0; i < n; i++) noise[i] += g.pl * Math.sin((2 * Math.PI * g.plHz * i) / FS);
    d.noise = noise;
    d.anyNoise = g.bw || g.ma || g.em || g.pl;
    d.noisy = rec.clean.map((v, i) => v + noise[i]);
    d.psdNoise = d.anyNoise ? welch(noise, FS) : null;
    d.inSnr = snrDb(rec.ref, d.noisy);
  }
  d.filtered = filterSignal(d.noisy);
  d.filteredClean = filterSignal(rec.clean);
  d.psdNoisy = welch(d.noisy, FS);
  d.psdFilt = welch(d.filtered, FS);
  d.outSnr = snrDb(rec.ref, d.filtered);
  d.resid = rms(d.filtered, d.filteredClean);
  d.dist = rms(d.filteredClean, rec.ref);
  d.rKept = median(rec.refPeaks.map((p) => qrsPtp(d.filteredClean, p) / Math.max(qrsPtp(rec.ref, p), 1e-6)));
  d.beatFiltered = medianBeat(d.filteredClean, rec.avgPeaks);

  // Detection on the user's filtered signal
  d.pt = panTompkins(d.filtered, FS);
  const r = d.pt.rPeaks;
  d.match = rec.beats.length ? matchBeats(rec.refSamples, r, FS) : null;

  // Classification with the exported model
  const feats = extractFeatures(d.filtered, r);
  d.feats = feats;
  d.probs = feats.X.map((row) => predictProba(state.model, row));
  d.labels = r.map(() => "?");
  feats.beatIdx.forEach((i, k) => { d.labels[i] = CLASSES[d.probs[k].indexOf(Math.max(...d.probs[k]))]; });
  const qi = state.model.features.indexOf("corr_qrs");
  d.quality = feats.X.length ? median(feats.X.map((row) => row[qi])) : null;
  d.metrics = rhythmMetrics(r, d.labels, FS);
  d.findings = findings(d.metrics, d.quality);
  d.verdict = verdict(d.findings);

  d.agree = null;
  if (d.match) {
    const pairs = [];
    d.match.matched.forEach((j, i) => {
      if (j >= 0 && d.labels[i] !== "?" && CLASSES.includes(rec.beats[j][2])) pairs.push([rec.beats[j][2], d.labels[i]]);
    });
    const cm = CLASSES.map(() => CLASSES.map(() => 0));
    pairs.forEach(([t, p]) => cm[CLASSES.indexOf(t)][CLASSES.indexOf(p)]++);
    d.agree = { n: pairs.length, exact: pairs.filter(([t, p]) => t === p).length,
      binary: pairs.filter(([t, p]) => (t === "N") === (p === "N")).length, cm };
  }
  if (state.selected === null || state.selected >= r.length) state.selected = defaultSelection();
  render();
}

function defaultSelection() {
  const { labels, pt } = state.d;
  const t0 = state.view.t0 * FS, t1 = (state.view.t0 + state.view.span) * FS;
  const inView = pt.rPeaks.map((p, i) => [p, i]).filter(([p]) => p >= t0 && p <= t1);
  const abnormal = inView.find(([, i]) => labels[i] !== "N" && labels[i] !== "?");
  const first = inView.find(([, i]) => labels[i] !== "?");
  return (abnormal || first || [0, null])[1];
}

// ---------- rendering ----------
// Background tabs pause requestAnimationFrame; a timer races it so work still completes.
function nextFrame(fn) {
  let done = false;
  const run = () => { if (!done) { done = true; fn(); } };
  requestAnimationFrame(run);
  setTimeout(run, 100);
  return true;
}
let renderQueued = false;
function render() {
  if (renderQueued) return;
  renderQueued = true;
  nextFrame(() => {
    renderQueued = false;
    try {
      renderSignal(); renderNoise(); renderFilter(); renderDetect(); renderClassify(); renderInspector(); renderReport();
    } catch (err) { showError(err); }
  });
}

function view() {
  return { t0: state.view.t0, t1: Math.min(state.view.t0 + state.view.span, state.rec.clean.length / FS), fs: FS };
}

function renderSignal() {
  const { rec } = state;
  const markers = rec.beats.map(([s, sym, cls]) => ({ t: s / FS, label: cls, color: colorOf(cls), row: 0 }));
  drawStrip($("#plot-raw"), { ...view(), grid: "ecg", yLabel: "mV", top: 22,
    traces: [{ y: rec.clean, color: cssVar("--trace"), width: 1.2 }], markers });
  if (rec.refTemplate) {
    const pre = WINDOW.pre / FS;
    const band = (a, b, c) => ({ t0: pre + a, t1: pre + b, color: cssVar(c) });
    drawStrip($("#plot-avg"), { t0: 0, t1: rec.refTemplate.length / FS, fs: FS, grid: "ecg", relMs: pre, top: 22,
      traces: [{ y: rec.refTemplate, color: cssVar("--trace"), width: 2 }],
      bands: [band(-0.22, -0.08, "--band-p"), band(-0.05, 0.06, "--band-qrs"), band(0.06, 0.16, "--band-st"), band(0.16, 0.42, "--band-t")],
      markers: [["P", -0.15], ["QRS", 0], ["ST", 0.11], ["T", 0.29]].map(([label, t]) => ({ t: pre + t, label, color: cssVar("--muted"), row: 0 })) });
  }
}

function spectrumRange(psds) {
  const mx = Math.max(...psds.filter(Boolean).flatMap((p) => toDb(p.p).filter((_, k) => p.f[k] >= 0.1)));
  const top = Math.ceil((mx + 5) / 10) * 10;
  return [top - 100, top];
}

function renderNoise() {
  const { rec, d, noise } = state;
  const traces = [];
  if ($("#show-clean-noise").checked) traces.push({ y: rec.clean, color: cssVar("--ref"), width: 1, alpha: 0.7, ignoreRange: true });
  traces.push({ y: d.noisy, color: cssVar("--trace"), width: 1.1 });
  drawStrip($("#plot-noisy"), { ...view(), grid: "ecg", yLabel: "mV", traces });
  const specTraces = [{ f: rec.psdRef.f, db: toDb(rec.psdRef.p), color: cssVar("--ref"), width: 1.6 }];
  if (d.psdNoise) specTraces.push({ f: d.psdNoise.f, db: toDb(d.psdNoise.p), color: cssVar("--V"), width: 1.4, alpha: 0.9 });
  drawSpectrum($("#spec-noise"), { traces: specTraces, dbRange: spectrumRange([rec.psdRef, d.psdNoise]),
    vlines: noise.pl ? [{ f: noise.plHz, color: cssVar("--V") }] : [] });
  const el = $("#s-insnr");
  el.textContent = d.anyNoise ? `${d.inSnr.toFixed(1)} dB` : "clean";
}

function renderFilter() {
  const { rec, d, filt } = state;
  const traces = [];
  if ($("#show-noisy").checked) traces.push({ y: d.noisy, color: cssVar("--noisy"), width: 1, alpha: 0.75, ignoreRange: true });
  if ($("#show-ref").checked) traces.push({ y: rec.ref, color: cssVar("--ref"), width: 1.4, dash: [5, 3], ignoreRange: true });
  traces.push({ y: d.filtered, color: cssVar("--trace"), width: 1.3 });
  drawStrip($("#plot-filtered"), { ...view(), grid: "ecg", yLabel: "mV", traces });

  const cfg = filterConfig();
  const freqs = Array.from({ length: 400 }, (_, k) => 0.1 * (1800 ** (k / 399)));
  const vlines = [];
  if (cfg.highpass_hz) vlines.push({ f: cfg.highpass_hz, color: cssVar("--accent") });
  if (cfg.lowpass_hz) vlines.push({ f: cfg.lowpass_hz, color: cssVar("--accent") });
  drawSpectrum($("#spec-filter"), {
    traces: [
      { f: d.psdNoisy.f, db: toDb(d.psdNoisy.p), color: cssVar("--noisy"), width: 1.3 },
      { f: d.psdFilt.f, db: toDb(d.psdFilt.p), color: cssVar("--trace"), width: 1.4 },
    ],
    response: { f: freqs, db: freqzDb(chainSos(cfg, FS), FS, freqs, cfg.zero_phase) },
    dbRange: spectrumRange([d.psdNoisy, d.psdFilt]), vlines,
  });
  if (d.beatFiltered && rec.refTemplate)
    drawBeat($("#plot-beat-dist"), { beat: d.beatFiltered, template: rec.refTemplate, fs: FS, pre: WINDOW.pre, color: cssVar("--accent-2") });

  const setStat = (id, text, good) => { const el = $(id); el.textContent = text; el.className = good === undefined ? "" : good ? "good" : "bad"; };
  setStat("#s-outsnr", `${d.outSnr.toFixed(1)} dB`);
  const gain = d.outSnr - d.inSnr;
  setStat("#s-gain", `${gain >= 0 ? "+" : ""}${gain.toFixed(1)} dB`, gain >= 0);
  setStat("#s-resid", `${(1000 * d.resid).toFixed(0)} µV`);
  setStat("#s-dist", `${(1000 * d.dist).toFixed(0)} µV`, d.dist < 0.05);
  setStat("#s-ramp", pct(d.rKept, 0), d.rKept > 0.85);
  $$("[data-for='butter']").forEach((el) => { el.hidden = filt.method !== "butter"; });
}

function renderDetect() {
  const { rec, d } = state;
  const r = d.pt.rPeaks, v = view();
  const markers = [];
  if (d.match) {
    r.forEach((p, i) => {
      const ok = d.match.matched[i] >= 0;
      markers.push({ t: p / FS, y: d.filtered[p], shape: ok ? "dot" : "x", color: cssVar(ok ? "--N" : "--V") });
    });
    const hit = new Set(Array.from(d.match.matched).filter((j) => j >= 0));
    rec.refSamples.forEach((s, j) => { if (!hit.has(j)) markers.push({ t: s / FS, shape: "tick", color: cssVar("--S"), label: "miss", row: -1 }); });
  } else r.forEach((p) => markers.push({ t: p / FS, y: d.filtered[p], shape: "dot", color: cssVar("--N") }));
  const sm = { ...v, grid: "plain", top: 6 };
  drawStrip($("#pt-ecg"), { ...sm, traces: [{ y: d.filtered, color: cssVar("--trace") }], markers });
  drawStrip($("#pt-bp"), { ...sm, traces: [{ y: d.pt.bandpassed, color: cssVar("--accent-2") }] });
  drawStrip($("#pt-sq"), { ...sm, traces: [{ y: d.pt.squared, color: cssVar("--S") }] });
  const thr = new Float64Array(d.pt.integrated.length);
  const c = d.pt.candidates;
  for (let k = 0; k < c.length; k++) thr.fill(d.pt.thresholds[k], k ? c[k] : 0, k + 1 < c.length ? c[k + 1] : thr.length);
  drawStrip($("#pt-mwi"), { ...sm, traces: [
    { y: d.pt.integrated, color: cssVar("--F") },
    { y: thr, color: cssVar("--accent"), dash: [5, 3], width: 1.2, ignoreRange: true }],
    markers: d.pt.qrsMwi.map((p) => ({ t: p / FS, y: d.pt.integrated[p], shape: "dot", color: cssVar("--N") })) });

  $("#s-det").textContent = `${r.length} beats`;
  $("#s-se").textContent = d.match ? pct(d.match.sensitivity) : "no labels";
  $("#s-ppv").textContent = d.match ? pct(d.match.ppv) : "no labels";
}

function renderClassify() {
  const { rec, d } = state;
  const r = d.pt.rPeaks;
  const markers = r.map((p, i) => ({ t: p / FS, y: d.filtered[p], shape: "dot", color: colorOf(d.labels[i]),
    label: d.labels[i], row: 0, selected: i === state.selected }));
  if (d.match) d.match.matched.forEach((j, i) => {
    if (j >= 0) markers.push({ t: r[i] / FS, label: rec.beats[j][2], color: colorOf(rec.beats[j][2]), row: -1 });
  });
  state.classPlot = drawStrip($("#plot-class"), { ...view(), grid: "ecg", yLabel: "mV", top: 22,
    traces: [{ y: d.filtered, color: cssVar("--trace"), width: 1.2 }], markers });
  $("#seg-agree").textContent = d.agree && d.agree.n
    ? `Over the whole minute the model agrees with the cardiologists on ${pct(d.agree.exact / d.agree.n)} of ${d.agree.n} beats (normal-vs-abnormal: ${pct(d.agree.binary / d.agree.n)}).`
    : rec.uploaded ? "Your file has no cardiologist labels to compare with." : "";
}

function renderInspector() {
  const { d } = state;
  const i = state.selected;
  const k = i === null ? -1 : d.feats.beatIdx.indexOf(i);
  if (k < 0) {
    $("#insp-title").textContent = i === null ? "no beat selected" : "first or last beat: not classified";
    $("#insp-probs").innerHTML = ""; $("#insp-feats").innerHTML = "";
    return;
  }
  const label = d.labels[i], probs = d.probs[k], row = d.feats.X[k], names = state.model.features;
  const f = (name) => row[names.indexOf(name)];
  let ref = "";
  if (d.match && d.match.matched[i] >= 0) ref = `, cardiologists: ${state.rec.beats[d.match.matched[i]][2]}`;
  $("#insp-title").textContent = `beat at ${(d.pt.rPeaks[i] / FS).toFixed(2)} s: predicted ${label}${ref}`;
  $("#insp-sw").style.setProperty("--c", colorOf(label));
  drawBeat($("#plot-insp"), { beat: d.feats.windows[k], template: d.feats.templates[k], fs: FS, pre: WINDOW.pre, color: colorOf(label) });
  $("#insp-probs").innerHTML = CLASSES.map((c, j) => `<div class="bar" style="--c:${colorOf(c)}"><b>${c}</b>
    <span class="track"><span class="fill" style="width:${(100 * probs[j]).toFixed(1)}%"></span></span><span>${(100 * probs[j]).toFixed(1)}%</span></div>`).join("");
  const rows = [
    ["RR before", `${(1000 * f("pre_rr")).toFixed(0)} ms`],
    ["RR after", `${(1000 * f("post_rr")).toFixed(0)} ms`],
    ["Local rhythm RR", `${(1000 * f("local_rr")).toFixed(0)} ms`],
    ["Prematurity (before / local)", `${(100 * f("pre_over_local")).toFixed(0)}%`],
    ["Pause ratio (after / before)", f("post_over_pre").toFixed(2)],
    ["QRS shape vs template (r)", f("corr_qrs").toFixed(3)],
    ["QRS width", `${(1000 * f("qrs_width")).toFixed(0)} ms`],
    ["QRS amplitude vs template", `${(100 * f("amp_ratio")).toFixed(0)}%`],
  ];
  $("#insp-feats").innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("");
}

function renderReport() {
  const { d, rec } = state;
  const m = d.metrics;
  const flagged = d.findings.filter((f) => f.severity === "warning" || f.severity === "alert");
  const v = $("#verdict");
  v.className = `verdict ${d.verdict}`;
  v.innerHTML = d.verdict === "abnormal"
    ? `ABNORMAL: ${flagged.length} finding${flagged.length === 1 ? "" : "s"} need attention<small>${rec.title} (${rec.id === "upload" ? "your file" : `MIT-BIH ${rec.id}`}), analysed with the filter chain from lesson 3</small>`
    : `NORMAL: no significant abnormality detected<small>${rec.title}, analysed with the filter chain from lesson 3</small>`;
  const sevColor = { normal: "--N", info: "--accent-2", warning: "--S", alert: "--V" };
  $("#findings").innerHTML = d.findings.map((f) => `<li><span class="sev" style="--c:var(${sevColor[f.severity]})">${f.severity}</span><b>${f.title}</b>: ${f.detail}</li>`).join("");
  const rows = [
    ["Duration analysed", `${(rec.clean.length / FS).toFixed(1)} s`],
    ["Beats detected", m.n_beats],
    ["Mean / underlying heart rate", `${fmt(m.mean_hr)} / ${fmt(m.sinus_hr)} bpm`],
    ["Min / max heart rate (5-beat average)", `${fmt(m.min_hr)} / ${fmt(m.max_hr)} bpm`],
    ["SDNN / RMSSD", `${fmt(m.sdnn_ms, 0, " ms")} / ${fmt(m.rmssd_ms, 0, " ms")}`],
    ["pNN50", fmt(m.pnn50_pct, 1, "%")],
    ["Beat classes", ["N", "S", "V", "F"].map((c) => `<b style="color:${colorOf(c)}">${c}</b> ${m.counts[c]}`).join(" &nbsp;")],
    ["RR irregularity index", fmt(m.irregularity, 3)],
    ["Signal quality (median QRS template r)", fmt(d.quality, 2)],
  ];
  if (d.agree && d.agree.n) rows.push(["Agreement with cardiologists", `${pct(d.agree.exact / d.agree.n)} (normal vs abnormal ${pct(d.agree.binary / d.agree.n)})`]);
  $("#tbl-report").innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join("");

  const r = d.pt.rPeaks, dur = rec.clean.length / FS;
  const pts = r.slice(1).map((p, i) => ({ t: p / FS, y: 1000 * (p - r[i]) / FS, shape: "dot", color: colorOf(d.labels[i + 1]) }));
  const ys = pts.map((p) => p.y);
  const lo = ys.length ? Math.min(...ys) : 0, hi = ys.length ? Math.max(...ys) : 1000;
  drawStrip($("#plot-tacho"), { t0: 0, t1: dur, fs: FS, grid: "plain", yLabel: "RR (ms)", traces: [], markers: pts,
    yRange: [Math.max(0, lo - 0.15 * (hi - lo + 100)), hi + 0.15 * (hi - lo + 100)],
    bands: [{ t0: view().t0, t1: view().t1, color: cssVar("--accent-soft") }] });
}

// ---------- static content from metrics.json ----------
function renderModelCard() {
  const M = state.metrics, sel = M.selected_model, res = M.models[sel], t = res.test_ds2;
  const nTest = t.confusion_matrix.flat().reduce((a, b) => a + b, 0);
  const nTrain = res.cv_ds1.confusion_matrix.flat().reduce((a, b) => a + b, 0);
  $("#n-features").textContent = state.model.features.length;
  $("#model-name").textContent = MODEL_NAMES[sel].toLowerCase().replace("mlp", "MLP");
  $("#class-list").innerHTML = CLASSES.map((c) => `<li style="--c:var(--${c})"><b>${c}</b>: ${CLASS_TEXT[c]}</li>`).join("");
  $("#model-summary").innerHTML = `${MODEL_NAMES[sel]} trained on <b>${nTrain.toLocaleString()}</b> beats from 22 patients (DS1)
    and tested on <b>${nTest.toLocaleString()}</b> beats from 22 <b>different</b> patients (DS2), the inter-patient protocol of de Chazal et al. (2004).`;
  const b = t.binary;
  $("#tbl-perclass").innerHTML = `<thead><tr><th>Class</th><th>Beats</th><th>Sensitivity</th><th>Precision (+P)</th><th>F1</th></tr></thead><tbody>` +
    CLASSES.map((c) => { const p = t.per_class[c]; return `<tr><td><b style="color:${colorOf(c)}">${c}</b></td><td>${p.support.toLocaleString()}</td><td>${pct(p.sensitivity)}</td><td>${pct(p.ppv)}</td><td>${p.f1.toFixed(3)}</td></tr>`; }).join("") +
    `<tr class="best"><td>Abnormal (S+V+F)</td><td>${(b.tp + b.fn).toLocaleString()}</td><td>${pct(b.sensitivity)}</td><td>${pct(b.ppv)}</td><td>${b.f1.toFixed(3)}</td></tr></tbody>`;
  const cm = t.confusion_matrix;
  $("#tbl-cm").innerHTML = `<thead><tr><th>true \\ predicted</th>${CLASSES.map((c) => `<th style="color:${colorOf(c)}">${c}</th>`).join("")}</tr></thead><tbody>` +
    cm.map((row, i) => { const s = row.reduce((a, v) => a + v, 0) || 1;
      return `<tr><th style="color:${colorOf(CLASSES[i])}">${CLASSES[i]}</th>${row.map((v, j) => `<td style="background:color-mix(in srgb, var(${i === j ? "--N" : "--V"}) ${Math.round(70 * v / s)}%, transparent)">${v.toLocaleString()}</td>`).join("")}</tr>`; }).join("") + "</tbody>";
  const names = Object.keys(M.models);
  $("#tbl-models").innerHTML = `<thead><tr><th>Algorithm</th><th>CV macro-F1<br><small>train patients</small></th><th>Accuracy<br><small>test</small></th><th>Abnormal Se<br><small>test</small></th><th>Normal Sp<br><small>test</small></th><th>V Se<br><small>test</small></th></tr></thead><tbody>` +
    names.map((n) => { const r = M.models[n], tt = r.test_ds2;
      return `<tr class="${n === sel ? "best" : ""}"><td>${MODEL_NAMES[n]}${n === sel ? " (chosen)" : ""}</td><td>${r.cv_ds1.macro_f1.toFixed(3)}</td><td>${pct(tt.accuracy)}</td><td>${pct(tt.binary.sensitivity)}</td><td>${pct(tt.binary.specificity)}</td><td>${pct(tt.per_class.V.sensitivity)}</td></tr>`; }).join("") + "</tbody>";
  const intra = M.intra_patient_demo, w232 = M.test_ds2_without_232;
  $("#leak-callout").innerHTML = `<b>The data-leakage trap.</b> Split beats at random, so the same patients appear in training and testing,
    and the very same model scores <b>${pct(intra.accuracy)}</b> accuracy. On new patients it scores <b>${pct(t.accuracy)}</b>.
    Many published "99 %" ECG classifiers made exactly this mistake. Without record 232 (see the exercise below), abnormal-beat sensitivity
    rises from ${pct(b.sensitivity)} to <b>${pct(w232.binary.sensitivity)}</b>.`;
  $("#tbl-records").innerHTML = `<thead><tr><th>Record</th><th>Beats</th><th>S</th><th>V</th><th>F</th><th>Accuracy</th><th>Abnormal Se</th><th>Normal Sp</th></tr></thead><tbody>` +
    res.test_ds2_per_record.map((r) => `<tr><td>${r.record}</td><td>${r.beats}</td><td>${r.counts.S}</td><td>${r.counts.V}</td><td>${r.counts.F}</td><td>${pct(r.accuracy)}</td><td>${pct(r.abnormal_sensitivity)}</td><td>${pct(r.normal_specificity)}</td></tr>`).join("") + "</tbody>";
  const det = M.detector;
  $("#detector-global").textContent = `On all 48 MIT-BIH records (${(det.tp + det.fn).toLocaleString()} beats) this implementation reaches sensitivity ${pct(det.sensitivity, 2)} and positive predictivity ${pct(det.ppv, 2)}, close to the original paper.`;
  const af = M.af_screening;
  $("#af-th").textContent = af.threshold_in_use.toFixed(2);
  $("#af-se").textContent = pct(af.ds2.sensitivity, 0);
  $("#af-sp").textContent = pct(af.ds2.specificity, 0);
  $("#legend-ref").innerHTML = ["N", "S", "V", "F", "Q"].map((c) => `<span><b style="--c:var(--${c})">${c}</b> ${c === "Q" ? "paced/unknown" : CLASS_TEXT[c].split(",")[0].split(" (")[0]}</span>`).join("");
  $("#legend-pred").innerHTML = CLASSES.map((c) => `<span><i class="sw dot" style="--c:var(--${c})"></i>${c}</span>`).join("") + `<span><i class="sw dot" style="--c:var(--Q)"></i>? not classified</span>`;
}

function renderCards() {
  $("#record-cards").innerHTML = state.index.map((r) => `
    <button type="button" class="rec-card" role="listitem" data-record="${r.record}" aria-pressed="false">
      <span class="num">Record ${r.record}</span>
      <span class="t">${r.title}</span>
      <span class="split ${r.split.startsWith("DS1") ? "train" : ""}">${r.split.startsWith("DS1") ? "training patient" : "unseen patient"}</span>
      <span class="counts">${["N", "S", "V", "F"].filter((c) => r.counts[c]).map((c) => `<span style="--c:var(--${c})">${c} ${r.counts[c]}</span>`).join("")}</span>
    </button>`).join("");
  $$(".rec-card").forEach((b) => b.addEventListener("click", () => openRecord(b.dataset.record)));
  $("#record-select").innerHTML = state.index.map((r) => `<option value="${r.record}">${r.record}: ${r.title}</option>`).join("");
}

function syncRecordUI() {
  const rec = state.rec;
  $$(".rec-card").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.record === rec.id)));
  const sel = $("#record-select");
  if (rec.id === "upload" && !sel.querySelector("option[value='upload']")) sel.insertAdjacentHTML("beforeend", `<option value="upload">Your file</option>`);
  sel.value = rec.id;
  const label = rec.id === "upload" ? `Your file: ${rec.title}` : `MIT-BIH ${rec.id}: ${rec.title}`;
  $("#viewbar-rec").textContent = label;
  $("#raw-title").textContent = rec.id === "upload" ? rec.title : `${rec.id} (lead ${rec.lead}, from ${rec.start_s.toFixed(0)} s)`;
  $("#record-note").textContent = rec.id === "upload" ? "Your own recording. No cardiologist labels are available."
    : `${rec.note} Annotated rhythm${rec.rhythms.length > 1 ? "s" : ""}: ${rec.rhythms.map(rhythmName).join(", ")}. ${rec.split}.`;
}

const RHYTHMS = { "(N": "normal sinus", "(AFIB": "atrial fibrillation", "(AFL": "atrial flutter", "(B": "ventricular bigeminy",
  "(T": "ventricular trigeminy", "(VT": "ventricular tachycardia", "(SVTA": "supraventricular tachyarrhythmia", "(SBR": "sinus bradycardia",
  "(BII": "2nd-degree heart block", "(NOD": "nodal rhythm", "(IVR": "idioventricular rhythm", "(PREX": "pre-excitation (WPW)",
  "(AB": "atrial bigeminy", "(VFL": "ventricular flutter", "(P": "paced", "(PR": "paced" };
const rhythmName = (r) => RHYTHMS[r] || r.replace("(", "");

function syncViewControls() {
  const dur = state.rec.clean.length / FS;
  const span = Math.min(state.view.span, dur);
  const slider = $("#view-start");
  slider.max = Math.max(0, dur - span);
  state.view.t0 = Math.min(state.view.t0, +slider.max);
  slider.value = state.view.t0;
  slider.disabled = dur <= span;
  $("#view-label").textContent = `${state.view.t0.toFixed(1)}-${(state.view.t0 + span).toFixed(1)} s`;
}

function syncNoiseControls() {
  for (const k of ["bw", "ma", "em", "pl"]) {
    $(`#n-${k}`).value = state.noise[k];
    $(`output[for='n-${k}']`).textContent = k === "pl" ? `${state.noise[k].toFixed(2)} mV` : `${state.noise[k].toFixed(2)}x`;
  }
  $$("[name='pl-hz']").forEach((el) => { el.checked = +el.value === state.noise.plHz; });
}

function syncFilterControls() {
  const f = state.filt;
  $$("[name='hp-method']").forEach((el) => { el.checked = el.value === f.method; });
  $("#f-hp").value = hpToSlider(f.highpass_hz);
  $("output[for='f-hp']").textContent = `${f.highpass_hz} Hz`;
  $("#f-hp-order").value = f.highpass_order;
  $("#f-lp-on").checked = f.lowpass_on;
  $("#f-lp").value = f.lowpass_hz;
  $("output[for='f-lp']").textContent = `${f.lowpass_hz} Hz`;
  $("#f-lp-order").value = f.lowpass_order;
  $("#f-notch-on").checked = f.notch_on;
  $("#f-notch-hz").value = f.notch_hz;
  $("#f-notch-q").value = f.notch_q;
  $("output[for='f-notch-q']").textContent = `${f.notch_q} (${(f.notch_hz / f.notch_q).toFixed(1)} Hz wide)`;
  $$("[name='phase']").forEach((el) => { el.checked = (el.value === "zero") === f.zero_phase; });
  const match = Object.entries(FILTER_PRESETS).find(([, p]) => Object.keys(p).every((k) => p[k] === f[k]));
  $$("#filter-presets button").forEach((b) => b.classList.toggle("on", !!match && b.dataset.filter === match[0]));
}

// ---------- interaction ----------
let computeTimer = null, pendingStage = Infinity;
function scheduleCompute(stage) {
  pendingStage = Math.min(pendingStage, stage);
  if (computeTimer) return;
  computeTimer = nextFrame(() => {
    const s = pendingStage;
    computeTimer = null; pendingStage = Infinity;
    try { compute(s); } catch (err) { showError(err); }
  });
}

async function openRecord(id) {
  if (id === "upload") return;
  const p = await loadJSON(`data/records/${id}.json`);
  setRecord({ id, title: p.title, note: p.note, split: p.split, lead: p.lead, start_s: p.start_s, rhythms: p.rhythms,
    clean: Float64Array.from(p.signal_uv, (u) => u / 1000), beats: p.beats, uploaded: false });
}

function bindControls() {
  $("#record-select").addEventListener("change", (e) => openRecord(e.target.value));
  $("#view-start").addEventListener("input", (e) => { state.view.t0 = +e.target.value; syncViewControls(); render(); });
  $("#view-span").addEventListener("change", (e) => { state.view.span = +e.target.value; syncViewControls(); render(); });

  for (const k of ["bw", "ma", "em", "pl"]) $(`#n-${k}`).addEventListener("input", (e) => {
    state.noise[k] = +e.target.value; syncNoiseControls(); scheduleCompute(STAGE.NOISE);
  });
  $$("[name='pl-hz']").forEach((el) => el.addEventListener("change", () => { state.noise.plHz = +el.value; scheduleCompute(STAGE.NOISE); }));
  $$("#noise-presets button").forEach((b) => b.addEventListener("click", () => {
    Object.assign(state.noise, NOISE_PRESETS[b.dataset.noise]); syncNoiseControls(); scheduleCompute(STAGE.NOISE);
  }));
  $("#show-clean-noise").addEventListener("change", render);
  $("#show-noisy").addEventListener("change", render);
  $("#show-ref").addEventListener("change", render);

  const onFilter = (fn) => (e) => { fn(e); syncFilterControls(); scheduleCompute(STAGE.FILTER); };
  $$("[name='hp-method']").forEach((el) => el.addEventListener("change", onFilter(() => { state.filt.method = el.value; })));
  $("#f-hp").addEventListener("input", onFilter((e) => { state.filt.highpass_hz = sliderToHp(+e.target.value); }));
  $("#f-hp-order").addEventListener("change", onFilter((e) => { state.filt.highpass_order = +e.target.value; }));
  $("#f-lp-on").addEventListener("change", onFilter((e) => { state.filt.lowpass_on = e.target.checked; }));
  $("#f-lp").addEventListener("input", onFilter((e) => { state.filt.lowpass_hz = +e.target.value; }));
  $("#f-lp-order").addEventListener("change", onFilter((e) => { state.filt.lowpass_order = +e.target.value; }));
  $("#f-notch-on").addEventListener("change", onFilter((e) => { state.filt.notch_on = e.target.checked; }));
  $("#f-notch-hz").addEventListener("change", onFilter((e) => { state.filt.notch_hz = +e.target.value; }));
  $("#f-notch-q").addEventListener("input", onFilter((e) => { state.filt.notch_q = +e.target.value; }));
  $$("[name='phase']").forEach((el) => el.addEventListener("change", onFilter(() => { state.filt.zero_phase = el.value === "zero"; })));
  $$("#filter-presets button").forEach((b) => b.addEventListener("click", onFilter(() => { state.filt = { ...FILTER_PRESETS[b.dataset.filter] }; })));

  const recipes = {
    st: () => { state.filt = { ...FILTER_PRESETS.monitor, highpass_hz: 2 }; syncFilterControls(); $("#show-ref").checked = true; scheduleCompute(STAGE.FILTER); },
    harsh: () => { state.filt = { ...FILTER_PRESETS.harsh }; syncFilterControls(); scheduleCompute(STAGE.FILTER); },
    motion: () => { Object.assign(state.noise, { em: 1.5 }); syncNoiseControls(); scheduleCompute(STAGE.NOISE); },
    r232: () => openRecord("232").then(() => $("#classify").scrollIntoView()),
  };
  $$("[data-recipe]").forEach((b) => b.addEventListener("click", () => recipes[b.dataset.recipe]()));

  $("#plot-class").addEventListener("click", (e) => {
    const plot = state.classPlot;
    if (!plot) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const t = plot.toTime(e.clientX - rect.left);
    const r = state.d.pt.rPeaks;
    let best = null, bestD = 0.4;
    r.forEach((p, i) => { const dist = Math.abs(p / FS - t); if (dist < bestD) { best = i; bestD = dist; } });
    if (best !== null) { state.selected = best; render(); }
  });

  $("#btn-print").addEventListener("click", () => window.print());
  $("#up-go").addEventListener("click", analyseUpload);
  $("#theme-toggle").addEventListener("click", toggleTheme);
  let resizeTimer;
  new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 80); }).observe(document.body);
  watchSections();
}

async function analyseUpload() {
  const status = $("#up-status");
  const file = $("#up-file").files[0];
  if (!file) { status.textContent = "Choose a file first."; return; }
  try {
    const fs = +$("#up-fs").value;
    if (!(fs >= 50)) throw new Error("Enter the sampling rate in Hz (at least 50).");
    const scale = +$("#up-units").value;
    const rows = (await file.text()).split(/\r?\n/).map((l) => l.trim()).filter(Boolean).map((l) => l.split(/[,;\t]|\s+/));
    let col = $("#up-col").value.trim();
    let start = 0;
    if (Number.isNaN(parseFloat(rows[0][/^\d+$/.test(col) ? +col : 0]))) {
      start = 1;
      if (!/^\d+$/.test(col)) col = rows[0].findIndex((h) => h.replace(/"/g, "") === col);
    }
    col = +col;
    if (!(col >= 0)) throw new Error("Column not found in the header row.");
    let x = Float64Array.from(rows.slice(start), (r) => parseFloat(r[col]) * scale).filter((v) => Number.isFinite(v));
    if (x.length < fs * 5) throw new Error("Need at least 5 seconds of signal.");
    const maxS = 300;
    let note = "";
    if (x.length > maxS * fs) { x = x.slice(0, Math.round(maxS * fs)); note = ` Only the first ${maxS / 60} minutes are analysed.`; }
    if (fs !== FS) {
      if (fs > FS) x = sosfiltfilt(butterSos("lowpass", 150, fs, 4), x); // anti-alias before resampling
      x = resampleLinear(x, fs, FS);
    }
    setRecord({ id: "upload", title: file.name, note: "", split: "uploaded", lead: "?", start_s: 0, rhythms: [],
      clean: x, beats: [], uploaded: true });
    status.textContent = `Loaded ${(x.length / FS).toFixed(1)} s from ${file.name}${fs !== FS ? ` (resampled ${fs} -> ${FS} Hz)` : ""}.${note} Every lesson now uses your file.`;
  } catch (err) {
    status.textContent = `Could not read the file: ${err.message}`;
  }
}

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("ecg-theme", root.dataset.theme); } catch { /* storage unavailable */ }
  render();
}

function watchSections() {
  const links = new Map($$(".steps-nav a").map((a) => [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) {
      links.forEach((a) => a.classList.remove("active"));
      links.get(e.target.id)?.classList.add("active");
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  links.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
}

function showError(err) {
  console.error(err);
  const el = $("#loading");
  el.hidden = false;
  el.classList.add("error");
  el.textContent = `Something went wrong: ${err.message}`;
}

async function init() {
  try { const t = localStorage.getItem("ecg-theme"); if (t) document.documentElement.dataset.theme = t; } catch { /* ignore */ }
  try {
    const [index, model, metrics, noise] = await Promise.all(
      ["data/records.json", "data/model.json", "data/metrics.json", "data/noise.json"].map(loadJSON));
    Object.assign(state, { index, model, metrics });
    for (const [k, v] of Object.entries(noise)) state.noiseRec[k] = Float64Array.from(v.signal_uv, (u) => u / 1000);
    renderCards();
    renderModelCard();
    bindControls();
    syncNoiseControls();
    syncFilterControls();
    await openRecord(index[0].record);
    $("#loading").hidden = true;
  } catch (err) {
    showError(err);
  }
}

init();
