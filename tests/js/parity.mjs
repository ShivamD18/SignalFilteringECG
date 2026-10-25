// Runs the browser pipeline under Node on the parity fixture and prints the results as JSON.
// Usage: node tests/js/parity.mjs <repo root> <fixture.json>
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [, , root, fixturePath] = process.argv;
const load = (p) => import(pathToFileURL(`${root}/docs/js/${p}`).href);
const [dsp, qrs, feat, mdl, rep] = await Promise.all(["dsp.js", "qrs.js", "features.js", "model.js", "report.js"].map(load));
const readJSON = (p) => JSON.parse(readFileSync(p, "utf8"));

const fx = readJSON(fixturePath);
const rec = readJSON(`${root}/docs/data/records/${fx.record}.json`);
const model = readJSON(`${root}/docs/data/model.json`);

const x = Float64Array.from(rec.signal_uv, (u) => u / 1000);
const ecg = dsp.applyChain(x, model.filter, fx.fs);
const pt = qrs.panTompkins(ecg, fx.fs);
const f = feat.extractFeatures(ecg, pt.rPeaks);
const probs = f.X.map((row) => mdl.predictProba(model, row));
const labels = pt.rPeaks.map(() => "?");
f.beatIdx.forEach((i, k) => { labels[i] = model.classes[probs[k].indexOf(Math.max(...probs[k]))]; });
const qi = model.features.indexOf("corr_qrs");
const q = f.X.map((r) => r[qi]).sort((a, b) => a - b);
const quality = q.length % 2 ? q[q.length >> 1] : (q[q.length / 2 - 1] + q[q.length / 2]) / 2;
const metrics = rep.rhythmMetrics(pt.rPeaks, labels, fx.fs);
const found = rep.findings(metrics, quality);

const every10 = (a) => Array.from(a).filter((_, i) => i % 10 === 0);
const out = {
  filtered_every_10: every10(ecg),
  integrated_every_10: every10(pt.integrated),
  r_peaks: pt.rPeaks,
  beat_index: f.beatIdx,
  features: f.X,
  probabilities: probs,
  labels,
  signal_quality: quality,
  metrics,
  findings: found,
  verdict: rep.verdict(found),
  designs: fx.designs.map((d) => {
    const sos = dsp.butterSos(d.kind, d.cutoff, d.fs, d.order);
    return { n_sections: sos.length, filtfilt: Array.from(dsp.sosfiltfilt(sos, Float64Array.from(fx.probe))) };
  }),
  notch_60: dsp.notchSos(60, fx.fs, 30)[0],
  feature_count: feat.FEATURE_COUNT,
};
process.stdout.write(JSON.stringify(out));
