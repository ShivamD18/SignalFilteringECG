// Rhythm metrics and plain-language findings, mirroring ecglab/rhythm.py.

export const RULES = {
  bradyBpm: 60,
  tachyBpm: 100,
  pauseS: 2.0,
  longPauseS: 3.0,
  irregularityLags: [1, 2, 3],
  minIrregularityPairs: 8,
  irregularityThreshold: 0.09,
};
const VENTRICULAR = ["V", "F"];

export function runsOf(labels, target) {
  const out = [];
  let start = null;
  [...labels, ""].forEach((lab, i) => {
    if (lab === target && start === null) start = i;
    else if (lab !== target && start !== null) { out.push([start, i - start]); start = null; }
  });
  return out;
}

export function patternEpisodes(labels, pattern, minCycles = 3) {
  const k = pattern.length, out = [];
  const matchAt = (j) => pattern.every((p, q) => labels[j + q] === p);
  let i = 0;
  while (i <= labels.length - k) {
    let j = i, cycles = 0;
    while (j + k <= labels.length && matchAt(j)) { cycles++; j += k; }
    if (cycles >= minCycles) { out.push([i, cycles]); i = j; } else i++;
  }
  return out;
}

const median = (a) => {
  const s = Float64Array.from(a).sort(), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

/** Median |RR[i] - RR[i-k]| / median RR, minimised over lags 1-3, ignoring intervals touching V/F beats. */
export function irregularityIndex(rr, labels) {
  const ok = rr.map((_, j) => !VENTRICULAR.includes(labels[j]) && !VENTRICULAR.includes(labels[j + 1]));
  if (ok.filter(Boolean).length < 2) return null;
  const diffs = [];
  for (const k of RULES.irregularityLags) {
    const d = [];
    for (let j = k; j < rr.length; j++) if (ok[j] && ok[j - k]) d.push(Math.abs(rr[j] - rr[j - k]));
    if (d.length < RULES.minIrregularityPairs) return null;
    diffs.push(median(d));
  }
  return Math.min(...diffs) / median(rr.filter((_, j) => ok[j]));
}

export function rhythmMetrics(r, labels, fs) {
  const n = r.length;
  const counts = Object.fromEntries(["N", "S", "V", "F", "?"].map((c) => [c, labels.filter((l) => l === c).length]));
  const classified = Math.max(n - counts["?"], 1);
  const m = { n_beats: n, counts, burden_pct: Object.fromEntries(["S", "V", "F"].map((c) => [c, (100 * counts[c]) / classified])) };
  if (n < 3) return m;

  const rr = [];
  for (let i = 1; i < n; i++) rr.push((r[i] - r[i - 1]) / fs);
  m.mean_hr = 60 / mean(rr);
  const nnMask = rr.map((_, j) => labels[j] === "N" && labels[j + 1] === "N");
  const nn = rr.filter((_, j) => nnMask[j]);
  m.sinus_hr = nn.length >= 2 ? 60 / mean(nn) : m.mean_hr;
  const k = Math.min(5, rr.length);
  const smooth = [];
  for (let i = 0; i + k <= rr.length; i++) smooth.push(mean(rr.slice(i, i + k)));
  m.min_hr = 60 / Math.max(...smooth);
  m.max_hr = 60 / Math.min(...smooth);

  const d = [];
  for (let j = 0; j < rr.length - 1; j++) if (nnMask[j] && nnMask[j + 1]) d.push(rr[j + 1] - rr[j]);
  const nnMean = nn.length ? mean(nn) : 0;
  m.sdnn_ms = nn.length >= 2 ? 1000 * Math.sqrt(mean(nn.map((v) => (v - nnMean) ** 2))) : null;
  m.rmssd_ms = d.length ? 1000 * Math.sqrt(mean(d.map((v) => v * v))) : null;
  m.pnn50_pct = d.length ? (100 * d.filter((v) => Math.abs(v) > 0.05).length) / d.length : null;

  m.pauses = rr.flatMap((v, j) => (v > RULES.pauseS ? [{ time_s: r[j] / fs, duration_s: v }] : []));
  m.longest_rr_s = Math.max(...rr);
  m.irregularity = irregularityIndex(rr, labels);

  const runInfo = (cls, minLen, maxLen = Infinity) =>
    runsOf(labels, cls)
      .filter(([, len]) => len >= minLen && len <= maxLen)
      .map(([s, len]) => {
        const span = (r[s + len - 1] - r[s]) / fs;
        return { start_s: r[s] / fs, beats: len, rate_bpm: span > 0 ? (60 * (len - 1)) / span : null };
      });
  m.v_couplets = runInfo("V", 2, 2);
  m.v_runs = runInfo("V", 3);
  m.s_runs = runInfo("S", 3);
  m.bigeminy = patternEpisodes(labels, ["N", "V"]).map(([s, c]) => ({ start_s: r[s] / fs, cycles: c }));
  m.trigeminy = patternEpisodes(labels, ["N", "N", "V"]).map(([s, c]) => ({ start_s: r[s] / fs, cycles: c }));
  return m;
}

export function findings(m, signalQuality = null) {
  const out = [];
  const add = (severity, title, detail) => out.push({ severity, title, detail });
  if (m.n_beats < 3 || m.mean_hr === undefined) {
    add("alert", "Not enough beats", "Fewer than three heartbeats were detected; check the recording and filters.");
    return out;
  }
  if (signalQuality !== null && signalQuality < 0.8)
    add("warning", "Poor signal quality", `Beats correlate only ${signalQuality.toFixed(2)} (median) with their template; noise may cause errors.`);

  const hr = m.sinus_hr;
  if (hr < RULES.bradyBpm) add("warning", "Bradycardia", `Underlying rate ${hr.toFixed(0)} bpm is below ${RULES.bradyBpm} bpm.`);
  else if (hr > RULES.tachyBpm) add("warning", "Tachycardia", `Underlying rate ${hr.toFixed(0)} bpm is above ${RULES.tachyBpm} bpm.`);
  else add("normal", "Heart rate in normal range", `Underlying rate ${hr.toFixed(0)} bpm (60-100 bpm).`);

  const names = { V: "Ventricular ectopic beats (PVCs)", S: "Supraventricular ectopic beats (PACs)", F: "Fusion beats" };
  for (const c of ["V", "S", "F"]) {
    const cnt = m.counts[c], pct = m.burden_pct[c];
    if (!cnt) continue;
    const sev = pct >= 10 ? "alert" : pct >= 1 || cnt >= 3 ? "warning" : "info";
    add(sev, names[c], `${cnt} beat${cnt !== 1 ? "s" : ""} (${pct.toFixed(1)}% of classified beats).`);
  }
  const longestBy = (arr, key) => arr.reduce((a, b) => (b[key] > a[key] ? b : a));
  if (m.v_runs.length) {
    const l = longestBy(m.v_runs, "beats");
    const rate = l.rate_bpm ? ` at ~${l.rate_bpm.toFixed(0)} bpm` : "";
    add("alert", "Run of ventricular beats",
      `${m.v_runs.length} run(s) of 3+ consecutive PVCs; longest ${l.beats} beats${rate}. ` +
        "Three or more at >100 bpm is called non-sustained ventricular tachycardia.");
  }
  if (m.v_couplets.length) add("warning", "Ventricular couplets", `${m.v_couplets.length} pair(s) of back-to-back PVCs.`);
  if (m.bigeminy.length)
    add("warning", "Ventricular bigeminy", `Every other beat is a PVC for ${Math.max(...m.bigeminy.map((e) => e.cycles))}+ cycles.`);
  if (m.trigeminy.length)
    add("warning", "Ventricular trigeminy", `Every third beat is a PVC for ${Math.max(...m.trigeminy.map((e) => e.cycles))}+ cycles.`);
  if (m.s_runs.length) {
    const l = longestBy(m.s_runs, "beats");
    add("warning", "Run of supraventricular beats", `${m.s_runs.length} run(s) of 3+ consecutive PACs; longest ${l.beats} beats.`);
  }
  if (m.pauses.length) {
    const longest = Math.max(...m.pauses.map((p) => p.duration_s));
    add(longest >= RULES.longPauseS ? "alert" : "warning", "Pause",
      `${m.pauses.length} gap(s) longer than ${RULES.pauseS.toFixed(0)} s between beats; longest ${longest.toFixed(1)} s.`);
  }
  if (m.irregularity !== null && m.irregularity > RULES.irregularityThreshold)
    add("warning", "Irregularly irregular rhythm",
      `RR irregularity index ${m.irregularity.toFixed(2)} exceeds ${RULES.irregularityThreshold.toFixed(2)}: a pattern seen in ` +
        "atrial fibrillation. Confirm by looking for absent P waves.");
  return out;
}

export const verdict = (found) => (found.some((f) => f.severity === "warning" || f.severity === "alert") ? "abnormal" : "normal");
