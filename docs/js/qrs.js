// Pan-Tompkins QRS detection, mirroring ecglab/detect.py line for line.
import { butterSos, sosfiltfilt } from "./dsp.js";

export const PT = {
  band: [5, 15],
  integrationS: 0.15,
  refractoryS: 0.2,
  tWaveS: 0.36,
  searchbackRR: 1.66,
  refineS: 0.08,
  learningS: 2.0,
  matchToleranceS: 0.15,
};

function localMaxima(x, minDistance) {
  const kept = [];
  for (let i = 1; i < x.length - 1; i++) {
    if (!(x[i] > x[i - 1] && x[i] >= x[i + 1])) continue;
    if (kept.length && i - kept[kept.length - 1] < minDistance) {
      if (x[i] > x[kept[kept.length - 1]]) kept[kept.length - 1] = i;
    } else kept.push(i);
  }
  return kept;
}

export function refinePeaks(ecg, approx, fs, halfWindowS = PT.refineS) {
  const hw = Math.round(halfWindowS * fs), n = ecg.length;
  return approx.map((p) => {
    const lo = Math.max(0, p - hw), hi = Math.min(n, p + hw + 1);
    let best = lo;
    for (let i = lo + 1; i < hi; i++) if (Math.abs(ecg[i]) > Math.abs(ecg[best])) best = i;
    return best;
  });
}

const meanOf = (a) => a.reduce((s, v) => s + v, 0) / a.length;

/** Returns every intermediate stage so the website can plot them. */
export function panTompkins(ecg, fs) {
  const n = ecg.length;
  const sos = [...butterSos("highpass", PT.band[0], fs, 2), ...butterSos("lowpass", PT.band[1], fs, 2)];
  const bp = sosfiltfilt(sos, ecg);

  const d = new Float64Array(n);
  for (let i = 2; i < n - 2; i++) d[i] = ((2 * bp[i + 1] + bp[i + 2] - bp[i - 2] - 2 * bp[i - 1]) * fs) / 8;
  const sq = d.map((v) => v * v);

  const w = Math.round(PT.integrationS * fs);
  const csum = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) csum[i + 1] = csum[i] + sq[i];
  const mwi = new Float64Array(n);
  const half = Math.floor(w / 2);
  for (let i = 0; i < n; i++) {
    const a = Math.min(n, Math.max(0, i - half)), b = Math.min(n, Math.max(0, i - half + w));
    mwi[i] = (csum[b] - csum[a]) / w;
  }

  const refractory = Math.round(PT.refractoryS * fs);
  const tWave = Math.round(PT.tWaveS * fs);
  const cands = localMaxima(mwi, refractory);

  const learnN = Math.max(1, Math.floor(PT.learningS * fs));
  let mx = -Infinity, sum = 0;
  for (let i = 0; i < Math.min(learnN, n); i++) { mx = Math.max(mx, mwi[i]); sum += mwi[i]; }
  let spki = mx / 3, npki = sum / Math.min(learnN, n) / 2;
  let th1 = npki + 0.25 * (spki - npki);

  const slopeAt = (p) => {
    let m = 0;
    for (let i = Math.max(0, p - half); i < Math.min(n, p + half + 1); i++) m = Math.max(m, Math.abs(d[i]));
    return m;
  };
  const qrs = [], slopes = [], rr = [];
  const thresholds = new Float64Array(cands.length);
  const isTWave = (p, slope) => qrs.length > 0 && p - qrs[qrs.length - 1] < tWave && slope < 0.5 * slopes[slopes.length - 1];

  for (let k = 0; k < cands.length; k++) {
    const p = cands[k];
    if (rr.length && p - qrs[qrs.length - 1] > PT.searchbackRR * meanOf(rr.slice(-8))) {
      let best = -1, bestVal = 0.5 * th1;
      for (let j = k - 1; j >= 0; j--) {
        const q = cands[j];
        if (q - qrs[qrs.length - 1] <= refractory) break;
        if (mwi[q] > bestVal && !isTWave(q, slopeAt(q))) { best = q; bestVal = mwi[q]; }
      }
      if (best >= 0) {
        rr.push(best - qrs[qrs.length - 1]);
        qrs.push(best);
        slopes.push(slopeAt(best));
        spki = 0.25 * bestVal + 0.75 * spki;
        th1 = npki + 0.25 * (spki - npki);
      }
    }
    const val = mwi[p];
    thresholds[k] = th1;
    if (val > th1) {
      const slope = slopeAt(p);
      if (isTWave(p, slope)) npki = 0.125 * val + 0.875 * npki;
      else {
        if (qrs.length) rr.push(p - qrs[qrs.length - 1]);
        qrs.push(p);
        slopes.push(slope);
        spki = 0.125 * val + 0.875 * spki;
      }
    } else npki = 0.125 * val + 0.875 * npki;
    th1 = npki + 0.25 * (spki - npki);
  }

  const r = refinePeaks(ecg, qrs, fs);
  const keep = [];
  for (const i of r) {
    const last = keep[keep.length - 1];
    if (keep.length && i - last < refractory) {
      if (Math.abs(ecg[i]) > Math.abs(ecg[last])) keep[keep.length - 1] = i;
    } else if (!keep.length || i !== last) keep.push(i);
  }
  return { rPeaks: keep, bandpassed: bp, derivative: d, squared: sq, integrated: mwi, qrsMwi: qrs, thresholds, candidates: cands };
}

/** One-to-one matching of detections to reference beats within +/- tolerance (AAMI EC57). */
export function matchBeats(reference, detected, fs, toleranceS = PT.matchToleranceS) {
  const tol = Math.round(toleranceS * fs);
  const used = new Uint8Array(reference.length);
  const matched = new Int32Array(detected.length).fill(-1);
  let j = 0;
  detected.forEach((dv, i) => {
    while (j < reference.length && reference[j] < dv - tol) j++;
    let best = -1, bestDist = tol + 1;
    for (let k = j; k < reference.length && reference[k] <= dv + tol; k++) {
      const dist = Math.abs(reference[k] - dv);
      if (!used[k] && dist < bestDist) { best = k; bestDist = dist; }
    }
    if (best >= 0) { used[best] = 1; matched[i] = best; }
  });
  const tp = used.reduce((s, v) => s + v, 0);
  return { tp, fp: detected.length - tp, fn: reference.length - tp, matched,
    sensitivity: tp / Math.max(reference.length, 1), ppv: tp / Math.max(detected.length, 1) };
}
