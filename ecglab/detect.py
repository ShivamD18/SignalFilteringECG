"""QRS (heartbeat) detection with the Pan-Tompkins algorithm.

Pan J, Tompkins WJ. "A real-time QRS detection algorithm."
IEEE Trans Biomed Eng 32(3):230-236, 1985.

Stages
    1. Band-pass 5-15 Hz      keeps the QRS energy, rejects P/T waves and noise
    2. Derivative             emphasises the steep QRS slopes
    3. Squaring               makes everything positive, boosts large slopes
    4. Moving integration     150 ms window merges each QRS into one hump
    5. Adaptive thresholds    signal/noise peak estimates track the amplitude,
                              with search-back for missed beats and T-wave
                              rejection by slope

This offline version uses zero-phase filtering and a centred integration
window, so no group-delay bookkeeping is needed. The same steps are mirrored
line-for-line in ``docs/js/qrs.js``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import apply_sos, butter_sos

BAND_HZ = (5.0, 15.0)
INTEGRATION_S = 0.150
REFRACTORY_S = 0.200
T_WAVE_S = 0.360
SEARCHBACK_RR = 1.66
REFINE_S = 0.080
LEARNING_S = 2.0
MATCH_TOLERANCE_S = 0.150  # AAMI EC57 beat-matching window


@dataclass
class PanTompkinsResult:
    r_peaks: np.ndarray          # refined R-peak sample indices in the input ECG
    bandpassed: np.ndarray
    derivative: np.ndarray
    squared: np.ndarray
    integrated: np.ndarray
    qrs_mwi: np.ndarray          # accepted peaks in the integrated signal
    threshold: np.ndarray        # THRESHOLD_I1 at each accepted/rejected candidate
    candidates: np.ndarray


def _local_maxima(x: np.ndarray, min_distance: int) -> np.ndarray:
    """Local maxima, keeping only the larger of any two closer than ``min_distance``."""
    idx = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
    kept: list[int] = []
    for i in idx:
        if kept and i - kept[-1] < min_distance:
            if x[i] > x[kept[-1]]:
                kept[-1] = int(i)
        else:
            kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def refine_peaks(ecg: np.ndarray, approx: np.ndarray, fs: float, half_window_s: float = REFINE_S) -> np.ndarray:
    """Move each approximate beat location to the largest absolute deflection nearby.

    The same rule is applied to cardiologist annotations during training and to
    detector output at inference, so beats are always aligned identically.
    """
    hw = int(round(half_window_s * fs))
    n = len(ecg)
    refined = []
    for p in approx:
        lo, hi = max(0, int(p) - hw), min(n, int(p) + hw + 1)
        refined.append(lo + int(np.argmax(np.abs(ecg[lo:hi]))))
    return np.asarray(refined, dtype=np.int64)


def pan_tompkins(ecg: np.ndarray, fs: float) -> PanTompkinsResult:
    """Detect R peaks in a (preferably pre-filtered) single-lead ECG."""
    ecg = np.asarray(ecg, dtype=np.float64)
    n = len(ecg)

    sos = np.vstack([butter_sos("highpass", BAND_HZ[0], fs, 2), butter_sos("lowpass", BAND_HZ[1], fs, 2)])
    bp = apply_sos(ecg, sos, zero_phase=True)

    # Five-point centred derivative: (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2]) / 8T
    d = np.zeros(n)
    d[2:-2] = (2 * bp[3:-1] + bp[4:] - bp[:-4] - 2 * bp[1:-3]) * fs / 8.0
    sq = d * d

    w = int(round(INTEGRATION_S * fs))
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    start = np.clip(np.arange(n) - w // 2, 0, n)
    stop = np.clip(np.arange(n) - w // 2 + w, 0, n)
    mwi = (csum[stop] - csum[start]) / w

    refractory = int(round(REFRACTORY_S * fs))
    t_wave = int(round(T_WAVE_S * fs))
    slope_hw = w // 2
    cands = _local_maxima(mwi, refractory)

    learn = mwi[: max(1, int(LEARNING_S * fs))]
    spki = float(np.max(learn)) / 3.0
    npki = float(np.mean(learn)) / 2.0
    th1 = npki + 0.25 * (spki - npki)

    def slope_at(p: int) -> float:
        return float(np.max(np.abs(d[max(0, p - slope_hw): p + slope_hw + 1])))

    qrs: list[int] = []
    slopes: list[float] = []
    rr: list[int] = []
    thresholds = np.zeros(len(cands))

    def is_t_wave(p: int, slope: float) -> bool:
        return bool(qrs) and p - qrs[-1] < t_wave and slope < 0.5 * slopes[-1]

    for k, p in enumerate(cands):
        p = int(p)
        # Search-back: if no beat for 166 % of the average RR, take the
        # biggest rejected candidate above the lower threshold.
        if rr and p - qrs[-1] > SEARCHBACK_RR * np.mean(rr[-8:]):
            best, best_val = -1, 0.5 * th1
            for j in range(k - 1, -1, -1):
                q = int(cands[j])
                if q - qrs[-1] <= refractory:
                    break
                if mwi[q] > best_val and not is_t_wave(q, slope_at(q)):
                    best, best_val = q, float(mwi[q])
            if best >= 0:
                rr.append(best - qrs[-1])
                qrs.append(best)
                slopes.append(slope_at(best))
                spki = 0.25 * best_val + 0.75 * spki
                th1 = npki + 0.25 * (spki - npki)

        val = float(mwi[p])
        thresholds[k] = th1
        if val > th1:
            slope = slope_at(p)
            if is_t_wave(p, slope):
                npki = 0.125 * val + 0.875 * npki
            else:
                if qrs:
                    rr.append(p - qrs[-1])
                qrs.append(p)
                slopes.append(slope)
                spki = 0.125 * val + 0.875 * spki
        else:
            npki = 0.125 * val + 0.875 * npki
        th1 = npki + 0.25 * (spki - npki)

    qrs_arr = np.asarray(qrs, dtype=np.int64)
    r = refine_peaks(ecg, qrs_arr, fs)
    # Two integration peaks can refine onto the same complex; keep the larger.
    keep: list[int] = []
    for i in r:
        if keep and i - keep[-1] < refractory:
            if abs(ecg[i]) > abs(ecg[keep[-1]]):
                keep[-1] = int(i)
        elif not keep or i != keep[-1]:
            keep.append(int(i))

    return PanTompkinsResult(
        r_peaks=np.asarray(keep, dtype=np.int64),
        bandpassed=bp,
        derivative=d,
        squared=sq,
        integrated=mwi,
        qrs_mwi=qrs_arr,
        threshold=thresholds,
        candidates=cands,
    )


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    return pan_tompkins(ecg, fs).r_peaks


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matched_ref: np.ndarray   # index into reference for each matched detection (-1 = FP)

    @property
    def sensitivity(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def ppv(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)


def match_beats(reference: np.ndarray, detected: np.ndarray, fs: float,
                tolerance_s: float = MATCH_TOLERANCE_S) -> MatchResult:
    """One-to-one matching of detections to reference beats within +/- tolerance."""
    tol = int(round(tolerance_s * fs))
    ref = np.asarray(reference)
    det = np.asarray(detected)
    matched = np.full(len(det), -1, dtype=np.int64)
    used = np.zeros(len(ref), dtype=bool)
    j = 0
    for i, d in enumerate(det):
        while j < len(ref) and ref[j] < d - tol:
            j += 1
        best, best_dist = -1, tol + 1
        k = j
        while k < len(ref) and ref[k] <= d + tol:
            if not used[k] and abs(int(ref[k]) - int(d)) < best_dist:
                best, best_dist = k, abs(int(ref[k]) - int(d))
            k += 1
        if best >= 0:
            used[best] = True
            matched[i] = best
    tp = int(used.sum())
    return MatchResult(tp=tp, fp=len(det) - tp, fn=len(ref) - tp, matched_ref=matched)
