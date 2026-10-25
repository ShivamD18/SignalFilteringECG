"""Rhythm-level analysis: heart rate, HRV, ectopy patterns, pauses, irregularity.

Beat labels are single letters: N, S, V, F, or ``?`` for beats that could not
be classified (the first and last beat of a recording). The rules below are
mirrored in ``docs/js/report.js``.

Atrial fibrillation is a *rhythm*, not a beat type: in MIT-BIH the beats
during AF are labelled N. It is screened for here with an RR irregularity
index whose threshold is calibrated on DS1 and checked on DS2 with
``calibrate_irregularity``.
"""

from __future__ import annotations

import numpy as np

BRADY_BPM = 60.0
TACHY_BPM = 100.0
PAUSE_S = 2.0
LONG_PAUSE_S = 3.0
VENTRICULAR = ("V", "F")
IRREGULARITY_LAGS = (1, 2, 3)
MIN_IRREGULARITY_PAIRS = 8
# Calibrated end-to-end (detector + classifier) on DS1 30 s windows: Se 0.97 / Sp 0.99.
# On the unseen DS2 patients: Se 0.72 / Sp 0.97. Reproduce with `python -m ecglab calibrate-af`.
IRREGULARITY_THRESHOLD = 0.09


def runs_of(labels: list[str], target: str) -> list[tuple[int, int]]:
    """(start, length) of every maximal run of consecutive ``target`` labels."""
    out, start = [], None
    for i, lab in enumerate(list(labels) + [""]):
        if lab == target and start is None:
            start = i
        elif lab != target and start is not None:
            out.append((start, i - start))
            start = None
    return out


def pattern_episodes(labels: list[str], pattern: list[str], min_cycles: int = 3) -> list[tuple[int, int]]:
    """(start, cycles) of stretches where ``pattern`` repeats at least ``min_cycles`` times."""
    k, i, out = len(pattern), 0, []
    labels = list(labels)
    while i <= len(labels) - k:
        j, cycles = i, 0
        while j + k <= len(labels) and labels[j:j + k] == pattern:
            cycles += 1
            j += k
        if cycles >= min_cycles:
            out.append((i, cycles))
            i = j
        else:
            i += 1
    return out


def irregularity_index(rr: np.ndarray, labels: list[str]) -> float | None:
    """Median |RR[i] - RR[i-k]| / median RR, minimised over lags k = 1, 2, 3.

    Intervals touching a ventricular (V/F) beat are ignored. Taking the minimum
    over lags keeps the index small for *regularly* irregular patterns such as
    bigeminy (period 2) or trigeminy (period 3), even when the classifier
    mislabels some of the ectopic beats, whereas atrial fibrillation is
    irregular at every lag.
    """
    rr = np.asarray(rr, dtype=np.float64)
    ok = np.array([labels[j] not in VENTRICULAR and labels[j + 1] not in VENTRICULAR for j in range(len(rr))],
                  dtype=bool)
    if ok.sum() < 2:
        return None
    diffs = []
    for k in IRREGULARITY_LAGS:
        pair = ok[k:] & ok[:-k]
        if pair.sum() < MIN_IRREGULARITY_PAIRS:
            return None
        diffs.append(np.median(np.abs(rr[k:] - rr[:-k])[pair]))
    return float(min(diffs) / np.median(rr[ok]))


def rhythm_metrics(r_peaks: np.ndarray, labels: list[str], fs: float) -> dict:
    r = np.asarray(r_peaks, dtype=np.float64)
    labels = list(labels)
    n = len(r)
    m: dict = {"n_beats": n, "counts": {c: labels.count(c) for c in ("N", "S", "V", "F", "?")}}
    classified = max(n - m["counts"]["?"], 1)
    m["burden_pct"] = {c: 100.0 * m["counts"][c] / classified for c in ("S", "V", "F")}
    if n < 3:
        return m

    rr = np.diff(r) / fs
    m["mean_hr"] = 60.0 / float(np.mean(rr))
    nn_mask = np.array([labels[j] == "N" and labels[j + 1] == "N" for j in range(n - 1)], dtype=bool)
    nn = rr[nn_mask]
    m["sinus_hr"] = 60.0 / float(np.mean(nn)) if len(nn) >= 2 else m["mean_hr"]
    k = min(5, len(rr))
    smooth = np.convolve(rr, np.ones(k) / k, mode="valid")
    m["min_hr"], m["max_hr"] = 60.0 / float(np.max(smooth)), 60.0 / float(np.min(smooth))

    pair = nn_mask[:-1] & nn_mask[1:]
    d = np.diff(rr)[pair]
    m["sdnn_ms"] = 1000.0 * float(np.std(nn)) if len(nn) >= 2 else None
    m["rmssd_ms"] = 1000.0 * float(np.sqrt(np.mean(d**2))) if len(d) else None
    m["pnn50_pct"] = 100.0 * float(np.mean(np.abs(d) > 0.05)) if len(d) else None

    pauses = np.flatnonzero(rr > PAUSE_S)
    m["pauses"] = [{"time_s": float(r[j] / fs), "duration_s": float(rr[j])} for j in pauses]
    m["longest_rr_s"] = float(np.max(rr))
    m["irregularity"] = irregularity_index(rr, labels)

    def run_info(cls: str, min_len: int, max_len: int | None = None):
        out = []
        for s, length in runs_of(labels, cls):
            if length >= min_len and (max_len is None or length <= max_len):
                span = (r[s + length - 1] - r[s]) / fs
                rate = 60.0 * (length - 1) / span if span > 0 else None
                out.append({"start_s": float(r[s] / fs), "beats": length, "rate_bpm": rate})
        return out

    m["v_couplets"] = run_info("V", 2, 2)
    m["v_runs"] = run_info("V", 3)
    m["s_runs"] = run_info("S", 3)
    m["bigeminy"] = [{"start_s": float(r[s] / fs), "cycles": c} for s, c in pattern_episodes(labels, ["N", "V"])]
    m["trigeminy"] = [{"start_s": float(r[s] / fs), "cycles": c} for s, c in pattern_episodes(labels, ["N", "N", "V"])]
    return m


def findings(m: dict, signal_quality: float | None = None) -> list[dict]:
    """Translate metrics into plain-language findings with a severity level."""
    out: list[dict] = []

    def add(severity: str, title: str, detail: str):
        out.append({"severity": severity, "title": title, "detail": detail})

    if m["n_beats"] < 3 or "mean_hr" not in m:
        add("alert", "Not enough beats", "Fewer than three heartbeats were detected; check the recording and filters.")
        return out

    if signal_quality is not None and signal_quality < 0.80:
        add("warning", "Poor signal quality",
            f"Beats correlate only {signal_quality:.2f} (median) with their template; noise may cause errors.")

    hr = m["sinus_hr"]
    if hr < BRADY_BPM:
        add("warning", "Bradycardia", f"Underlying rate {hr:.0f} bpm is below {BRADY_BPM:.0f} bpm.")
    elif hr > TACHY_BPM:
        add("warning", "Tachycardia", f"Underlying rate {hr:.0f} bpm is above {TACHY_BPM:.0f} bpm.")
    else:
        add("normal", "Heart rate in normal range", f"Underlying rate {hr:.0f} bpm (60-100 bpm).")

    names = {"V": "Ventricular ectopic beats (PVCs)", "S": "Supraventricular ectopic beats (PACs)",
             "F": "Fusion beats"}
    for c in ("V", "S", "F"):
        cnt, pct = m["counts"][c], m["burden_pct"][c]
        if cnt:
            sev = "alert" if pct >= 10 else "warning" if pct >= 1 or cnt >= 3 else "info"
            add(sev, names[c], f"{cnt} beat{'s' * (cnt != 1)} ({pct:.1f}% of classified beats).")

    if m["v_runs"]:
        longest = max(m["v_runs"], key=lambda x: x["beats"])
        rate = f" at ~{longest['rate_bpm']:.0f} bpm" if longest["rate_bpm"] else ""
        add("alert", "Run of ventricular beats",
            f"{len(m['v_runs'])} run(s) of 3+ consecutive PVCs; longest {longest['beats']} beats{rate}. "
            "Three or more at >100 bpm is called non-sustained ventricular tachycardia.")
    if m["v_couplets"]:
        add("warning", "Ventricular couplets", f"{len(m['v_couplets'])} pair(s) of back-to-back PVCs.")
    if m["bigeminy"]:
        add("warning", "Ventricular bigeminy",
            f"Every other beat is a PVC for {max(e['cycles'] for e in m['bigeminy'])}+ cycles.")
    if m["trigeminy"]:
        add("warning", "Ventricular trigeminy",
            f"Every third beat is a PVC for {max(e['cycles'] for e in m['trigeminy'])}+ cycles.")
    if m["s_runs"]:
        longest = max(m["s_runs"], key=lambda x: x["beats"])
        add("warning", "Run of supraventricular beats",
            f"{len(m['s_runs'])} run(s) of 3+ consecutive PACs; longest {longest['beats']} beats.")
    if m["pauses"]:
        longest = max(p["duration_s"] for p in m["pauses"])
        add("alert" if longest >= LONG_PAUSE_S else "warning", "Pause",
            f"{len(m['pauses'])} gap(s) longer than {PAUSE_S:.0f} s between beats; longest {longest:.1f} s.")
    irr = m.get("irregularity")
    if irr is not None and irr > IRREGULARITY_THRESHOLD:
        add("warning", "Irregularly irregular rhythm",
            f"RR irregularity index {irr:.2f} exceeds {IRREGULARITY_THRESHOLD:.2f}: a pattern seen in "
            "atrial fibrillation. Confirm by looking for absent P waves.")
    return out


def verdict(found: list[dict]) -> str:
    return "abnormal" if any(f["severity"] in ("warning", "alert") for f in found) else "normal"


def calibrate_irregularity(window_s: float = 30.0, classifier: dict | None = None) -> dict:
    """Pick the AF threshold on DS1 30 s windows, then report it on DS2.

    Runs end-to-end (detector + classifier labels), exactly as a report would.
    Windows where the index is undefined count as "not AF".
    """
    from . import data
    from .model import load_classifier
    from .report import analyze

    clf = classifier or load_classifier()

    def windows(records):
        xs, ys = [], []
        for name in records:
            rec = data.load_record(name)
            an = analyze(rec.signal, rec.fs, clf)
            w = int(window_s * rec.fs)
            for start in range(0, len(rec.signal) - w + 1, w):
                sel = np.flatnonzero((an.r_peaks >= start) & (an.r_peaks < start + w))
                if len(sel) < 3:
                    continue
                rr = np.diff(an.r_peaks[sel]) / an.fs
                idx = irregularity_index(rr, [an.labels[i] for i in sel])
                xs.append(-1.0 if idx is None else idx)
                ys.append(rec.rhythm_at(start + w // 2) == "(AFIB")
        return np.array(xs), np.array(ys)

    def scores(x, y, th):
        p = x > th
        se = float(np.mean(p[y])) if y.any() else 0.0
        sp = float(np.mean(~p[~y]))
        return {"sensitivity": se, "specificity": sp, "balanced_accuracy": (se + sp) / 2,
                "windows": int(len(y)), "af_windows": int(y.sum())}

    x1, y1 = windows([r for r in data.DS1])
    grid = np.round(np.arange(0.03, 0.40, 0.01), 2)
    best = max(grid, key=lambda t: scores(x1, y1, t)["balanced_accuracy"])
    x2, y2 = windows([r for r in data.DS2])
    return {"threshold": float(best), "window_s": window_s,
            "ds1": scores(x1, y1, best), "ds2": scores(x2, y2, best)}
