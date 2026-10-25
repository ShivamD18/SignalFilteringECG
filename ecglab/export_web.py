"""Export sample recordings, real noise, the trained model and its metrics for the website.

Writes to ``docs/data`` (served by GitHub Pages) and refreshes the
Python-vs-JavaScript parity fixture in ``tests/fixtures``. Run after
``python -m ecglab train``.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import signal

from . import data
from .detect import detect_r_peaks, match_beats, pan_tompkins
from .features import FEATURE_NAMES, extract_features
from .filters import TRAINING_FILTER, butter_sos, notch_sos, preprocess
from .model import CLASSES, MODELS_DIR, load_classifier
from .rhythm import IRREGULARITY_THRESHOLD, calibrate_irregularity, findings, rhythm_metrics, verdict

WEB_DATA = data.ROOT / "docs" / "data"
FIXTURES = data.ROOT / "tests" / "fixtures"
SEGMENT_S = 60
PARITY_RECORD = "200"


def _score_windows(rec: data.Record, score) -> int:
    """Start sample of the 60 s window (10 s steps) that maximises ``score``."""
    w = int(SEGMENT_S * rec.fs)
    best, best_s = 0, -np.inf
    classes = rec.beat_classes
    for start in range(0, len(rec.signal) - w + 1, int(10 * rec.fs)):
        sel = (rec.beat_samples >= start) & (rec.beat_samples < start + w)
        s = score(rec, start, w, sel, classes[sel], rec.beat_symbols[sel])
        if s > best_s:
            best, best_s = start, s
    return best


def _count(cls, c):
    return float(np.sum(cls == c))


def _rhythm_frac(label):
    def f(rec, start, w, sel, cls, sym):
        beats = rec.beat_samples[sel]
        return np.mean([rec.rhythm_at(int(b)) == label for b in beats]) if len(beats) else 0.0
    return f


def _bigeminy(rec, start, w, sel, cls, sym):
    s = "".join(cls)
    return float(s.count("NVNV"))


def _pause_and_s(rec, start, w, sel, cls, sym):
    rr = np.diff(rec.beat_samples[sel]) / rec.fs
    return (rr.max() if len(rr) else 0) * 100 + _count(cls, "S")


def _noisiest(rec, start, w, sel, cls, sym):
    seg = rec.signal[start:start + w]
    hf = seg - preprocess(seg, rec.fs)
    return float(np.std(hf))


SAMPLES = [
    ("100", "Normal sinus rhythm",
     "A healthy-looking rhythm: regular P-QRS-T complexes at 60-100 bpm. Start here.",
     lambda rec, s, w, sel, cls, sym: -float(np.sum(cls != "N")) - s / 1e9),
    ("212", "Right bundle branch block",
     "Wide, notched QRS complexes. Still a normal (N) beat in AAMI terms because it starts in the sinus node.",
     lambda rec, s, w, sel, cls, sym: float(np.sum(sym == "R"))),
    ("200", "Ventricular bigeminy",
     "Every other beat is a premature ventricular contraction (PVC): wide, bizarre, early, then a pause.",
     _bigeminy),
    ("233", "Frequent multiform PVCs",
     "Many PVCs with different shapes, including couplets and short runs.",
     lambda rec, s, w, sel, cls, sym: _count(cls, "V")),
    ("213", "Fusion beats",
     "A ventricular and a normal beat fire at the same time and merge into a hybrid shape (F). Hard for any classifier.",
     lambda rec, s, w, sel, cls, sym: _count(cls, "F") * 3 + _count(cls, "V")),
    ("221", "Atrial fibrillation",
     "No P waves and an 'irregularly irregular' rhythm. Each beat is normal, so the rhythm analysis must catch it.",
     lambda rec, s, w, sel, cls, sym: _rhythm_frac("(AFIB")(rec, s, w, sel, cls, sym) * 100 - _count(cls, "V")),
    ("232", "Bradycardia, APCs and long pauses",
     "Slow ectopic atrial rhythm with pauses of several seconds. The abnormal beats are the patient's *usual* "
     "rhythm here, which defeats classifiers that compare beats with the patient's norm.",
     _pause_and_s),
    ("231", "2:1 heart block",
     "Every second P wave is not conducted, halving the ventricular rate. Watch the long, regular RR intervals.",
     _rhythm_frac("(BII")),
    ("105", "Heavy real-world noise",
     "Large muscle and motion artifacts from an ambulatory recording. A stress test for filtering and detection.",
     _noisiest),
    ("209", "Supraventricular tachycardia",
     "Bursts of rapid supraventricular (S) beats. This patient was in the training set (DS1).",
     _rhythm_frac("(SVTA")),
]


def _round(x, nd=6):
    return [round(float(v), nd) for v in x]


def export_records() -> list[dict]:
    out_dir = WEB_DATA / "records"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for name, title, note, score in SAMPLES:
        rec = data.load_record(name)
        start = _score_windows(rec, score)
        w = int(SEGMENT_S * rec.fs)
        sel = (rec.beat_samples >= start) & (rec.beat_samples < start + w)
        seg = rec.signal[start:start + w]
        rhythms = sorted({rec.rhythm_at(int(b)) for b in rec.beat_samples[sel]})
        payload = {
            "record": name, "title": title, "note": note, "fs": rec.fs, "lead": rec.lead,
            "start_s": start / rec.fs, "units": "microvolts",
            "split": "DS1 (training patient)" if name in data.DS1 else "DS2 (unseen test patient)",
            "signal_uv": [int(round(v * 1000)) for v in seg],
            "beats": [[int(s - start), str(sym), data.SYMBOL_TO_AAMI[sym]]
                      for s, sym in zip(rec.beat_samples[sel], rec.beat_symbols[sel])],
            "rhythms": rhythms,
        }
        (out_dir / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")))
        counts = {c: int(np.sum(rec.beat_classes[sel] == c)) for c in ("N", "S", "V", "F", "Q")}
        index.append({k: payload[k] for k in ("record", "title", "note", "lead", "start_s", "split", "rhythms")}
                     | {"counts": counts})
        print(f"  record {name}: {title} @ {start / rec.fs:.0f}s  {counts}")
    return index


def export_noise() -> None:
    out = {}
    for name, title in data.NOISE_RECORDS.items():
        x = data.load_noise(name)
        seg = x[int(60 * data.FS): int((60 + SEGMENT_S) * data.FS)]
        seg = seg - np.mean(seg)
        out[name] = {"title": title, "signal_uv": [int(round(v * 1000)) for v in seg]}
    (WEB_DATA / "noise.json").write_text(json.dumps(out, separators=(",", ":")))


def model_to_json(clf: dict) -> dict:
    m, name = clf["model"], clf["name"]
    base = {"type": name, "classes": list(CLASSES), "features": FEATURE_NAMES, "filter": clf["filter"]}
    if name in ("logistic_regression", "mlp"):
        scaler, est = m.steps[0][1], m.steps[-1][1]
        base["scaler"] = {"mean": _round(scaler.mean_, 9), "scale": _round(scaler.scale_, 9)}
        if name == "logistic_regression":
            base["coef"] = [_round(row, 9) for row in est.coef_]
            base["intercept"] = _round(est.intercept_, 9)
        else:
            base["layers"] = [{"W": [_round(r, 9) for r in W], "b": _round(b, 9)}
                              for W, b in zip(est.coefs_, est.intercepts_)]
        return base
    if name == "gradient_boosting":
        base["baseline"] = _round(np.ravel(m._baseline_prediction), 12)
        trees = []
        for iteration in m._predictors:
            it = []
            for pred in iteration:
                nodes = pred.nodes
                leaf = nodes["is_leaf"].astype(bool)
                it.append({
                    "f": [int(-1 if lf else f) for f, lf in zip(nodes["feature_idx"], leaf)],
                    # Thresholds keep full precision: bin edges can equal feature values exactly
                    # (e.g. QRS width = k / 360 s), so rounding would flip `x <= t` decisions.
                    "t": [float(f"{v:.10g}") if lf else float(t) for v, t, lf in
                          zip(nodes["value"], nodes["num_threshold"], leaf)],
                    "l": nodes["left"].astype(int).tolist(),
                    "r": nodes["right"].astype(int).tolist(),
                    "m": nodes["missing_go_to_left"].astype(int).tolist(),
                })
            trees.append(it)
        base["trees"] = trees
        return base
    raise ValueError(f"model type {name!r} cannot be exported to the browser")


def export_model() -> dict:
    clf = load_classifier()
    model_json = model_to_json(clf)
    (WEB_DATA / "model.json").write_text(json.dumps(model_json, separators=(",", ":")))
    return clf


def export_metrics() -> None:
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    print("  evaluating detector on all records ...")
    per_record = []
    tp = fn = fp = 0
    for name in data.ALL_RECORDS:
        rec = data.load_record(name)
        m = match_beats(rec.beat_samples, detect_r_peaks(preprocess(rec.signal, rec.fs), rec.fs), rec.fs)
        tp, fp, fn = tp + m.tp, fp + m.fp, fn + m.fn
        per_record.append({"record": name, "tp": m.tp, "fp": m.fp, "fn": m.fn,
                           "sensitivity": m.sensitivity, "ppv": m.ppv})
    metrics["detector"] = {"tp": tp, "fp": fp, "fn": fn, "sensitivity": tp / (tp + fn),
                           "ppv": tp / (tp + fp), "records": per_record}
    print("  calibrating AF irregularity screen ...")
    metrics["af_screening"] = calibrate_irregularity()
    metrics["af_screening"]["threshold_in_use"] = IRREGULARITY_THRESHOLD
    (WEB_DATA / "metrics.json").write_text(json.dumps(metrics, separators=(",", ":")))


def export_parity_fixture(clf: dict) -> None:
    """Reference outputs the JavaScript port must reproduce (see tests/test_js_parity.py)."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    payload = json.loads((WEB_DATA / "records" / f"{PARITY_RECORD}.json").read_text())
    x = np.array(payload["signal_uv"], dtype=np.float64) / 1000.0
    fs = payload["fs"]
    ecg = preprocess(x, fs, TRAINING_FILTER)
    pt = pan_tompkins(ecg, fs)
    X, idx = extract_features(ecg, pt.r_peaks, fs)
    proba = clf["model"].predict_proba(X)
    labels = ["?"] * len(pt.r_peaks)
    for i, k in zip(idx, proba.argmax(axis=1)):
        labels[i] = CLASSES[k]
    quality = float(np.median(X[:, FEATURE_NAMES.index("corr_qrs")]))
    m = rhythm_metrics(pt.r_peaks, labels, fs)
    found = findings(m, quality)

    rng = np.random.default_rng(0)
    probe = np.cumsum(rng.normal(size=600)) * 0.05 + np.sin(np.arange(600) / 9)
    designs = []
    for kind, fc, order in [("highpass", 0.5, 2), ("highpass", 0.05, 1), ("highpass", 1.0, 5),
                            ("lowpass", 40.0, 4), ("lowpass", 100.0, 3), ("lowpass", 15.0, 6)]:
        sos = butter_sos(kind, fc, fs, order)
        designs.append({"kind": kind, "cutoff": fc, "order": order, "fs": fs, "n_sections": len(sos),
                        "filtfilt": _round(signal.sosfiltfilt(sos, probe), 12)})
    notch = notch_sos(60.0, fs, 30.0)

    fixture = {
        "record": PARITY_RECORD, "fs": fs,
        "filtered_every_10": _round(ecg[::10], 12),
        "integrated_every_10": _round(pt.integrated[::10], 12),
        "r_peaks": pt.r_peaks.tolist(),
        "beat_index": idx.tolist(),
        "features": [[float(v) for v in r] for r in X],  # full precision: values can sit on tree thresholds
        "probabilities": [_round(r, 12) for r in proba],
        "labels": labels,
        "signal_quality": quality,
        "metrics": json.loads(json.dumps(m)),
        "findings": found,
        "verdict": verdict(found),
        "probe": _round(probe, 15),
        "designs": designs,
        "notch_60": _round(notch[0], 15),
    }
    (FIXTURES / "parity.json").write_text(json.dumps(fixture, separators=(",", ":")))


def export_all() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    print("Exporting sample records ...")
    index = export_records()
    (WEB_DATA / "records.json").write_text(json.dumps(index, indent=1))
    print("Exporting noise ...")
    export_noise()
    print("Exporting model ...")
    clf = export_model()
    print("Exporting metrics ...")
    export_metrics()
    print("Writing parity fixture ...")
    export_parity_fixture(clf)
    print("Done:", WEB_DATA)
