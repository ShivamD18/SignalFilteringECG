"""End-to-end analysis (filter -> detect -> classify -> interpret) and the ECG report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data
from .detect import detect_r_peaks, match_beats
from .features import FEATURE_NAMES, FS, extract_features
from .filters import TRAINING_FILTER, FilterConfig, preprocess, resample_linear
from .model import CLASSES, load_classifier
from .rhythm import findings, rhythm_metrics, verdict

DISCLAIMER = ("Educational tool only. This is not a medical device, it has not been clinically "
              "validated, and it must never be used to diagnose or treat anyone.")
CLASS_COLORS = {"N": "#2a7d4f", "S": "#c47f00", "V": "#c62828", "F": "#6a3fb5", "?": "#888888"}


@dataclass
class Analysis:
    ecg: np.ndarray               # conditioned signal at 360 Hz
    r_peaks: np.ndarray
    labels: list[str]             # N/S/V/F per beat, "?" if it could not be classified
    probabilities: np.ndarray     # (beats, classes); NaN rows for "?" beats
    metrics: dict
    findings: list[dict]
    verdict: str
    signal_quality: float | None
    title: str = "ECG"
    reference: dict | None = None
    fs: float = FS

    @property
    def duration_s(self) -> float:
        return len(self.ecg) / self.fs


def analyze(signal: np.ndarray, fs: float, classifier: dict | None = None,
            filter_config: FilterConfig = TRAINING_FILTER, title: str = "ECG") -> Analysis:
    clf = classifier or load_classifier()
    ecg = preprocess(signal, fs, filter_config)
    if fs != FS:
        ecg = resample_linear(ecg, fs, FS)  # already low-passed at 40 Hz, so no aliasing
    r = detect_r_peaks(ecg, FS)
    X, idx = extract_features(ecg, r, FS)

    labels = ["?"] * len(r)
    probs = np.full((len(r), len(CLASSES)), np.nan)
    quality = None
    if len(idx):
        p = clf["model"].predict_proba(X)
        probs[idx] = p
        for i, k in zip(idx, p.argmax(axis=1)):
            labels[i] = CLASSES[k]
        quality = float(np.median(X[:, FEATURE_NAMES.index("corr_qrs")]))

    m = rhythm_metrics(r, labels, FS)
    found = findings(m, quality)
    return Analysis(ecg=ecg, r_peaks=r, labels=labels, probabilities=probs, metrics=m,
                    findings=found, verdict=verdict(found), signal_quality=quality, title=title)


def analyze_record(name: str, start_s: float = 0.0, duration_s: float | None = None,
                   classifier: dict | None = None) -> Analysis:
    """Analyse (part of) an MIT-BIH record and compare with the cardiologists' labels."""
    rec = data.load_record(name)
    a = int(start_s * rec.fs)
    b = len(rec.signal) if duration_s is None else min(len(rec.signal), a + int(duration_s * rec.fs))
    span = f"{start_s:g}-{b / rec.fs:g} s"
    an = analyze(rec.signal[a:b], rec.fs, classifier, title=f"MIT-BIH record {name} ({rec.lead}, {span})")

    sel = (rec.beat_samples >= a) & (rec.beat_samples < b)
    ref_pos = rec.beat_samples[sel] - a
    ref_cls = rec.beat_classes[sel]
    match = match_beats(ref_pos, an.r_peaks, FS)
    pairs = [(ref_cls[j], an.labels[i]) for i, j in enumerate(match.matched_ref)
             if j >= 0 and an.labels[i] != "?" and ref_cls[j] in CLASSES]
    agree = sum(t == p for t, p in pairs)
    agree_bin = sum((t == "N") == (p == "N") for t, p in pairs)
    an.reference = {
        "detector": {"tp": match.tp, "fp": match.fp, "fn": match.fn,
                     "sensitivity": match.sensitivity, "ppv": match.ppv},
        "compared_beats": len(pairs),
        "class_agreement": agree / max(len(pairs), 1),
        "normal_abnormal_agreement": agree_bin / max(len(pairs), 1),
        "reference_counts": {c: int(np.sum(ref_cls == c)) for c in ("N", "S", "V", "F", "Q")},
        "rhythms": sorted({rec.rhythm_at(int(s)) for s in rec.beat_samples[sel]}),
        "split": "DS1" if name in data.DS1 else "DS2" if name in data.DS2 else "excluded (paced)",
    }
    return an


def _fmt(v, spec=".0f", unit=""):
    return "n/a" if v is None else f"{v:{spec}}{unit}"


def to_markdown(an: Analysis, figure: str | None = None) -> str:
    m = an.metrics
    flagged = [f for f in an.findings if f["severity"] in ("warning", "alert")]
    lines = [f"# ECG report: {an.title}", "", f"> **{DISCLAIMER}**", ""]
    if an.verdict == "abnormal":
        lines.append(f"## Overall: ABNORMAL ({len(flagged)} finding{'s' * (len(flagged) != 1)} need attention)")
    else:
        lines.append("## Overall: NORMAL (no significant abnormality detected)")
    lines += ["", "| Measurement | Value |", "|---|---|",
              f"| Duration analysed | {an.duration_s:.1f} s |",
              f"| Beats detected | {m['n_beats']} |",
              f"| Mean / underlying heart rate | {_fmt(m.get('mean_hr'))} / {_fmt(m.get('sinus_hr'))} bpm |",
              f"| Min / max heart rate (5-beat average) | {_fmt(m.get('min_hr'))} / {_fmt(m.get('max_hr'))} bpm |",
              f"| SDNN / RMSSD / pNN50 | {_fmt(m.get('sdnn_ms'))} ms / {_fmt(m.get('rmssd_ms'))} ms / "
              f"{_fmt(m.get('pnn50_pct'), '.1f', ' %')} |",
              "| Beat classes | " + ", ".join(f"{c}: {m['counts'][c]}" for c in ("N", "S", "V", "F", "?")) + " |",
              f"| RR irregularity index | {_fmt(m.get('irregularity'), '.3f')} |",
              f"| Signal quality (median QRS template correlation) | {_fmt(an.signal_quality, '.2f')} |",
              "", "## Findings", ""]
    for f in an.findings:
        lines.append(f"- **[{f['severity'].upper()}] {f['title']}**: {f['detail']}")
    if an.reference:
        ref = an.reference
        d = ref["detector"]
        lines += ["", "## Comparison with the cardiologists' annotations", "",
                  f"- Beat detection: sensitivity {d['sensitivity']:.1%}, positive predictivity {d['ppv']:.1%} "
                  f"(TP {d['tp']}, FP {d['fp']}, FN {d['fn']})",
                  f"- Beat class agreement: {ref['class_agreement']:.1%} of {ref['compared_beats']} beats; "
                  f"normal-vs-abnormal agreement {ref['normal_abnormal_agreement']:.1%}",
                  "- Reference beat counts: " + ", ".join(f"{k}: {v}" for k, v in ref["reference_counts"].items()),
                  "- Annotated rhythms: " + ", ".join(ref["rhythms"])]
        if ref["split"] == "DS1":
            lines.append("- Note: this patient is in the training set (DS1), so agreement is optimistic. "
                         "Use a DS2 record to see performance on an unseen patient.")
    if figure:
        lines += ["", f"![ECG strip]({figure})"]
    lines += ["", "## Method", "",
              "Zero-phase Butterworth band-pass (0.5-40 Hz) -> Pan-Tompkins QRS detection -> RR, "
              "template and morphology features -> machine-learning beat classifier (AAMI N/S/V/F) "
              "trained on MIT-BIH DS1 patients -> rule-based rhythm interpretation.", ""]
    return "\n".join(lines)


def plot(an: Analysis, path: Path, seconds: float = 10.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = an.fs
    n = min(len(an.ecg), int(seconds * fs))
    t = np.arange(n) / fs
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 6.5), gridspec_kw={"height_ratios": [2.2, 1]})

    # ECG paper: 1 mm = 0.04 s / 0.1 mV, bold every 5 mm
    lo, hi = np.min(an.ecg[:n]) - 0.3, np.max(an.ecg[:n]) + 0.5
    for x in np.arange(0, t[-1] + 0.04, 0.04):
        ax.axvline(x, color="#f4c7c7", lw=1.0 if round(x / 0.04) % 5 == 0 else 0.3, zorder=0)
    for y in np.arange(np.floor(lo * 10) / 10, hi, 0.1):
        ax.axhline(y, color="#f4c7c7", lw=1.0 if round(y * 10) % 5 == 0 else 0.3, zorder=0)
    ax.plot(t, an.ecg[:n], color="#111", lw=0.9)
    for r, lab in zip(an.r_peaks, an.labels):
        if r < n:
            ax.text(r / fs, hi - 0.15, lab, color=CLASS_COLORS[lab], ha="center", fontsize=10, weight="bold")
            ax.plot(r / fs, an.ecg[r], "o", ms=4, color=CLASS_COLORS[lab])
    ax.set(xlim=(0, t[-1]), ylim=(lo, hi), xlabel="Time (s)", ylabel="mV",
           title=f"{an.title}: first {t[-1]:.0f} s, beats labelled by the classifier")

    rr_t = an.r_peaks[1:] / fs
    rr = np.diff(an.r_peaks) / fs
    colors = [CLASS_COLORS[lab] for lab in an.labels[1:]]
    ax2.scatter(rr_t, rr * 1000, c=colors, s=10)
    ax2.set(xlabel="Time (s)", ylabel="RR (ms)", title="RR tachogram (colour = class of the beat ending the interval)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=CLASS_COLORS[c], label=f"{c}: {data.CLASS_NAMES[c]}")
               for c in ("N", "S", "V", "F")]
    ax2.legend(handles=handles, loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(an: Analysis, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    plot(an, png)
    md = out_dir / f"{stem}.md"
    md.write_text(to_markdown(an, png.name), encoding="utf-8")
    return md


def load_csv(path: Path, column: int | str = 0) -> np.ndarray:
    """Read one ECG column (millivolts) from a CSV/TXT file; a header row is optional."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.reader(fh) if row]
    header = rows[0]
    try:
        float(header[0 if isinstance(column, str) else column])
        body = rows
    except ValueError:
        body = rows[1:]
        if isinstance(column, str):
            column = header.index(column)
    if isinstance(column, str):
        raise ValueError(f"column {column!r} not found (file has no header row)")
    return np.array([float(r[column]) for r in body])
