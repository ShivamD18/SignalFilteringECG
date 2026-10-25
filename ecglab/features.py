"""Beat-level feature extraction for arrhythmia classification.

Every beat is described by three families of features:

RR (rhythm) features
    When did the beat arrive compared with its neighbours? A premature beat
    (short pre-RR) followed by a compensatory pause (long post-RR) is the
    classic signature of an ectopic beat. Ratios to the *local* rhythm make
    the features comparable between a resting and an exercising patient.

Template features
    How different is the beat from this patient's own dominant beat? The
    template is the sample-wise median of the surrounding ~60 beats, built
    without any labels. Comparing against the patient's template rather than
    against "textbook" values is what lets a model trained on one group of
    people generalise to new ones (ventricular beats look wide and different;
    supraventricular beats look normal but come early).

Morphology samples
    A coarse, amplitude-normalised copy of the waveform from 250 ms before to
    450 ms after the R peak, one point every ~28 ms.

The browser version (``docs/js/features.js``) mirrors this module exactly.
"""

from __future__ import annotations

import numpy as np

FS = 360.0
PRE_S, POST_S = 0.25, 0.45     # beat window around the R peak
QRS_HALF_S = 0.10              # QRS region = R +/- 100 ms
WIDTH_HALF_S = 0.15            # region used to measure QRS width
LOCAL_BEATS = 10               # RR neighbours on each side for the local rhythm
LONG_BEATS = 100               # ... and for the longer-term rhythm
TEMPLATE_BEATS = 30            # beats on each side used to build the template
MORPH_STEP = 10                # downsampling step for morphology samples

RR_FEATURES = ["pre_rr", "post_rr", "local_rr", "long_rr",
               "pre_over_local", "post_over_local", "post_over_pre", "local_over_long",
               "pre_over_prev", "pre_minus_local"]
TEMPLATE_FEATURES = ["corr_qrs", "corr_beat", "corr_twave", "amp_ratio", "diff_rms",
                     "energy_ratio", "r_value", "qrs_width", "width_ratio", "qrs_amp_mv"]


def _geometry(fs: float) -> tuple[int, int, int, int]:
    if fs != FS:
        raise ValueError(f"features are defined at {FS:g} Hz; resample the ECG first (got {fs} Hz)")
    return (int(round(PRE_S * fs)), int(round(POST_S * fs)),
            int(round(QRS_HALF_S * fs)), int(round(WIDTH_HALF_S * fs)))


def _morph_names(fs: float = FS) -> list[str]:
    pre, post, _, _ = _geometry(fs)
    return [f"morph_{int(round((i - pre) * 1000 / fs)):+d}ms" for i in range(0, pre + post + 1, MORPH_STEP)]


FEATURE_NAMES = RR_FEATURES + TEMPLATE_FEATURES + _morph_names()


def beat_windows(ecg: np.ndarray, r_peaks: np.ndarray, fs: float = FS):
    """Cut a fixed window around each beat; returns (indices of beats kept, windows)."""
    pre, post, _, _ = _geometry(fs)
    r = np.asarray(r_peaks, dtype=np.int64)
    idx = np.flatnonzero((r - pre >= 0) & (r + post < len(ecg)))
    if len(idx) == 0:
        return idx, np.empty((0, pre + post + 1))
    windows = np.stack([ecg[r[i] - pre: r[i] + post + 1] for i in idx])
    windows = windows - np.median(windows, axis=1, keepdims=True)
    return idx, windows


def beat_templates(windows: np.ndarray, k: int = TEMPLATE_BEATS) -> np.ndarray:
    """Sample-wise median of the 2k+1 surrounding beats (the patient's dominant beat)."""
    m = len(windows)
    out = np.empty_like(windows)
    for i in range(m):
        out[i] = np.median(windows[max(0, i - k): min(m, i + k + 1)], axis=0)
    return out


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    num = np.sum(a * b, axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _energy_width(seg: np.ndarray, fs: float) -> np.ndarray:
    """Time between 10 % and 90 % of the cumulative slope energy: a robust QRS width."""
    c = np.cumsum(np.diff(seg, axis=1) ** 2, axis=1)
    total = c[:, -1:]
    lo = np.argmax(c >= 0.1 * total, axis=1)
    hi = np.argmax(c >= 0.9 * total, axis=1)
    return np.where(total[:, 0] > 0, (hi - lo) / fs, 0.0)


def extract_features(ecg: np.ndarray, r_peaks: np.ndarray, fs: float = FS):
    """Return ``(X, beat_index)``: one feature row per usable beat.

    A beat is usable when it has a previous and a next beat (for RR features)
    and its whole window lies inside the recording.
    """
    pre, post, qh, wh = _geometry(fs)
    r = np.asarray(r_peaks, dtype=np.int64)
    n = len(r)
    win_idx, W = beat_windows(np.asarray(ecg, dtype=np.float64), r, fs)
    if len(win_idx) == 0 or n < 3:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0, dtype=np.int64)
    T = beat_templates(W)

    use = (win_idx >= 1) & (win_idx <= n - 2)
    beat_idx, W, T = win_idx[use], W[use], T[use]

    # --- RR features (seconds and dimensionless ratios) ---
    rr = np.diff(r) / fs
    pre_rr = rr[beat_idx - 1]
    post_rr = rr[beat_idx]
    local_rr = np.array([np.median(rr[max(0, i - LOCAL_BEATS): min(n - 1, i + LOCAL_BEATS)]) for i in beat_idx])
    long_rr = np.array([np.median(rr[max(0, i - LONG_BEATS): min(n - 1, i + LONG_BEATS)]) for i in beat_idx])
    prev_rr = np.where(beat_idx >= 2, rr[np.maximum(beat_idx - 2, 0)], local_rr)
    rr_feats = np.column_stack([
        pre_rr, post_rr, local_rr, long_rr,
        pre_rr / local_rr, post_rr / local_rr, post_rr / pre_rr, local_rr / long_rr,
        pre_rr / prev_rr, pre_rr - local_rr,
    ])

    # --- Template features ---
    qrs = slice(pre - qh, pre + qh + 1)
    twave = slice(pre + qh, pre + post + 1)
    wq, tq = W[:, qrs], T[:, qrs]
    amp_t = np.maximum(np.ptp(tq, axis=1), 1e-3)
    amp_w = np.ptp(wq, axis=1)
    width_w = _energy_width(W[:, pre - wh: pre + wh + 1], fs)
    width_t = _energy_width(T[:, pre - wh: pre + wh + 1], fs)
    tmpl_feats = np.column_stack([
        _rowwise_corr(wq, tq),
        _rowwise_corr(W, T),
        _rowwise_corr(W[:, twave], T[:, twave]),
        amp_w / amp_t,
        np.sqrt(np.mean((wq - tq) ** 2, axis=1)) / amp_t,
        np.sum(wq**2, axis=1) / np.maximum(np.sum(tq**2, axis=1), 1e-9),
        W[:, pre] / amp_t,
        width_w,
        width_w / np.maximum(width_t, 1.0 / fs),
        amp_w,
    ])

    morph = W[:, ::MORPH_STEP] / amp_t[:, None]
    return np.column_stack([rr_feats, tmpl_feats, morph]), beat_idx
