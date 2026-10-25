import json
from pathlib import Path

import numpy as np
import pytest
from conftest import FS, synthetic_ecg

from ecglab import data, detect, features, filters, rhythm
from ecglab.model import JsonModel, WEB_MODEL

ROOT = Path(__file__).resolve().parent.parent


# ---------- detection ----------
def test_pan_tompkins_finds_every_beat(ecg):
    x, truth = ecg
    m = detect.match_beats(truth, detect.detect_r_peaks(filters.preprocess(x, FS), FS), FS)
    assert m.fn == 0 and m.fp == 0


def test_pan_tompkins_survives_noise():
    x, truth = synthetic_ecg(duration=60)
    rng = np.random.default_rng(0)
    t = np.arange(len(x)) / FS
    noisy = x + 0.05 * rng.normal(size=len(x)) + 0.5 * np.sin(2 * np.pi * 0.25 * t) + 0.2 * np.sin(2 * np.pi * 60 * t)
    m = detect.match_beats(truth, detect.detect_r_peaks(filters.preprocess(noisy, FS), FS), FS)
    assert m.sensitivity > 0.98 and m.ppv > 0.98


def test_pan_tompkins_finds_premature_beats():
    x, truth = synthetic_ecg(premature=(5, 12, 20))
    m = detect.match_beats(truth, detect.detect_r_peaks(filters.preprocess(x, FS), FS), FS)
    assert m.fn == 0 and m.fp == 0


def test_match_beats_counts():
    m = detect.match_beats(np.array([100, 500, 900]), np.array([110, 520, 1500]), FS)
    assert (m.tp, m.fp, m.fn) == (2, 1, 1)
    assert list(m.matched_ref) == [0, 1, -1]


def test_refine_peaks_moves_to_extremum():
    x = np.zeros(1000)
    x[510] = 2.0
    assert detect.refine_peaks(x, np.array([500]), FS)[0] == 510


# ---------- features ----------
def test_feature_matrix_shape(ecg):
    x, truth = ecg
    X, idx = features.extract_features(filters.preprocess(x, FS), truth, FS)
    assert X.shape[1] == len(features.FEATURE_NAMES) == 46
    assert idx[0] >= 1 and idx[-1] <= len(truth) - 2
    assert np.isfinite(X).all()


def test_premature_wide_beat_stands_out():
    x, truth = synthetic_ecg(duration=40, premature=(10, 25))
    X, idx = features.extract_features(filters.preprocess(x, FS), truth, FS)
    col = {n: i for i, n in enumerate(features.FEATURE_NAMES)}
    pvc = np.isin(idx, [10, 25])
    assert np.all(X[pvc, col["pre_over_local"]] < 0.75)
    assert np.all(X[pvc, col["post_over_pre"]] > 1.5)
    assert X[pvc, col["corr_qrs"]].max() < X[~pvc, col["corr_qrs"]].min()
    assert X[pvc, col["qrs_width"]].min() > np.median(X[~pvc, col["qrs_width"]])


def test_features_require_360hz(ecg):
    x, truth = ecg
    with pytest.raises(ValueError):
        features.extract_features(x, truth, 250.0)


# ---------- rhythm ----------
def test_runs_and_patterns():
    labels = list("NNVVNVNVNVNVNNVVVN")
    assert rhythm.runs_of(labels, "V") == [(2, 2), (5, 1), (7, 1), (9, 1), (11, 1), (14, 3)]
    assert rhythm.pattern_episodes(labels, ["N", "V"]) == [(4, 4)]


def _regular(hr, n=60):
    return (np.arange(n) * 60 / hr * FS).astype(int)


def test_normal_rhythm_is_normal():
    r = _regular(72)
    labels = ["?"] + ["N"] * (len(r) - 2) + ["?"]
    m = rhythm.rhythm_metrics(r, labels, FS)
    assert m["mean_hr"] == pytest.approx(72, abs=0.5)
    f = rhythm.findings(m, 0.99)
    assert rhythm.verdict(f) == "normal"


def test_bradycardia_and_pause():
    r = list(_regular(45, 30))
    r = np.array(r[:15] + [p + int(2.5 * FS) for p in r[15:]])
    labels = ["N"] * len(r)
    titles = {f["title"] for f in rhythm.findings(rhythm.rhythm_metrics(r, labels, FS))}
    assert {"Bradycardia", "Pause"} <= titles


def test_bigeminy_is_abnormal():
    r = _regular(75, 40)
    labels = ["N", "V"] * 20
    f = rhythm.findings(rhythm.rhythm_metrics(r, labels, FS))
    titles = {x["title"] for x in f}
    assert "Ventricular bigeminy" in titles and "Ventricular ectopic beats (PVCs)" in titles
    assert rhythm.verdict(f) == "abnormal"


def test_bigeminy_is_not_mistaken_for_af_even_when_mislabelled():
    rr = np.tile([0.55, 0.95], 30)                       # short-long alternation
    r = np.concatenate([[0], np.cumsum(rr)]) * FS
    labels = ["N", "S"] * 30 + ["N"]                     # ectopics mislabelled as S, not V
    m = rhythm.rhythm_metrics(r.astype(int), labels, FS)
    assert m["irregularity"] < rhythm.IRREGULARITY_THRESHOLD


def test_irregular_rhythm_flagged():
    rng = np.random.default_rng(3)
    r = np.cumsum(rng.uniform(0.45, 1.1, size=60) * FS).astype(int)
    m = rhythm.rhythm_metrics(r, ["N"] * len(r), FS)
    assert m["irregularity"] > rhythm.IRREGULARITY_THRESHOLD
    assert "Irregularly irregular rhythm" in {x["title"] for x in rhythm.findings(m)}


# ---------- dataset definitions ----------
def test_inter_patient_split_is_clean():
    assert len(data.DS1) == len(data.DS2) == 22
    assert not set(data.DS1) & set(data.DS2)
    assert not (set(data.DS1) | set(data.DS2)) & set(data.PACED_RECORDS)
    assert set(data.DS1) | set(data.DS2) | set(data.PACED_RECORDS) == set(data.ALL_RECORDS)
    assert set(data.SYMBOL_TO_AAMI.values()) == {"N", "S", "V", "F", "Q"}


# ---------- exported model ----------
@pytest.mark.skipif(not WEB_MODEL.exists(), reason="docs/data/model.json not exported")
def test_json_model_reproduces_training_output():
    fixture = json.loads((ROOT / "tests" / "fixtures" / "parity.json").read_text())
    model = JsonModel(json.loads(WEB_MODEL.read_text()))
    np.testing.assert_allclose(model.predict_proba(np.array(fixture["features"])), fixture["probabilities"], atol=1e-6)


@pytest.mark.skipif(not (data.MITDB_DIR / "100.hea").exists(), reason="MIT-BIH not downloaded")
def test_detector_on_real_record():
    rec = data.load_record("100")
    m = detect.match_beats(rec.beat_samples, detect.detect_r_peaks(filters.preprocess(rec.signal, rec.fs), rec.fs), rec.fs)
    assert m.sensitivity > 0.995 and m.ppv > 0.995
