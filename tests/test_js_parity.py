"""The website's JavaScript must reproduce the Python pipeline (fixture made by `python -m ecglab export-web`)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "parity.json"

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not FIXTURE.exists(),
                                reason="needs Node.js and tests/fixtures/parity.json")


@pytest.fixture(scope="module")
def results():
    proc = subprocess.run(["node", str(ROOT / "tests" / "js" / "parity.mjs"), str(ROOT), str(FIXTURE)],
                          capture_output=True, text=True, check=True)
    return json.loads(FIXTURE.read_text()), json.loads(proc.stdout)


def test_filter_designs_match_scipy(results):
    py, js = results
    for p, j in zip(py["designs"], js["designs"]):
        assert p["n_sections"] == j["n_sections"], p
        np.testing.assert_allclose(j["filtfilt"], p["filtfilt"], atol=1e-8, err_msg=str(p["kind"]))
    np.testing.assert_allclose(js["notch_60"], py["notch_60"], atol=1e-12)


def test_conditioning_and_integration(results):
    py, js = results
    np.testing.assert_allclose(js["filtered_every_10"], py["filtered_every_10"], atol=1e-8)
    np.testing.assert_allclose(js["integrated_every_10"], py["integrated_every_10"], rtol=1e-7, atol=1e-9)


def test_same_beats_detected(results):
    py, js = results
    assert js["r_peaks"] == py["r_peaks"]
    assert js["beat_index"] == py["beat_index"]


def test_features_and_model(results):
    py, js = results
    np.testing.assert_allclose(js["features"], py["features"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(js["probabilities"], py["probabilities"], atol=1e-6)
    assert js["labels"] == py["labels"]


def test_report(results):
    py, js = results
    for key in ("mean_hr", "sinus_hr", "min_hr", "max_hr", "sdnn_ms", "rmssd_ms", "pnn50_pct", "irregularity"):
        assert js["metrics"][key] == pytest.approx(py["metrics"][key], rel=1e-9), key
    assert js["metrics"]["counts"] == py["metrics"]["counts"]
    for key in ("v_couplets", "v_runs", "s_runs", "bigeminy", "trigeminy", "pauses"):
        assert len(js["metrics"][key]) == len(py["metrics"][key]), key
    assert [(f["severity"], f["title"]) for f in js["findings"]] == [(f["severity"], f["title"]) for f in py["findings"]]
    assert js["verdict"] == py["verdict"]
