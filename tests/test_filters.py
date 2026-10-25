import numpy as np
import pytest

from ecglab import filters

FS = 360.0
t = np.arange(int(20 * FS)) / FS


def amplitude(x, f):
    """Amplitude of the f-Hz component (least squares on sin/cos), ignoring the edges."""
    sl = slice(int(2 * FS), -int(2 * FS))
    A = np.column_stack([np.sin(2 * np.pi * f * t[sl]), np.cos(2 * np.pi * f * t[sl])])
    coef, *_ = np.linalg.lstsq(A, x[sl], rcond=None)
    return float(np.hypot(*coef))


def test_sections_per_order():
    assert len(filters.butter_sos("lowpass", 40, FS, 4)) == 2
    assert len(filters.butter_sos("highpass", 0.5, FS, 3)) == 2
    assert len(filters.TRAINING_FILTER.sos(FS)) == 3


def test_invalid_cutoff():
    with pytest.raises(ValueError):
        filters.butter_sos("lowpass", 200, FS)
    with pytest.raises(ValueError):
        filters.butter_sos("bandstop", 10, FS)


def test_highpass_removes_baseline_keeps_ecg_band():
    x = np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 10 * t)
    y = filters.apply_sos(x, filters.butter_sos("highpass", 0.5, FS, 2))
    assert amplitude(y, 0.1) < 0.01
    assert amplitude(y, 10) == pytest.approx(1.0, abs=0.01)


def test_notch_kills_mains_only():
    x = np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 10 * t)
    y = filters.apply_sos(x, filters.notch_sos(60, FS, 30))
    assert amplitude(y, 60) < 10 ** (-40 / 20)
    assert amplitude(y, 10) == pytest.approx(1.0, abs=0.01)


def test_zero_phase_has_no_delay_but_causal_does(ecg):
    x, _ = ecg
    cfg = filters.FilterConfig(highpass_hz=0.5, lowpass_hz=20)
    lag = lambda y: int(np.argmax(np.correlate(y, x, "full")) - (len(x) - 1))
    assert lag(filters.preprocess(x, FS, cfg)) == 0
    causal = filters.FilterConfig(highpass_hz=0.5, lowpass_hz=20, zero_phase=False)
    assert lag(filters.preprocess(x, FS, causal)) > 0


def test_median_baseline_tracks_wander(ecg):
    x, _ = ecg
    wander = 0.8 * np.sin(2 * np.pi * 0.2 * np.arange(len(x)) / FS)
    cleaned = filters.remove_baseline_median(x + wander, FS)
    resid = cleaned - filters.remove_baseline_median(x, FS)
    assert np.std(resid) < 0.1 * np.std(wander)


def test_add_noise_hits_target_snr(ecg):
    x, _ = ecg
    noise = np.random.default_rng(1).normal(size=len(x))
    noisy = filters.add_noise(x, noise, 6.0)
    assert filters.snr_db(x, noisy) == pytest.approx(6.0, abs=0.05)


def test_frequency_response_cutoff_is_3db_per_pass():
    sos = filters.butter_sos("lowpass", 40, FS, 4)
    f, db = filters.frequency_response(sos, FS, zero_phase=False)
    assert np.interp(40, f, db) == pytest.approx(-3.01, abs=0.05)
    f, db = filters.frequency_response(sos, FS, zero_phase=True)
    assert np.interp(40, f, db) == pytest.approx(-6.02, abs=0.1)


def test_resample_linear():
    x = np.sin(2 * np.pi * 2 * np.arange(500) / 250)
    y = filters.resample_linear(x, 250, 360)
    assert len(y) == int(np.floor(499 * 360 / 250)) + 1
    assert np.max(np.abs(y - np.sin(2 * np.pi * 2 * np.arange(len(y)) / 360))) < 0.01
