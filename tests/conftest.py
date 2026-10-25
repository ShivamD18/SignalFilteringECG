import numpy as np
import pytest

FS = 360.0

# (offset s, amplitude mV, width s) of the Gaussian bumps that make up one synthetic beat
NORMAL_BEAT = [(-0.20, 0.15, 0.025), (-0.03, -0.10, 0.010), (0.0, 1.20, 0.012), (0.03, -0.25, 0.010), (0.25, 0.30, 0.050)]
WIDE_BEAT = [(-0.02, -0.40, 0.030), (0.03, 1.40, 0.040), (0.30, -0.45, 0.060)]  # PVC-like: no P, wide QRS


def synthetic_ecg(duration=30.0, hr=72.0, premature=(), fs=FS):
    """Gaussian-sum ECG. ``premature`` lists beat indices to replace by an early, wide beat."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    rr = 60.0 / hr
    beats = list(np.arange(0.6, duration - 0.6, rr))
    for i in premature:
        beats[i] -= 0.35 * rr
    x = np.zeros(n)
    for i, b in enumerate(beats):
        for off, amp, w in (WIDE_BEAT if i in premature else NORMAL_BEAT):
            x += amp * np.exp(-((t - b - off) ** 2) / (2 * w**2))
    return x, (np.array(beats) * fs).round().astype(int)


@pytest.fixture
def ecg():
    return synthetic_ecg()
