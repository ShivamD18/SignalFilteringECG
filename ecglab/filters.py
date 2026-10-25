"""Digital filters for ECG conditioning.

=================  ====================  =========================  ============================
Noise source       Typical band          Physiological cause        Filter used here
=================  ====================  =========================  ============================
Baseline wander    below ~0.5 Hz         breathing, sweat, drift    high-pass or median baseline
Power-line hum     50 or 60 Hz           mains coupling             IIR notch
Muscle (EMG)       ~20-500 Hz, broad     skeletal muscle activity   low-pass
Electrode motion   ~1-10 Hz              skin stretch at electrode  hard: overlaps the ECG itself
=================  ====================  =========================  ============================

Every filter is built as cascaded second-order sections (SOS), which are
numerically robust, and can be applied either causally (like a bedside monitor,
introducing phase delay) or forward-backward ("zero-phase", like offline
analysis software). The browser version in ``docs/js/dsp.js`` implements the
same maths so the website reproduces these results exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, signal


def butter_sos(kind: str, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """Butterworth high-pass or low-pass filter as second-order sections."""
    if kind not in ("highpass", "lowpass"):
        raise ValueError("kind must be 'highpass' or 'lowpass'")
    if not 0 < cutoff < fs / 2:
        raise ValueError(f"cutoff {cutoff} Hz must lie between 0 and Nyquist ({fs / 2} Hz)")
    return signal.butter(order, cutoff, btype=kind, fs=fs, output="sos")


def notch_sos(f0: float, fs: float, q: float = 30.0) -> np.ndarray:
    """Second-order IIR notch at ``f0`` Hz; -3 dB bandwidth is ``f0 / q``."""
    b, a = signal.iirnotch(f0, q, fs)
    return np.hstack([b, a])[None, :]


@dataclass(frozen=True)
class FilterConfig:
    """A complete conditioning chain: high-pass -> low-pass -> notch."""

    highpass_hz: float | None = 0.5
    highpass_order: int = 2
    lowpass_hz: float | None = 40.0
    lowpass_order: int = 4
    notch_hz: float | None = None
    notch_q: float = 30.0
    zero_phase: bool = True

    def sos(self, fs: float) -> np.ndarray:
        sections = []
        if self.highpass_hz:
            sections.append(butter_sos("highpass", self.highpass_hz, fs, self.highpass_order))
        if self.lowpass_hz and self.lowpass_hz < fs / 2:
            sections.append(butter_sos("lowpass", self.lowpass_hz, fs, self.lowpass_order))
        if self.notch_hz and self.notch_hz < fs / 2:
            sections.append(notch_sos(self.notch_hz, fs, self.notch_q))
        if not sections:
            return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        return np.vstack(sections)

    def to_dict(self) -> dict:
        return asdict(self)


# The conditioning used to train the classifier ("monitoring-mode" bandwidth).
# 0.5 Hz is acceptable for the ST segment only because the filter is zero-phase
# (AHA/ACCF/HRS recommendations, Kligfield et al., Circulation 2007).
TRAINING_FILTER = FilterConfig()


def apply_sos(x: np.ndarray, sos: np.ndarray, zero_phase: bool = True) -> np.ndarray:
    """Filter ``x``. Causal mode starts from the steady state of ``x[0]`` to avoid a start-up jump."""
    x = np.asarray(x, dtype=np.float64)
    if zero_phase:
        return signal.sosfiltfilt(sos, x)
    zi = signal.sosfilt_zi(sos) * x[0]
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


def preprocess(x: np.ndarray, fs: float, config: FilterConfig = TRAINING_FILTER) -> np.ndarray:
    """Run ``x`` through the filter chain described by ``config``."""
    return apply_sos(x, config.sos(fs), config.zero_phase)


def median_baseline(x: np.ndarray, fs: float, windows: tuple[float, float] = (0.2, 0.6)) -> np.ndarray:
    """Estimate baseline wander with two cascaded median filters (de Chazal et al., 2004).

    The 200 ms median removes the QRS complex and P wave, the 600 ms median
    removes the T wave; what is left is the slowly varying baseline. Unlike a
    linear high-pass, this does not distort the ST segment.
    """
    base = np.asarray(x, dtype=np.float64)
    for w in windows:
        size = int(round(w * fs)) | 1  # odd length keeps it centred
        base = ndimage.median_filter(base, size=size, mode="nearest")
    return base


def remove_baseline_median(x: np.ndarray, fs: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) - median_baseline(x, fs)


def frequency_response(sos: np.ndarray, fs: float, n: int = 2048, zero_phase: bool = True):
    """Return (frequencies in Hz, magnitude in dB). Zero-phase filtering squares the magnitude."""
    f, h = signal.sosfreqz(sos, worN=n, fs=fs)
    mag = np.abs(h) ** (2 if zero_phase else 1)
    return f, 20 * np.log10(np.maximum(mag, 1e-12))


def psd(x: np.ndarray, fs: float, nperseg: int = 2048):
    """Welch power spectral density (mV^2/Hz)."""
    return signal.welch(x, fs=fs, nperseg=min(nperseg, len(x)))


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """SNR of ``test`` relative to the reference ``clean`` signal (both mean-removed)."""
    clean = np.asarray(clean, dtype=np.float64) - np.mean(clean)
    err = np.asarray(test, dtype=np.float64) - np.mean(test) - clean
    return float(10 * np.log10(np.sum(clean**2) / max(np.sum(err**2), 1e-20)))


def add_noise(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Add ``noise`` scaled so the result has the requested SNR (power ratio, as in nstdb)."""
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise[: len(clean)], dtype=np.float64)
    noise = noise - noise.mean()
    p_sig = np.mean((clean - clean.mean()) ** 2)
    p_noise = np.mean(noise**2)
    gain = np.sqrt(p_sig / (p_noise * 10 ** (target_snr_db / 10)))
    return clean + gain * noise


def resample_linear(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Linear-interpolation resampling. Low-pass first when downsampling (preprocess does)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    n_out = int(np.floor((len(x) - 1) * fs_out / fs_in)) + 1
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, np.arange(len(x)) / fs_in, x)
