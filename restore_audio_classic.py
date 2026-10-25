"""Classical audio restoration (numpy + scipy only).

Steps: rumble filter -> mains hum notch (auto-detected) -> click removal -> spectral-gate denoise.
Cannot fix wow/flutter, clipping or lost high frequencies; those need the diffusion model.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    return x.astype(np.float32)


def detect_hum(x, sr, mains=(50.0, 60.0)):
    """Return the mains frequency (50/60) if a strong narrow peak is present, else None."""
    f, p = signal.welch(x, sr, nperseg=min(len(x), 8192))
    best, ratio_best = None, 4.0
    for m in mains:
        band = (f > m - 1.5) & (f < m + 1.5)
        side = ((f > m - 12) & (f < m - 4)) | ((f > m + 4) & (f < m + 12))
        if band.any() and side.any():
            r = p[band].max() / (np.median(p[side]) + 1e-12)
            if r > ratio_best:
                best, ratio_best = m, r
    return best


def remove_hum(x, sr, mains, harmonics=2, q=40.0):
    for k in range(1, harmonics + 1):
        fk = mains * k
        if fk < sr / 2 - 50:
            b, a = signal.iirnotch(fk, q, fs=sr)
            x = signal.filtfilt(b, a, x)
    return x.astype(np.float32)


def remove_clicks(x, k=7.0, grow=3, win=801, max_frac=0.02):
    """Find impulses (residual to a median filter, judged against the LOCAL residual level)
    and interpolate across them. Skips itself if it would touch more than `max_frac` of samples."""
    res = x - signal.medfilt(x, 5)
    local = uniform_filter1d(np.abs(res), win, mode="nearest") + 1e-6
    bad = np.abs(res) > k * local * 2.0
    bad = np.convolve(bad.astype(float), np.ones(2 * grow + 1), "same") > 0
    if not bad.any() or bad.mean() > max_frac:
        return x
    idx = np.arange(len(x))
    y = x.copy()
    y[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return y.astype(np.float32)


def spectral_gate(x, sr, strength=0.4, n_fft=512):
    """Spectral subtraction. Noise profile = average of the quietest 10% of frames."""
    if strength <= 0:
        return x
    hop = n_fft // 4
    f, t, X = signal.stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(X)
    energy = (mag ** 2).sum(0)
    quiet = energy <= np.percentile(energy, 10)
    noise = mag[:, quiet].mean(1, keepdims=True)
    over = 0.5 + 2.0 * strength
    gain = np.clip(1.0 - over * noise / (mag + 1e-9), 0.15 + 0.5 * (1 - strength), 1.0)
    gain = signal.convolve2d(gain, np.ones((1, 3)) / 3, mode="same", boundary="symm")
    _, y = signal.istft(X * gain, sr, nperseg=n_fft, noverlap=n_fft - hop)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def restore_audio(x, sr, hum="auto", clicks=True, denoise=0.3, rumble_hz=40.0):
    """Returns (restored, list_of_steps_applied)."""
    x = to_mono_float(x)
    steps = []
    if rumble_hz:
        x = signal.sosfiltfilt(signal.butter(2, rumble_hz, "highpass", fs=sr, output="sos"), x).astype(np.float32)
        steps.append(f"rumble filter {rumble_hz:.0f} Hz")
    if hum:
        m = detect_hum(x, sr) if hum == "auto" else float(hum)
        if m:
            x = remove_hum(x, sr, m)
            steps.append(f"hum notch {m:.0f} Hz + harmonics")
    if clicks:
        x = remove_clicks(x)
        steps.append("click removal")
    if denoise > 0:
        x = spectral_gate(x, sr, denoise)
        steps.append(f"noise reduction ({denoise:.1f})")
    pk = np.abs(x).max()
    if pk > 1.0:
        x = x / pk * 0.98
    return x.astype(np.float32), steps
