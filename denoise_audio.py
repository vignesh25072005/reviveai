"""Audio denoising that keeps the music: bass, punch and stereo image.

How it works
  1. Noise profile.  If the recording has a quiet passage (an intro, a gap between songs) its hiss is measured
     directly. Otherwise the hiss is tracked with minimum statistics. Either way the estimate is smoothed over
     frequency so a sustained bass note or a drone is never mistaken for noise.
  2. Wiener filter (decision-directed) removes the hiss. The window length scales with the sample rate, so a
     48 kHz recording gets the frequency resolution it needs to separate bass notes from hiss.
  3. Protection.  Gain below ~150 Hz can never drop more than 3 dB (hiss lives in the highs), and the maximum
     cut elsewhere is limited by `strength`, which prevents the dull, underwater sound of over-processing.
  4. Stereo-linked.  One gain curve, computed from both channels together, is applied to both, so the stereo
     image does not wander.

denoise_audio(x, sr, strength=0.5, hum="auto")  x: (n,) or (n, channels)  ->  (y same shape, info)
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d, median_filter

from .restore_audio_classic import detect_hum, remove_hum, to_mono_float


def fft_size(sr: int) -> int:
    return int(np.clip(1 << int(round(np.log2(0.032 * sr))), 256, 4096))     # 16 kHz: 512, 44.1/48 kHz: 2048


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def _freq_smooth(psd: np.ndarray, bin_hz: float, width_hz: float) -> np.ndarray:
    return median_filter(psd, size=(_odd(width_hz / bin_hz), 1), mode="nearest")


def _minstat_psd(power: np.ndarray, hop_s: float, bin_hz: float, win_s: float = 1.5, bias: float = 1.6) -> np.ndarray:
    smooth = uniform_filter1d(power, size=5, axis=1, mode="nearest")
    w = max(5, int(win_s / hop_s))
    mn = minimum_filter1d(smooth, size=w, axis=1, mode="nearest")
    est = bias * uniform_filter1d(mn, size=max(3, w // 2), axis=1, mode="nearest")
    return _freq_smooth(est, bin_hz, 500.0)


def estimate_noise_psd(power: np.ndarray, hop_s: float, bin_hz: float):
    """-> (noise PSD per bin and frame, how it was found: 'quiet passage' or 'tracked')."""
    energy = power.sum(0)
    k = max(3, int(0.08 * len(energy)))
    order = np.argsort(energy)
    quiet = order[:k]
    gap_db = 10 * np.log10((np.median(energy) + 1e-20) / (energy[quiet].mean() + 1e-20))
    if gap_db >= 10.0:                                      # a genuinely quiet passage exists: measure the hiss directly
        prof = _freq_smooth(power[:, quiet].mean(1, keepdims=True), bin_hz, 250.0)
        return np.repeat(prof, power.shape[1], 1), "quiet passage"
    return _minstat_psd(power, hop_s, bin_hz), "tracked"


def _floor_curve(freqs: np.ndarray, strength: float) -> np.ndarray:
    """Minimum allowed gain per bin: 3 dB at the bass, up to `strength`-dependent depth in the mids/highs."""
    deep = 10 ** (-(9 + 15 * strength) / 20)               # -9 dB (gentle) .. -24 dB (strong)
    shallow = 10 ** (-3 / 20)
    t = np.clip((freqs - 150.0) / 450.0, 0, 1)
    t = t * t * (3 - 2 * t)                                  # smoothstep from 150 Hz to 600 Hz
    return shallow + (deep - shallow) * t


def wiener_multi(x2: np.ndarray, sr: int, strength: float = 0.5, progress=None):
    """x2: (n, ch). Returns (y2, input_snr_db, how_noise_was_found)."""
    n_fft = fft_size(sr); hop = n_fft // 4
    n = x2.shape[0]
    specs = [signal.stft(x2[:, c], sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)[2] for c in range(x2.shape[1])]
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    P = np.mean([np.abs(Y) ** 2 for Y in specs], 0)         # shared power -> one gain for all channels
    bin_hz = sr / n_fft
    N, how = estimate_noise_psd(P, hop / sr, bin_hz)
    N = np.maximum(N, 1e-14)
    alpha = 0.95 + 0.04 * strength
    over = 1.0 + 0.5 * strength
    gmin = _floor_curve(freqs, strength)[:, None]
    G = np.ones_like(P); prev = np.ones(P.shape[0])
    for t in range(P.shape[1]):
        gamma = P[:, t] / (over * N[:, t])
        xi = alpha * prev + (1 - alpha) * np.maximum(gamma - 1, 0)
        g = np.maximum(xi / (1 + xi), gmin[:, 0])
        G[:, t] = g
        prev = (g ** 2) * gamma
        if progress and t % 300 == 0:
            progress(t / P.shape[1])
    G = uniform_filter1d(G, size=3, axis=0, mode="nearest")
    G = uniform_filter1d(G, size=3, axis=1, mode="nearest")
    out = []
    for c, Y in enumerate(specs):
        _, y = signal.istft(Y * G, sr, nperseg=n_fft, noverlap=n_fft - hop)
        y = y[:n]
        out.append(np.pad(y, (0, n - len(y))) if len(y) < n else y)
    sig_e = max(float(np.sum(np.maximum(P - N, 0))), 1e-20)
    snr = 10 * np.log10(sig_e / max(float(np.sum(np.minimum(P, N))), 1e-20))
    return np.stack(out, 1).astype(np.float32), float(snr), how


def wiener_denoise(x: np.ndarray, sr: int, strength: float = 0.5, n_fft: int | None = None, progress=None):
    """Mono convenience wrapper (kept for the benchmark scripts). Returns (denoised, snr_db)."""
    y, snr, _ = wiener_multi(np.asarray(x, np.float32)[:, None], sr, strength, progress)
    return y[:, 0], snr


def denoise_audio(x, sr: int, strength: float = 0.5, hum: str | float | None = "auto", rumble_hz: float = 20.0,
                  progress=None):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1)
    x = x.astype(np.float32)
    was_1d = x.ndim == 1
    x2 = x[:, None] if was_1d else x
    steps = []
    if rumble_hz:
        sos = signal.butter(2, rumble_hz, "highpass", fs=sr, output="sos")
        x2 = signal.sosfiltfilt(sos, x2, axis=0).astype(np.float32)
        steps.append(f"rumble filter {rumble_hz:.0f} Hz")
    if hum:
        m = detect_hum(x2.mean(1), sr) if hum == "auto" else float(hum)
        if m:
            x2 = np.stack([remove_hum(x2[:, c], sr, m) for c in range(x2.shape[1])], 1)
            steps.append(f"{m:.0f} Hz hum removed")
    y2, snr, how = wiener_multi(x2, sr, float(strength), progress)
    steps.append(f"hiss measured from a {how}, Wiener denoise (strength {strength:.2f})")
    pk = float(np.abs(y2).max())
    if pk > 0.999:
        y2 = y2 / pk * 0.98
    y = y2[:, 0] if was_1d else y2
    return y.astype(np.float32), {"steps": steps, "input_snr_db": round(snr, 1), "noise_source": how}


def noise_floor_dbfs(x, sr: int) -> float:
    """Level of the hiss in the QUIETEST passages (lowest 8% of frames), in dBFS (RMS).
    Exact for steady white noise. In music with no quiet gap it reads high, because the quietest frames still
    contain music, so compare before/after on the same file rather than treating it as an absolute figure."""
    x = to_mono_float(x)
    n_fft = fft_size(sr); hop = n_fft // 4
    _, _, Y = signal.stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    P = np.abs(Y) ** 2
    energy = P.sum(0)
    quiet = np.argsort(energy)[: max(3, int(0.08 * len(energy)))]
    power = float(P[:, quiet].mean()) * n_fft / 1.5          # hann window: |Y|^2 = v * 1.5 / n_fft for noise variance v
    return 10 * np.log10(max(power, 1e-14))
