"""Synthetic damage for old recordings (mono float32 in [-1, 1]).

degrade_audio(x, sr, rng, cfg) -> (damaged, applied_ops)
Damage types: reverb, wow/flutter, band-limiting, clipping, mains hum,
tape/vinyl hiss and vinyl crackle.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass
class AudioDamageConfig:
    severity: float = 1.0
    mains_hz: float = 50.0          # India = 50 Hz
    p_reverb: float = 0.30
    p_wow: float = 0.45
    p_band: float = 0.85
    p_clip: float = 0.30
    p_hum: float = 0.50
    p_hiss: float = 0.90
    p_crackle: float = 0.60


def _rms(a):
    return float(np.sqrt(np.mean(a ** 2) + 1e-12))


def _add_at_snr(x, noise, snr_db):
    noise = noise / (_rms(noise) + 1e-12) * _rms(x) * 10 ** (-snr_db / 20)
    return x + noise


def degrade_audio(x: np.ndarray, sr: int, rng: np.random.Generator,
                  cfg: AudioDamageConfig | None = None):
    cfg = cfg or AudioDamageConfig()
    s = float(cfg.severity)
    x = x.astype(np.float32).copy()
    n = len(x)
    ops: list[str] = []

    def hit(p):
        return rng.random() < min(1.0, p * (0.5 + 0.5 * s))

    if hit(cfg.p_reverb):
        t60 = rng.uniform(0.05, 0.35)
        L = int(t60 * sr)
        ir = rng.normal(0, 1, L) * np.exp(-6.9 * np.arange(L) / L)
        ir[0] = 4.0
        wet = signal.fftconvolve(x, ir / np.abs(ir).sum() * 3, mode="full")[:n]
        mix = rng.uniform(0.15, 0.4)
        x = ((1 - mix) * x + mix * wet).astype(np.float32)
        ops.append(f"reverb {t60:.2f}s")

    if hit(cfg.p_wow):   # speed wobble expressed as displacement, so no cumulative drift
        f = rng.uniform(0.5, 9.0)
        D = rng.uniform(0.5, 6.0) * s
        pos = np.arange(n) + D * np.sin(2 * np.pi * f * np.arange(n) / sr + rng.uniform(0, 6.28))
        x = np.interp(pos, np.arange(n), x).astype(np.float32)
        ops.append(f"wow/flutter {f:.1f} Hz")

    if hit(cfg.p_band):
        hi = rng.uniform(3000, min(7500, sr / 2 - 200))
        lo = rng.uniform(60, 220)
        sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        x = signal.sosfiltfilt(sos, x).astype(np.float32)
        ops.append(f"band-limited {lo:.0f}-{hi:.0f} Hz")

    if hit(cfg.p_clip):
        th = rng.uniform(0.45, 0.85)
        pk = np.abs(x).max() + 1e-9
        x = np.clip(x / pk * rng.uniform(1.1, 1.8), -th, th) / th * pk
        ops.append(f"clipping at {th:.2f}")

    if hit(cfg.p_hum):
        snr = rng.uniform(10, 30) - 4 * (s - 1)
        t = np.arange(n) / sr
        hum = sum((1 / k) * np.sin(2 * np.pi * cfg.mains_hz * k * t + rng.uniform(0, 6.28)) for k in range(1, 5))
        x = _add_at_snr(x, hum, snr).astype(np.float32)
        ops.append(f"{cfg.mains_hz:.0f} Hz hum (SNR {snr:.0f} dB)")

    if hit(cfg.p_hiss):
        snr = rng.uniform(5, 28) - 6 * (s - 1)
        white = rng.normal(0, 1, n)
        tilt = rng.uniform(0.0, 1.0)
        sos = signal.butter(1, 2000, btype="low", fs=sr, output="sos")
        hiss = (1 - tilt) * white + tilt * signal.sosfilt(sos, white) * 3
        x = _add_at_snr(x, hiss, snr).astype(np.float32)
        ops.append(f"tape hiss (SNR {snr:.0f} dB)")

    if hit(cfg.p_crackle):
        rate = rng.uniform(2, 8 + 12 * s)
        k = max(1, int(rate * n / sr))
        idx = rng.integers(0, n - 64, k)
        cr = np.zeros(n)
        peak = np.abs(x).max() + 1e-9
        for i in idx:
            L = int(rng.integers(8, 48))
            burst = rng.normal(0, 1, L) * np.exp(-np.arange(L) / (L / 4))
            cr[i:i + L] += burst * rng.uniform(0.05, 0.5) * peak
        x = (x + cr).astype(np.float32)
        ops.append(f"vinyl crackle {rate:.1f}/s")

    return np.clip(x, -1, 1).astype(np.float32), ops
