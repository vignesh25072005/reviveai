"""Placeholder 'clean' audio so the pipeline runs without any downloads.

Generates a raga-flavoured melody (Bhairav-like scale, drone + harmonics + vibrato)
with a simple tabla-like percussion loop. This is ONLY for smoke-testing and demos.
For real results train on real clean recordings (Common Voice, IndicTTS, Saraga...).
"""
from __future__ import annotations
import numpy as np

# Bhairav-like scale degrees in semitones above Sa
SCALE = np.array([0, 1, 4, 5, 7, 8, 11, 12])


def _note(f, dur, sr, rng):
    t = np.arange(int(dur * sr)) / sr
    vib = 1 + 0.004 * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t)
    ph = 2 * np.pi * np.cumsum(f * vib) / sr
    y = sum(a * np.sin(k * ph) for k, a in [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15), (5, 0.08)])
    env = np.minimum(1, t / 0.03) * np.exp(-t * rng.uniform(1.0, 3.0))
    return y * env


def _tabla_hit(sr, low=True):
    t = np.arange(int(0.25 * sr)) / sr
    f = (90 if low else 220) * (1 + 1.5 * np.exp(-t * 40))
    return np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * (14 if low else 25))


def synth_clean_audio(rng: np.random.Generator, seconds: float = 4.0, sr: int = 16000) -> np.ndarray:
    n = int(seconds * sr)
    y = np.zeros(n, np.float32)
    sa = rng.uniform(130, 220)
    t = np.arange(n) / sr
    y += 0.05 * (np.sin(2 * np.pi * sa * t) + 0.6 * np.sin(2 * np.pi * sa * 1.5 * t))   # tanpura-ish drone
    pos, deg = 0, int(rng.integers(0, len(SCALE)))
    while pos < n:
        deg = int(np.clip(deg + rng.integers(-2, 3), 0, len(SCALE) - 1))
        dur = float(rng.choice([0.2, 0.3, 0.45, 0.6]))
        f = sa * 2 ** (SCALE[deg] / 12) * 2
        if rng.random() > 0.25:                       # ~25% of beats are rests, like real phrasing
            seg = _note(f, dur + 0.25, sr, rng)[: n - pos]
            y[pos:pos + len(seg)] += 0.35 * seg.astype(np.float32)
        pos += int(dur * sr)
    beat = int(sr * rng.uniform(0.35, 0.6))
    for i, p in enumerate(range(0, n, beat)):
        h = _tabla_hit(sr, low=(i % 4 != 2))[: n - p]
        y[p:p + len(h)] += 0.3 * h.astype(np.float32)
    y /= max(np.abs(y).max(), 1e-6) / 0.8
    return y.astype(np.float32)


def synth_rock(rng: np.random.Generator, seconds: float = 12.0, sr: int = 48000, intro: float = 0.0) -> np.ndarray:
    """Stereo rock-style test music (sustained bass, kick, snare, hats, guitar-like chords) -> (n, 2).
    Used to check that denoising keeps the bass and the punch. `intro` seconds of silence come first."""
    from scipy import signal as sg
    n = int(seconds * sr)
    t = np.arange(n) / sr
    bpm = rng.uniform(100, 140); beat = 60 / bpm
    roots = 41.2 * 2 ** (np.array([0, 0, 5, 7, 0, 3, 5, 7]) / 12)            # E1 minor-ish bass line
    bass = np.zeros(n); kick = np.zeros(n); snare = np.zeros(n); hats = np.zeros(n); gtr = np.zeros(n)
    i = 0
    while i * beat < seconds:
        s0 = int(i * beat * sr); f = roots[(i // 2) % len(roots)]
        L = min(int(beat * 1.9 * sr), n - s0)
        tt = np.arange(L) / sr
        note = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt) + 0.15 * np.sin(6 * np.pi * f * tt))
        bass[s0:s0 + L] += 0.38 * note * np.minimum(1, tt / 0.01) * np.exp(-tt * 0.7)
        if i % 2 == 0:
            Lk = min(int(0.2 * sr), n - s0); tk = np.arange(Lk) / sr
            kick[s0:s0 + Lk] += 0.55 * np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-tk * 30)) / sr) * np.exp(-tk * 18)
        else:
            Ls = min(int(0.16 * sr), n - s0); ts = np.arange(Ls) / sr
            snare[s0:s0 + Ls] += 0.3 * rng.normal(0, 1, Ls) * np.exp(-ts * 22)
        for h in (0, 0.5):
            s1 = int((i + h) * beat * sr); Lh = min(int(0.05 * sr), n - s1)
            if Lh > 0:
                hats[s1:s1 + Lh] += 0.07 * rng.normal(0, 1, Lh) * np.exp(-np.arange(Lh) / sr * 90)
        i += 1
    chord = roots[0] * 4 * 2 ** (np.array([0, 3, 7, 12]) / 12)
    for k, fc in enumerate(chord):
        gtr += sum(np.sin(2 * np.pi * fc * h * t + rng.uniform(0, 6.28)) / h for h in range(1, 12))
    gtr = sg.sosfilt(sg.butter(2, 4500, "low", fs=sr, output="sos"), gtr) * 0.045
    gtr *= 0.7 + 0.3 * np.sin(2 * np.pi * t / (beat * 4))
    sn = sg.sosfilt(sg.butter(2, [1200, 7000], "band", fs=sr, output="sos"), snare)
    ht = sg.sosfilt(sg.butter(2, 7000, "high", fs=sr, output="sos"), hats)
    mid = bass + kick + sn + ht
    left = mid + gtr * 1.0 + 0.3 * np.roll(gtr, 40)
    right = mid + gtr * 0.8 + 0.3 * np.roll(gtr, 90)
    out = np.stack([left, right], 1)
    out /= np.abs(out).max() / 0.9
    if intro > 0:
        out[: int(intro * sr)] = 0
    return out.astype(np.float32)
