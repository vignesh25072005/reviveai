"""Synthetic noise of the kinds found in real photographs (for testing and benchmarking)."""
from __future__ import annotations
import numpy as np
import cv2


def _clip(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def gaussian(img, rng, sigma):                     # sensor / high-ISO grain
    return _clip(img + rng.normal(0, sigma, img.shape))


def chroma(img, rng, sigma):                       # colour blotches, brightness untouched
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    for c in (1, 2):
        ycc[..., c] += cv2.GaussianBlur(rng.normal(0, sigma, ycc.shape[:2]).astype(np.float32), (0, 0), 1.2) * 2.0
    return cv2.cvtColor(_clip(ycc), cv2.COLOR_YCrCb2RGB)


def salt_pepper(img, rng, density):                # classic 1-pixel impulses, independent per channel
    out = img.copy()
    r = rng.random(img.shape)
    out[r < density / 2] = 0
    out[r > 1 - density / 2] = 255
    return out


def snow(img, rng, density, size=2):               # white specks a few pixels wide, like resampled "snow"
    out = img.copy()
    h, w = img.shape[:2]
    n = int(density * h * w / (size * size))
    ys, xs = rng.integers(0, h - size, n), rng.integers(0, w - size, n)
    for y, x in zip(ys, xs):
        out[y:y + size, x:x + size] = 255 if rng.random() < 0.8 else 0
    return out


def multiplicative(img, rng, level):               # radar / ultrasound-style speckle
    return _clip(img.astype(np.float32) * (1 + rng.normal(0, level, img.shape)))


def mixed(img, rng, sigma, density):
    return salt_pepper(gaussian(img, rng, sigma), rng, density)
