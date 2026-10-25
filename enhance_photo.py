"""Old-photo restoration (classical image processing, no model download).

For scans of faded, washed-out, soft or lightly marked prints. Steps:
  1. repair   find thin scratches, hairs and dust specks and inpaint them
  2. clean    gentle grain / scan-noise removal (kept light so faces stay natural)
  3. tone     recover local contrast in shadows and blown highlights (CLAHE on lightness), optional colour clean-up
  4. sharpen  unsharp mask on lightness only (no colour fringes)

restore_old_photo(img_rgb, contrast=0.6, repair=0.5, sharpen=0.5, clean=0.3, tone="keep")
    -> (restored_rgb_uint8, info)
tone: "keep" keeps the print's colour, "neutral" removes most of the colour cast, "bw" makes it black and white.
It cannot invent detail that was never captured; it makes what is there easier to see.
"""
from __future__ import annotations
import time
import numpy as np
import cv2

from .dust import find_marks, repair_marks


def _levels(L: np.ndarray, amount: float) -> np.ndarray:
    lo, hi = np.percentile(L, (0.5, 99.5))
    stretched = np.clip((L - lo) / max(hi - lo, 30.0) * 255.0, 0, 255)
    return (1 - amount) * L + amount * stretched


def restore_old_photo(img: np.ndarray, contrast: float = 0.6, repair: float = 0.5, sharpen: float = 0.5,
                      clean: float = 0.5, tone: str = "keep"):
    t0 = time.time()
    img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    h, w = img.shape[:2]
    cover = 0.0
    if repair > 0:
        mask = find_marks(img, sensitivity=float(np.clip(repair, 0, 1)))
        cover = 100.0 * float((mask > 0).mean())
        if mask.any():
            img = repair_marks(img, mask)
    bgr = img[..., ::-1].copy()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    if clean > 0:      # denoise BEFORE boosting contrast, otherwise the boost amplifies shadow grain
        cl = float(np.clip(clean, 0, 1))
        lab[..., 0] = cv2.fastNlMeansDenoising(lab[..., 0], None, 3.0 + 9.0 * cl, 5, 15)
        for ch in (1, 2):
            lab[..., ch] = cv2.fastNlMeansDenoising(lab[..., ch], None, 2.0 + 4.0 * cl, 5, 15)
    L, a, b = [lab[..., i].astype(np.float32) for i in range(3)]
    c = float(np.clip(contrast, 0, 1))
    if c > 0:
        L = _levels(L, 0.5 + 0.4 * c)
        clahe = cv2.createCLAHE(clipLimit=1.0 + 2.2 * c, tileGridSize=(8, 8))
        Lc = clahe.apply(np.clip(L, 0, 255).astype(np.uint8)).astype(np.float32)
        L = (1 - 0.75 * c) * L + 0.75 * c * Lc
    if tone == "bw":
        a[:] = 128; b[:] = 128
    elif tone == "neutral":
        a = 128 + (a - 128) * 0.15; b = 128 + (b - 128) * 0.15
    s = float(np.clip(sharpen, 0, 1))
    if s > 0:
        blur = cv2.GaussianBlur(L, (0, 0), 1.6)
        detail = L - blur
        detail = np.where(np.abs(detail) < 3.5, 0, detail)           # leave faint grain alone
        L = L + (0.5 + 1.5 * s) * detail
    lab = np.stack([np.clip(L, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255)], -1).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out, {"mode": "restore", "damage_repaired_pct": round(cover, 1), "seconds": round(time.time() - t0, 2),
                 "size": [w, h], "tone": tone}
