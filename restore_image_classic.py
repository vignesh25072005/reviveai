"""Classical (no GPU, no weights) photo restoration.

Pipeline: detect scratches/dust -> inpaint -> denoise -> restore tone/colour -> light sharpen.
It is the always-available engine in the UI and the baseline the diffusion
model must beat in `scripts/evaluate.py`.
"""
from __future__ import annotations
import numpy as np
import cv2

DEFAULTS = dict(sensitivity=0.5, denoise=10.0, tone=0.0, sharpen=0.0)


def detect_damage_mask(img_rgb: np.ndarray, sensitivity: float = 0.5) -> np.ndarray:
    """Thin bright/dark structures (scratches, hairs, dust). Returns uint8 0/255."""
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
    resp = np.zeros_like(g)
    for k in [(15, 1), (1, 15), (7, 7)]:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, k)
        white = g - cv2.morphologyEx(g, cv2.MORPH_OPEN, se)
        black = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se) - g
        resp = np.maximum(resp, np.maximum(white, black))
    thr = 110.0 - 70.0 * float(np.clip(sensitivity, 0, 1))         # 110 .. 40 grey levels (0.5 -> 75)
    m = (resp > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            keep[lab == i] = 1
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    return keep * 255


def restore_tone(img_rgb: np.ndarray, amount: float = 0.8) -> np.ndarray:
    """Undo fading and colour cast: per-channel percentile stretch blended with the input."""
    x = img_rgb.astype(np.float32)
    lo = np.percentile(x, 1, axis=(0, 1))
    hi = np.percentile(x, 99, axis=(0, 1))
    stretched = np.clip((x - lo) / np.maximum(hi - lo, 20.0) * 255, 0, 255)
    return np.clip((1 - amount) * x + amount * stretched, 0, 255).astype(np.uint8)


def restore_image(img_rgb: np.ndarray, user_mask: np.ndarray | None = None,
                  sensitivity: float = DEFAULTS["sensitivity"], denoise: float = DEFAULTS["denoise"],
                  tone: float = DEFAULTS["tone"], sharpen: float = DEFAULTS["sharpen"]):
    """Returns (restored_rgb, damage_mask). `user_mask`: uint8, >0 where the user painted damage."""
    img = np.ascontiguousarray(img_rgb[..., :3]).astype(np.uint8)
    mask = detect_damage_mask(img, sensitivity) if sensitivity > 0 else np.zeros(img.shape[:2], np.uint8)
    if user_mask is not None:
        um = (user_mask > 0).astype(np.uint8) * 255
        mask = np.maximum(mask, cv2.dilate(um, np.ones((3, 3), np.uint8)))
    bgr = img[..., ::-1].copy()
    if mask.any():
        bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    if denoise > 0:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, denoise, denoise, 5, 15)
    out = bgr[..., ::-1]
    if tone > 0:
        out = restore_tone(out, tone)
    if sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1 + sharpen, blur, -sharpen, 0)
    return np.clip(out, 0, 255).astype(np.uint8), mask
