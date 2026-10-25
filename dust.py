"""Scale-aware dust, speck and scratch removal for scanned prints.

Why this exists: dust on a high-resolution scan is many pixels wide, so a detector tuned for tiny specks misses it.
Everything here scales with the image size (reference: a 700 px short side).

How a mark is recognised
  1. A median-filtered copy of the photo shows what the picture looks like WITHOUT specks smaller than half the filter.
  2. Pixels far brighter or darker than that copy are candidates.
  3. A candidate group is kept only if it is small enough (or thin and long, like a scratch) AND sits on a calm
     background. Real detail (eyes, teeth, collar edges, hair) has busy surroundings and is left alone.
  4. Kept marks are filled from the median copy, blended at the edges, and large ones are inpainted.

find_marks(img_rgb, sensitivity) -> uint8 mask (255 = mark)
repair_marks(img_rgb, mask)      -> repaired rgb
"""
from __future__ import annotations
import numpy as np
import cv2


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def find_marks(img: np.ndarray, sensitivity: float = 0.5, max_components: int = 6000, max_side: int = 1800) -> np.ndarray:
    """Marks mask for any image size. Large scans are analysed on a downscaled copy (fast), then the mask is scaled back."""
    h0, w0 = img.shape[:2]
    if max(h0, w0) > max_side:
        f = max_side / max(h0, w0)
        small = cv2.resize(img, (int(w0 * f), int(h0 * f)), interpolation=cv2.INTER_AREA)
        m = _find_marks(small, sensitivity, max_components)
        m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
        r = max(1, int(round(0.8 / f)))
        return cv2.dilate((m > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) * 255
    return _find_marks(img, sensitivity, max_components)


def _find_marks(img: np.ndarray, sensitivity: float, max_components: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = min(h, w)
    s = max(1.0, m / 700.0)
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    if sens <= 0:
        return np.zeros((h, w), np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thr = 85.0 - 55.0 * sens                                    # 85 (cautious) .. 30 (thorough) grey levels

    # 1) multi-scale bright / dark "bumps" narrower than d: sizes grow with the image
    resp = np.zeros(g.shape, np.uint8)
    for frac in (0.004, 0.008, 0.016, 0.03):
        d = _odd(max(3, frac * m))
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        resp = np.maximum(resp, np.maximum(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se), cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)))
    cand = cv2.dilate((resp > thr).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if not cand.any():
        return np.zeros((h, w), np.uint8)

    # 2) judge every candidate group by its size/shape and by the calm ring of pixels around it
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    a_max = (0.035 * m) ** 2
    long_max = 0.7 * max(h, w)
    gf = g.astype(np.float32)
    order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1 if n > max_components else range(1, n)
    keep = np.zeros(n, bool)
    for i in list(order)[:max_components]:
        x, y, bw, bh, area = st[i]
        longest, shortest = max(bw, bh), min(bw, bh)
        blob = area <= a_max and longest <= 0.06 * m
        scratch = shortest <= max(5, 0.008 * m) and longest <= long_max and area >= 6 and longest >= 3 * shortest
        if not (blob or scratch):
            continue
        pad = int(max(4, 0.7 * longest)) if blob else int(max(4, 2 * shortest))
        y0, y1, x0, x1 = max(0, y - pad), min(h, y + bh + pad), max(0, x - pad), min(w, x + bw + pad)
        box_lab = lab[y0:y1, x0:x1]
        inside = box_lab == i
        near = cv2.dilate(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ring = ~near & (box_lab == 0)                           # surroundings that are not any other candidate
        if ring.sum() < 12:
            continue
        gbox = gf[y0:y1, x0:x1]
        ring_std = float(gbox[ring].std())
        ring_mean = float(gbox[ring].mean())
        diff = abs(float(gbox[inside].mean()) - ring_mean)
        if ring_std < (9.0 + 16.0 * sens) and diff > 0.55 * thr:
            keep[i] = True
    mask = keep[lab].astype(np.uint8)
    if not mask.any():
        return np.zeros((h, w), np.uint8)
    r = max(1, int(round(1.5 * s)))
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) * 255


def repair_marks(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill marks from the local median, blended at the edges; inpaint anything the median cannot cover."""
    if not mask.any():
        return img
    h, w = img.shape[:2]
    s = max(1.0, min(h, w) / 700.0)
    k = _odd(max(7, 15 * s))
    bg = cv2.medianBlur(img, min(k, 255))
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), max(0.8, 0.8 * s))[..., None]
    soft = np.maximum(soft, (mask > 0)[..., None].astype(np.float32) * 0.9)
    out = img.astype(np.float32) * (1 - soft) + bg.astype(np.float32) * soft
    out = np.clip(out, 0, 255).astype(np.uint8)
    # large marks: the median is itself contaminated, so inpaint them properly
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    big = np.zeros_like(mask)
    for i in range(1, n):
        if max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) > 9 * s:
            big[lab == i] = 255
    if big.any():
        bgr = cv2.inpaint(out[..., ::-1].copy(), big, max(3, int(round(2 * s))), cv2.INPAINT_TELEA)
        out = bgr[..., ::-1]
    return out
