"""Synthetic damage for photographs.

Real (damaged, clean) pairs do not exist, so we damage clean photos on purpose.
Every operation is randomised so a model cannot memorise one damage style
(this is what narrows the synthetic-to-real gap).

degrade_image(img, rng, cfg) -> (damaged_rgb_uint8, mask_uint8, applied_ops)
`mask` is 255 where pixel content was destroyed (scratches, holes, dust).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class ImageDamageConfig:
    severity: float = 1.0          # global multiplier, 0 = almost none, 2 = heavy
    p_lowres: float = 0.40
    p_blur: float = 0.50
    p_fade: float = 0.85
    p_sepia: float = 0.35
    p_vignette: float = 0.30
    p_stain: float = 0.40
    p_grain: float = 0.85
    p_scratch: float = 0.70
    p_hole: float = 0.30
    p_dust: float = 0.70
    p_jpeg: float = 0.60


def _blob(h, w, rng, radius):
    """Irregular blob polygon (float 0/1) with a rough, torn-looking edge."""
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
    n = 24
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = radius * rng.uniform(0.55, 1.0, n)
    rad = np.convolve(np.r_[rad, rad[:2]], [0.25, 0.5, 0.25], mode="same")[:n]
    pts = np.stack([cx + rad * np.cos(ang) * rng.uniform(0.7, 1.3),
                    cy + rad * np.sin(ang)], 1).astype(np.int32)
    m = np.zeros((h, w), np.float32)
    cv2.fillPoly(m, [pts], 1.0)
    return m


def degrade_image(img: np.ndarray, rng: np.random.Generator,
                  cfg: ImageDamageConfig | None = None):
    cfg = cfg or ImageDamageConfig()
    s = float(cfg.severity)
    h, w = img.shape[:2]
    x = img.astype(np.float32) / 255.0
    mask = np.zeros((h, w), np.float32)
    ops: list[str] = []

    def hit(p):
        return rng.random() < min(1.0, p * (0.5 + 0.5 * s))

    # 1. optics / resolution
    if hit(cfg.p_lowres):
        f = rng.uniform(1.5, 1.5 + 2.5 * min(s, 1.5))
        small = cv2.resize(x, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        ops.append(f"low resolution x{f:.1f}")
    if hit(cfg.p_blur):
        if rng.random() < 0.7:
            sg = rng.uniform(0.6, 0.6 + 1.8 * s)
            x = cv2.GaussianBlur(x, (0, 0), sg)
            ops.append(f"blur sigma={sg:.1f}")
        else:
            k = int(rng.integers(5, 5 + int(10 * s) + 1)) | 1
            ker = np.zeros((k, k), np.float32)
            ker[k // 2, :] = 1
            M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1)
            ker = cv2.warpAffine(ker, M, (k, k))
            ker /= max(ker.sum(), 1e-6)
            x = cv2.filter2D(x, -1, ker)
            ops.append(f"motion blur k={k}")

    # 2. ageing: fade, colour cast, sepia, vignette, stains
    if hit(cfg.p_fade):
        c = rng.uniform(1 - 0.5 * min(s, 1.4), 0.95)
        lift = rng.uniform(0.0, 0.18) * s
        gain = rng.uniform(0.92, 1.0, 3) * np.array([1.0, 0.97, 0.88]) ** rng.uniform(0, 1)
        x = np.clip((x - 0.5) * c + 0.5 + lift, 0, 1) * gain
        ops.append(f"fading contrast={c:.2f}")
    if hit(cfg.p_sepia):
        g = x @ np.array([0.299, 0.587, 0.114], np.float32)
        tint = np.array([1.07, 0.95, 0.74], np.float32)
        a = rng.uniform(0.6, 1.0)
        x = (1 - a) * x + a * (g[..., None] * tint)
        ops.append("sepia tone")
    if hit(cfg.p_vignette):
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        x = x * (1 - rng.uniform(0.15, 0.45) * np.clip(r - 0.3, 0, 1) ** 1.5)[..., None]
        ops.append("vignette")
    if hit(cfg.p_stain):
        for _ in range(int(rng.integers(1, 4))):
            m = _blob(h, w, rng, rng.uniform(0.08, 0.28) * min(h, w))
            m = cv2.GaussianBlur(m, (0, 0), rng.uniform(6, 20))
            col = np.array([0.55, 0.40, 0.22]) * rng.uniform(0.6, 1.1)
            a = (rng.uniform(0.15, 0.45) * min(s, 1.5)) * m
            x = x * (1 - a[..., None]) + (x * col) * a[..., None]
        ops.append("stains")

    # 3. film grain
    if hit(cfg.p_grain):
        sg = rng.uniform(0.02, 0.02 + 0.07 * s)
        lum = x.mean(-1, keepdims=True)
        x = x + rng.normal(0, 1, (h, w, 1)).astype(np.float32) * sg * (0.6 + 0.8 * (1 - np.abs(lum - 0.5) * 2))
        x = x + rng.normal(0, sg * 0.25, x.shape).astype(np.float32)
        ops.append(f"film grain sigma={sg:.3f}")

    # 4. physical damage (destroys content -> goes into the mask)
    if hit(cfg.p_scratch):
        layer = np.zeros((h, w), np.float32)
        n = int(rng.integers(2, 4 + int(8 * s)))
        for _ in range(n):
            x0 = rng.uniform(0, w)
            ln = rng.uniform(0.15, 1.0) * h
            y0 = rng.uniform(0, max(1.0, h - ln))
            steps = max(4, int(ln / 6))
            xs = x0 + np.cumsum(rng.normal(0, 0.8, steps))
            ys = np.linspace(y0, y0 + ln, steps)
            pts = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(layer, [pts], False, 1.0, int(rng.integers(1, 3)), cv2.LINE_AA)
        colv = 1.0 if rng.random() < 0.65 else 0.05
        a = np.clip(layer * rng.uniform(0.6, 1.0), 0, 1)
        x = x * (1 - a[..., None]) + colv * a[..., None]
        mask = np.maximum(mask, (a > 0.25).astype(np.float32))
        ops.append(f"scratches x{n}")
    if hit(cfg.p_hole):
        for _ in range(int(rng.integers(1, 3 + int(2 * s)))):
            m = _blob(h, w, rng, rng.uniform(0.03, 0.10 + 0.05 * s) * min(h, w))
            fill = np.array([0.93, 0.90, 0.82]) if rng.random() < 0.7 else np.array([0.06, 0.05, 0.05])
            m3 = cv2.GaussianBlur(m, (0, 0), 0.8)[..., None]
            x = x * (1 - m3) + fill * m3
            mask = np.maximum(mask, m)
        ops.append("tears / holes")
    if hit(cfg.p_dust):
        n = int(rng.integers(20, 60 + int(200 * s)))
        layer = np.zeros((h, w), np.float32)
        for _ in range(n):
            cv2.circle(layer, (int(rng.uniform(0, w)), int(rng.uniform(0, h))),
                       int(rng.integers(0, 3)), 1.0, -1, cv2.LINE_AA)
        colv = 1.0 if rng.random() < 0.6 else 0.05
        x = x * (1 - layer[..., None]) + colv * layer[..., None]
        mask = np.maximum(mask, (layer > 0.3).astype(np.float32))
        ops.append(f"dust specks x{n}")

    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)

    # 5. compression last (a scan saved as a low-quality JPEG)
    if hit(cfg.p_jpeg):
        q = int(rng.integers(15, 60))
        _, enc = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
        ops.append(f"JPEG quality={q}")
    return out, (mask * 255).astype(np.uint8), ops
