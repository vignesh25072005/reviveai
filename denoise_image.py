"""Image denoising for any common kind of photographic noise.

What it handles
  grain / sensor / high-ISO noise ........ BM3D (Studio) or non-local means (Fast), noise level estimated automatically
  speckle, salt-and-pepper, "snow" ....... detected pixel by pixel (ROAD statistic), repaired from clean neighbours
  colour (chroma) noise .................. BM3D works in a luma/chroma space and smooths colour noise harder than detail
  mixtures of the above .................. speckle is removed first, then the remaining grain is measured and removed
  black-and-white photos ................. detected and processed as one channel (3x faster)

denoise_image(img_rgb_uint8, mode="studio", strength=0.5, grain=0.1, sigma=None, progress=None, workers=None)
    -> (denoised_rgb_uint8, info_dict)
analyze_noise(img_rgb_uint8) -> dict(kind, severity, impulse_pct, sigma, gray)
"""
from __future__ import annotations
import os
import time
import warnings
import numpy as np
import cv2
from scipy.signal import convolve2d

_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
SEVERITY = ["clean", "light", "moderate", "heavy", "severe"]


# ---------------------------------------------------------------- measuring noise
def is_gray(img: np.ndarray) -> bool:
    if img.ndim == 2:
        return True
    a = img.astype(np.int16)
    return float(np.abs(a[..., 0] - a[..., 2]).mean()) < 1.0 and float(np.abs(a[..., 1] - a[..., 2]).mean()) < 1.0


def estimate_sigma(img: np.ndarray) -> float:
    """Gaussian-like noise level (0-255 units). Median-based, so edges and texture barely bias it."""
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    est = [np.median(np.abs(convolve2d(x[..., c], _MASK, mode="valid"))) / 0.6745 / 6.0 for c in range(x.shape[2])]
    return float(np.mean(est))


def _road(ch: np.ndarray, band: int = 512) -> np.ndarray:
    """ROAD statistic: sum of the 4 smallest absolute differences to the 8 neighbours.
    An isolated noisy pixel differs from ALL neighbours (large value); a real edge or texture pixel
    always has some similar neighbours (small value). Processed in row bands to keep memory low."""
    h, w = ch.shape
    p = np.pad(ch.astype(np.float32), 1, mode="reflect")
    out = np.empty((h, w), np.float32)
    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        c = p[1 + y0:1 + y1, 1:1 + w]
        d = np.stack([np.abs(p[1 + dy + y0:1 + dy + y1, 1 + dx:1 + dx + w] - c)
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)], 0)
        out[y0:y1] = np.partition(d, 3, axis=0)[:4].sum(0)
    return out


def _blob_response(ch: np.ndarray) -> np.ndarray:
    """How strongly each pixel stands out as a small (up to ~4 px) bright or dark blob."""
    g = np.clip(ch, 0, 255).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return np.maximum(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se), cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)).astype(np.float32)


def _flat_region(ch: np.ndarray) -> np.ndarray:
    """Areas with no real detail (after a median blur), where any speckle must be noise, not texture."""
    m = cv2.medianBlur(np.clip(ch, 0, 255).astype(np.uint8), 5).astype(np.float32)
    mean, sq = cv2.blur(m, (9, 9)), cv2.blur(m * m, (9, 9))
    return np.sqrt(np.maximum(sq - mean * mean, 0)) < 5.0


SPECKLE_MIN_DENSITY = 0.008      # share of FLAT pixels that must be flagged before we call it speckle noise


def _outlier_mask(ch: np.ndarray, flat: np.ndarray, sig: float, relax: float = 1.0, tex: float = 1.0) -> np.ndarray:
    """Pixels that stand out as small blobs or isolated impulses. Thresholds scale with the grain level `sig`
    so Gaussian noise tails are not mistaken for speckle, and are stricter in textured areas so detail survives."""
    r = _blob_response(ch)
    # later passes (relax < 1) only get more sensitive in FLAT areas, where a speck cannot be real detail
    hit = np.where(flat, r > relax * max(45.0, 7.0 * sig), r > tex * max(85.0, 9.0 * sig)) | (_road(ch) > 130.0 + 5.0 * sig)
    return cv2.dilate(hit.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _is_even(mask: np.ndarray, flat: np.ndarray, grid: int = 6) -> bool:
    """Speckle noise is spread evenly over the frame; natural highlights and fine detail sit in a few places.
    True when at least half of the image regions that have flat area contain a noise-like density of flags."""
    h, w = mask.shape
    dens = []
    for i in range(grid):
        for j in range(grid):
            sl = (slice(i * h // grid, (i + 1) * h // grid), slice(j * w // grid, (j + 1) * w // grid))
            f = flat[sl]
            if f.sum() >= 150:
                dens.append((mask[sl] & f).sum() / f.sum())
    if len(dens) < 3:
        return bool(len(dens) and np.median(dens) >= 0.015)
    return bool(np.median(dens) >= 0.005 and np.mean(np.array(dens) >= 0.003) >= 0.6)


def speckle_mask(img: np.ndarray, sigma: float = 0.0, relax: float = 1.0, gate: bool = True, tex: float = 1.0):
    """-> (mask HxWx3 bool, density). Whether speckle exists is decided on LUMINANCE only (colour noise leaves
    brightness alone), by the share of flat-area pixels flagged: real detail lives in textured areas, so flat
    areas isolate noise. The mask is empty when that share is tiny."""
    gray = img.ndim == 2 or is_gray(img)
    luma = img.astype(np.float32) if img.ndim == 2 else img.astype(np.float32).mean(-1)
    flat = _flat_region(luma)
    sig_l = estimate_sigma(luma[..., None])
    m_luma = _outlier_mask(luma, flat, sig_l, relax, tex)
    dens = float((m_luma & flat).sum() / max(flat.sum(), 1))
    if gate and (dens < SPECKLE_MIN_DENSITY or not _is_even(m_luma, flat)):
        return np.zeros(img.shape[:2] + (3,), bool), 0.0      # rejected: report no speckle rather than a misleading density
    if gray:
        return np.repeat(m_luma[..., None], 3, 2), dens
    masks = [_outlier_mask(img[..., c], flat, estimate_sigma(img[..., c][..., None]), relax, tex) for c in range(3)]
    return np.stack(masks, -1), dens


def _repair_channel(ch: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace flagged pixels by the median of CLEAN neighbours, growing the window until enough exist."""
    from skimage.filters import rank
    from skimage.morphology import disk
    out, todo, valid = ch.copy(), bad.copy(), (~bad)
    vmask = valid.astype(np.uint8)
    for r in (1, 2, 3, 4, 6, 9):
        if not todo.any():
            break
        fp = disk(r)
        med = rank.median(ch, footprint=fp, mask=vmask)
        cnt = cv2.filter2D(valid.astype(np.float32), -1, fp.astype(np.float32), borderType=cv2.BORDER_REFLECT)
        fill = todo & (cnt >= 3)
        out[fill] = med[fill]
        todo &= ~fill
    if todo.any():                                   # pathological density: last resort
        out = cv2.inpaint(out, todo.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    return out


def remove_impulses(img: np.ndarray, passes: int = 3, strength: float = 0.5):
    """Detect and repair speckle. Returns (repaired, fraction_of_pixels_repaired, flat_area_density)."""
    out = img.copy()
    total = np.zeros(img.shape[:2], bool)
    dens0 = 0.0
    k = 2.0 * (float(np.clip(strength, 0, 1)) - 0.5)           # -1 gentle ... 0 default ... +1 thorough
    relax = (1.0, 0.8 - 0.15 * k, 0.65 - 0.2 * k)
    tex = 1.0 - 0.2 * max(k, 0.0)
    for i in range(passes):
        bad, dens = speckle_mask(out, estimate_sigma(out) if out.size < 4_000_000 else 0.0,
                                 relax=relax[min(i, 2)], gate=(i == 0), tex=(tex if i else 1.0))
        if i == 0:
            dens0 = dens
        if not bad.any():
            break
        total |= bad.any(-1)
        for c in range(3):
            out[..., c] = _repair_channel(out[..., c], bad[..., c])
    return out, float(total.mean()), dens0


def analyze_noise(img: np.ndarray) -> dict:
    """Cheap description of the noise in a photo (used for the readout before denoising)."""
    img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    gray = is_gray(img)
    h, w = img.shape[:2]
    if h * w > 2_000_000:                              # look at a central crop on huge images
        cy, cx = h // 2, w // 2
        img = img[max(0, cy - 700):cy + 700, max(0, cx - 700):cx + 700]
    rep, frac, dens = remove_impulses(img, passes=1)
    sigma = estimate_sigma(rep)
    sev_imp = 0 if dens < SPECKLE_MIN_DENSITY else 1 if dens < 0.01 else 2 if dens < 0.04 else 3 if dens < 0.12 else 4
    sev_g = 0 if sigma < 2.5 else 1 if sigma < 8 else 2 if sigma < 18 else 3 if sigma < 32 else 4
    kind = ("speckle and grain" if sev_imp and sev_g else "speckle" if sev_imp else "grain" if sev_g else "none")
    if not gray and not sev_imp:
        s_y, s_c = chroma_noise_levels(rep)
        if s_c >= 3.5 and s_y < 2.5:
            kind, sigma = "colour noise", s_c
            sev_g = 1 if s_c < 8 else 2 if s_c < 18 else 3 if s_c < 32 else 4
    return dict(kind=kind, severity=SEVERITY[max(sev_imp, sev_g)], impulse_pct=round(100 * dens, 1),
                sigma=round(sigma, 1), gray=bool(gray))


# ---------------------------------------------------------------- BM3D (tiled, guarded)
def _feather(h, w, ov):
    ry = np.ones(h, np.float32); rx = np.ones(w, np.float32)
    ramp = np.linspace(0.05, 1, ov, dtype=np.float32)
    ry[:ov], ry[-ov:] = ramp, ramp[::-1]
    rx[:ov], rx[-ov:] = ramp, ramp[::-1]
    return (ry[:, None] * rx[None, :])[..., None]


def _bm3d_tile(args):
    """One tile -> denoised float32 tile in [0,1]. Never returns NaN: flat tiles pass through and any
    numerical failure falls back to non-local means (a flat black background used to give NaN -> black)."""
    z, sigma, gray = args
    if float(z.std()) < 1.5e-3:
        return z
    out = None
    try:
        import bm3d
        prof = bm3d.BM3DProfileLC()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if gray:
                g = bm3d.bm3d(z[..., 0], sigma / 255.0, profile=prof)
                out = np.repeat(np.asarray(g, np.float32)[..., None], 3, 2)
            else:
                out = np.asarray(bm3d.bm3d_rgb(z, sigma / 255.0, profile=prof), np.float32)
    except Exception:
        out = None
    if out is None or not np.isfinite(out).all():
        u8 = (np.clip(z, 0, 1) * 255 + 0.5).astype(np.uint8)
        h = max(sigma * 0.55, 1.0)
        out = cv2.fastNlMeansDenoisingColored(u8[..., ::-1].copy(), None, h, h, 7, 21)[..., ::-1].astype(np.float32) / 255.0
    return np.clip(out, 0, 1)


_POOL = None


def _pool(n):
    global _POOL
    if _POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        _POOL = ProcessPoolExecutor(max_workers=n)
    return _POOL


def _bm3d_tiled(img, sigma, gray, tile=256, ov=24, progress=None, workers=None):
    h, w = img.shape[:2]
    z = img.astype(np.float32) / 255.0
    if max(h, w) <= tile + ov:
        out = _bm3d_tile((z, sigma, gray))
        if progress:
            progress(1, 1)
        return (out * 255 + 0.5).astype(np.uint8)
    step = tile - 2 * ov
    boxes = []
    for y in range(0, max(h - 2 * ov, 1), step):
        for x in range(0, max(w - 2 * ov, 1), step):
            y0, x0 = min(y, max(h - tile, 0)), min(x, max(w - tile, 0))
            boxes.append((y0, min(y0 + tile, h), x0, min(x0 + tile, w)))
    acc = np.zeros_like(z); wsum = np.zeros((h, w, 1), np.float32)
    jobs = [(z[a:b, c:d].copy(), sigma, gray) for a, b, c, d in boxes]
    workers = workers if workers is not None else min(4, (os.cpu_count() or 1))
    results = None
    if workers > 1 and len(jobs) > 1:
        try:
            ex = _pool(workers)
            futs = [ex.submit(_bm3d_tile, j) for j in jobs]
            results = []
            for i, f in enumerate(futs):
                results.append(f.result())
                if progress:
                    progress(i + 1, len(jobs))
        except Exception:
            results = None                                   # process pool unavailable: do it serially
    if results is None:
        results = []
        for i, j in enumerate(jobs):
            results.append(_bm3d_tile(j))
            if progress:
                progress(i + 1, len(jobs))
    for (a, b, c, d), t in zip(boxes, results):
        f = _feather(b - a, d - c, ov)
        acc[a:b, c:d] += t * f; wsum[a:b, c:d] += f
    out = acc / np.maximum(wsum, 1e-6)
    return (np.nan_to_num(out, nan=0.0) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def chroma_noise_levels(img: np.ndarray):
    """(luma sigma, chroma sigma): colour noise leaves brightness alone but blotches Cr/Cb."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    def blotch(c):          # colour noise is blotchy (low frequency): compare with a smoothed copy, median keeps edges out
        c = c.astype(np.float32)
        return float(np.median(np.abs(c - cv2.GaussianBlur(c, (0, 0), 2.5))) / 0.6745)
    return estimate_sigma(ycc[..., 0]), max(blotch(ycc[..., 1]), blotch(ycc[..., 2]))


def denoise_chroma(img: np.ndarray, sigma_c: float) -> np.ndarray:
    """Smooth colour channels hard, leave brightness (all the visible detail) untouched."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    h = float(np.clip(1.5 * sigma_c + 2.0, 4.0, 40.0))
    for c in (1, 2):
        ycc[..., c] = cv2.fastNlMeansDenoising(ycc[..., c], None, h, 7, 21)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


# ---------------------------------------------------------------- public entry point
def denoise_image(img: np.ndarray, mode: str = "studio", strength: float = 0.5, grain: float = 0.1,
                  sigma: float | None = None, progress=None, workers: int | None = None):
    t0 = time.time()
    img = np.ascontiguousarray(img[..., :3]).astype(np.uint8)
    gray = is_gray(img)
    stages = []
    if progress:
        progress(0, 1)

    # 1) speckle / salt-and-pepper
    work, frac, _dens = remove_impulses(img, strength=strength)
    if frac >= 0.001:
        stages.append(f"speckle removed ({100 * frac:.1f}% of pixels)")
    else:
        work, frac = img, 0.0

    # 1b) colour noise only (brightness is clean): fix the colour channels, keep the detail
    if not gray and sigma is None:
        s_y, s_c = chroma_noise_levels(work)
        if s_c >= 3.5 and s_y < 2.5:
            out = denoise_chroma(work, s_c)
            stages.append("colour noise smoothed")
            if progress:
                progress(1, 1)
            return out, {"sigma_estimated": round(s_c, 1), "sigma_used": round(s_c, 1), "mode": "chroma", "kind": "colour noise",
                         "impulse_pct": round(100 * frac, 1), "stages": stages, "gray": False,
                         "seconds": round(time.time() - t0, 2), "size": [int(img.shape[1]), int(img.shape[0])]}

    # 2) remaining grain
    est = estimate_sigma(work) if sigma is None else float(sigma)
    used = max(est * (0.6 + 0.8 * float(np.clip(strength, 0, 1))), 0.5)
    if frac > 0:
        used = max(used, 3.0)                                # also blends the patches that were repaired
    do_grain = (sigma is not None) or est >= 2.5 or frac > 0
    if not do_grain:
        out, mode_used = work.copy(), "skipped (no noise found)"
        if progress:
            progress(1, 1)
    elif mode in ("ai", "custom"):
        from . import ai_denoise                                  # raises AIUnavailable with a user-readable message
        eng = "custom" if mode == "custom" else "scunet"
        out = ai_denoise.run(work, strength=strength, progress=(lambda d, t: progress(d, t)) if progress else None, engine=eng)
        mode_used = mode; stages.append("noise removed by your own trained model" if mode == "custom" else "noise removed by the AI model (SCUNet)")
    elif mode == "fast":
        h = used * 0.55
        out = cv2.fastNlMeansDenoisingColored(work[..., ::-1].copy(), None, h, h, 7, 21)[..., ::-1].copy()
        mode_used = "fast"; stages.append("grain smoothed (fast)")
        if progress:
            progress(1, 1)
    else:
        out = _bm3d_tiled(work, used, gray, progress=progress, workers=workers)
        mode_used = "studio"; stages.append("grain removed (BM3D)")
    if grain > 0 and mode_used in ("fast", "studio", "ai", "custom"):
        out = np.clip(out.astype(np.float32) + float(grain) * 0.35 * (work.astype(np.float32) - out), 0, 255).astype(np.uint8)
    kind = "speckle and grain" if frac > 0 and est >= 2.5 else "speckle" if frac > 0 else "grain" if est >= 2.5 else "none"
    return out, {"sigma_estimated": round(est, 1), "sigma_used": round(used, 1), "mode": mode_used, "kind": kind,
                 "impulse_pct": round(100 * frac, 1), "stages": stages, "gray": bool(gray),
                 "seconds": round(time.time() - t0, 2), "size": [int(img.shape[1]), int(img.shape[0])]}
