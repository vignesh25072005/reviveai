"""Evaluation metrics. PSNR/SSIM/SI-SDR need only numpy + scikit-image.
LPIPS, PESQ and STOI are used when their packages are installed."""
from __future__ import annotations
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref: np.ndarray, est: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, est, data_range=255))


def ssim(ref: np.ndarray, est: np.ndarray) -> float:
    return float(structural_similarity(ref, est, channel_axis=-1, data_range=255))


def lpips_score(ref: np.ndarray, est: np.ndarray):
    """Lower is better. Returns None if `lpips`/`torch` are not installed."""
    try:
        import torch, lpips
    except ImportError:
        return None
    if not hasattr(lpips_score, "_net"):
        lpips_score._net = lpips.LPIPS(net="alex", verbose=False)
    t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float() / 127.5 - 1
    with torch.no_grad():
        return float(lpips_score._net(t(ref), t(est)))


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    n = min(len(ref), len(est))
    ref, est = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    ref, est = ref - ref.mean(), est - est.mean()
    alpha = (est @ ref) / (ref @ ref + eps)
    target = alpha * ref
    return float(10 * np.log10((target @ target + eps) / (((est - target) ** 2).sum() + eps)))


def stoi_score(ref, est, sr):
    try:
        from pystoi import stoi
    except ImportError:
        return None
    n = min(len(ref), len(est))
    return float(stoi(ref[:n], est[:n], sr, extended=False))


def pesq_score(ref, est, sr):
    try:
        from pesq import pesq
    except ImportError:
        return None
    n = min(len(ref), len(est))
    return float(pesq(sr, ref[:n], est[:n], "wb" if sr == 16000 else "nb"))
