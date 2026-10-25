"""Spectrogram rendering with a ReviveAI colour ramp (ink -> lapis -> marigold -> white)."""
from __future__ import annotations
import numpy as np
import cv2
from scipy import signal

_STOPS = np.array([0.0, 0.35, 0.72, 1.0])
_COLS = np.array([[15, 23, 34], [45, 79, 180], [242, 167, 27], [255, 250, 235]], np.float32)
LUT = np.stack([np.interp(np.linspace(0, 1, 256), _STOPS, _COLS[:, i]) for i in range(3)], 1).astype(np.uint8)


def spectrogram_png(x: np.ndarray, sr: int, ref_db: float | None = None, w: int = 1200, h: int = 220):
    """Returns (png_bytes, ref_db). Pass the first call's ref_db to the second so both share one scale."""
    _, _, Z = signal.stft(x, sr, nperseg=1024, noverlap=768)
    S = 20 * np.log10(np.abs(Z) + 1e-7)
    ref = float(S.max()) if ref_db is None else ref_db
    S = np.clip((S - (ref - 90)) / 90, 0, 1)[::-1]
    S = cv2.resize(S.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    rgb = LUT[(S * 255).astype(np.uint8)]
    ok, buf = cv2.imencode(".png", rgb[..., ::-1])
    return buf.tobytes(), ref
