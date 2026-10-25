"""Photos that ship inside scikit-image (work offline). Used for demos and smoke tests only."""
from __future__ import annotations
import numpy as np

NAMES = ["astronaut", "coffee", "chelsea", "rocket", "camera", "coins"]


def builtin_images() -> dict[str, np.ndarray]:
    from skimage import data
    out = {}
    for n in NAMES:
        try:
            im = getattr(data, n)()
        except Exception as e:                      # a few need a download
            print(f"skip builtin {n}: {e}")
            continue
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        out[n] = im[..., :3].astype(np.uint8)
    return out


def random_crop(img: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Random square crop (55-100% of the short side), resized to `size`, randomly mirrored."""
    import cv2
    h, w = img.shape[:2]
    s = int(min(h, w) * rng.uniform(0.55, 1.0))
    if s < 16:
        s = min(h, w)
    y, x = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
    c = cv2.resize(img[y:y + s, x:x + s], (size, size),
                   interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)
    return c[:, ::-1].copy() if rng.random() < 0.5 else c
