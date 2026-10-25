"""Optional AI engine: SCUNet, a pretrained *blind* neural-network denoiser.

SCUNet (Zhang et al., "Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis", Apache-2.0) was trained
on photos with many kinds of synthetic noise mixed together (Gaussian, Poisson/sensor noise, JPEG artefacts,
resizing...), so it needs no noise level and copes with real camera noise better than a single-purpose filter.
It is not trained for salt-and-pepper speckle, which is why ReviveAI removes speckle first and hands the rest to it.

Runs through onnxruntime on CPU (no PyTorch). The model is a community ONNX export of the authors' weights:
  https://huggingface.co/Heliosoph/scunet-onnx   (scunet_color_real_psnr.onnx + .onnx.data, ~75 MB, one-time download)

    python -m reviveai.ai_denoise --download      # fetch the model into ./models
    python -m reviveai.ai_denoise photo.png out.png

NOTE: the pipeline here (tiling, padding, blending, download, errors) is tested with a stand-in ONNX model.
The real SCUNet weights could not be downloaded where this was built, so check the first real result yourself.
"""
from __future__ import annotations
import os
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "scunet_color_real_psnr"
FILES = [MODEL + ".onnx", MODEL + ".onnx.data"]
CUSTOM = "custom_denoiser.onnx"          # produced by scripts/train_denoiser.py


class AIUnavailable(RuntimeError):
    """Raised with a message that is safe to show the user."""


def model_dir() -> Path:
    return Path(os.environ.get("REVIVEAI_MODELS", ROOT / "models"))


def url_base() -> str:
    return os.environ.get("REVIVEAI_MODEL_URL_BASE", "https://huggingface.co/Heliosoph/scunet-onnx/resolve/main/")


def status(engine: str = "scunet") -> dict:
    """{available, reason, installed[, info]}  - cheap, never loads the model."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return dict(available=False, installed=False,
                    reason="Needs one library: run  pip install onnxruntime  and restart ReviveAI.")
    if engine == "custom":
        if not (model_dir() / CUSTOM).exists():
            return dict(available=False, installed=False, info=None,
                        reason="You have not trained a model yet. Run scripts/train_denoiser.py on your own photos.")
        info = None
        try:
            import json
            info = json.loads((model_dir() / "custom_denoiser.json").read_text())
        except Exception:
            pass
        return dict(available=True, installed=True, reason="ready", info=info)
    have = all((model_dir() / f).exists() for f in FILES)
    if not have:
        return dict(available=False, installed=False, reason="The AI model has not been downloaded yet (about 75 MB, one time).")
    return dict(available=True, installed=True, reason="ready")


def download(progress=None) -> None:
    """Download the model files into model_dir(). progress(fraction 0..1). Verifies the model loads afterwards."""
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    base = url_base()
    try:
        sizes = []
        for f in FILES:
            req = urllib.request.Request(base + f, method="HEAD", headers={"User-Agent": "ReviveAI"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    sizes.append(int(r.headers.get("Content-Length") or 0))
            except Exception:
                sizes.append(0)
        total = sum(sizes) or 75_000_000
        done = 0
        for f in FILES:
            dest, part = d / f, d / (f + ".part")
            req = urllib.request.Request(base + f, headers={"User-Agent": "ReviveAI"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(min(0.98, done / total))
            part.replace(dest)
    except Exception as e:
        for f in FILES:
            (d / (f + ".part")).unlink(missing_ok=True)
        raise AIUnavailable(f"Could not download the AI model ({e}). Check your internet connection and try again.")
    try:
        sess = _session("scunet", force=True)
        sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, 64, 64), np.float32)})   # proves the files are a working model
    except Exception as e:
        for f in FILES:
            (d / f).unlink(missing_ok=True)
        raise AIUnavailable(f"The downloaded model files did not load ({e}). They were removed; try the download again.")
    if progress:
        progress(1.0)


_SESS: dict = {}


def _session(engine: str = "scunet", force: bool = False):
    if force or engine not in _SESS:
        st = status(engine)
        if not st["available"]:
            raise AIUnavailable(st["reason"])
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 1)
        prefer = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        path = model_dir() / (CUSTOM if engine == "custom" else FILES[0])
        _SESS[engine] = ort.InferenceSession(str(path), so, providers=prefer)
    return _SESS[engine]


def _feather(h, w, ov):
    ry = np.ones(h, np.float32); rx = np.ones(w, np.float32)
    ramp = np.linspace(0.05, 1, ov, dtype=np.float32)
    ry[:ov], ry[-ov:] = ramp, ramp[::-1]
    rx[:ov], rx[-ov:] = ramp, ramp[::-1]
    return (ry[:, None] * rx[None, :])[None]


def run(img: np.ndarray, strength: float = 0.5, tile: int = 512, overlap: int = 48, progress=None, engine: str = "scunet") -> np.ndarray:
    """Denoise an RGB uint8 image. strength 0.5 and up = full network output; below that the original is blended back."""
    sess = _session(engine)
    name = sess.get_inputs()[0].name
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8                                   # the network needs sizes divisible by 8
    x = np.pad(img.astype(np.float32) / 255.0, ((0, ph), (0, pw), (0, 0)), mode="reflect").transpose(2, 0, 1)
    H, W = x.shape[1:]
    tile = max(64, (tile // 8) * 8)
    ys = list(range(0, max(H - overlap, 1), tile - overlap)) or [0]
    xs = list(range(0, max(W - overlap, 1), tile - overlap)) or [0]
    acc = np.zeros_like(x); wsum = np.zeros((1, H, W), np.float32)
    total, done = len(ys) * len(xs), 0
    for y in ys:
        for xx in xs:
            y0, x0 = min(y, max(H - tile, 0)), min(xx, max(W - tile, 0))
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            piece = np.ascontiguousarray(x[None, :, y0:y1, x0:x1])
            pad_b, pad_r = (-piece.shape[2]) % 8, (-piece.shape[3]) % 8
            if pad_b or pad_r:
                piece = np.pad(piece, ((0, 0), (0, 0), (0, pad_b), (0, pad_r)), mode="reflect")
            o = sess.run(None, {name: piece.astype(np.float32)})[0][0]
            o = np.nan_to_num(o[:, :y1 - y0, :x1 - x0], nan=0.0)
            f = _feather(y1 - y0, x1 - x0, min(overlap, (y1 - y0) // 2, (x1 - x0) // 2) or 1)
            acc[:, y0:y1, x0:x1] += o * f; wsum[:, y0:y1, x0:x1] += f
            done += 1
            if progress:
                progress(done, total)
    out = np.clip(acc / np.maximum(wsum, 1e-6), 0, 1)[:, :h, :w].transpose(1, 2, 0)
    alpha = float(np.clip(0.4 + 1.2 * strength, 0, 1))             # 0.5 -> 1.0 (full), 0 -> 0.4 (gentle)
    out = alpha * out + (1 - alpha) * (img.astype(np.float32) / 255.0)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    import cv2
    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        print("Downloading the AI model into", model_dir())
        download(lambda f: print(f"\r  {f * 100:5.1f}%", end="", flush=True))
        print("\nDone.")
    elif len(sys.argv) >= 3:
        t = time.time()
        im = cv2.imread(sys.argv[1])[..., ::-1]
        cv2.imwrite(sys.argv[2], run(im)[..., ::-1])
        print(f"saved {sys.argv[2]} ({time.time() - t:.1f} s)")
    else:
        print(status())
