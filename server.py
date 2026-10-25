"""ReviveAI server + web app.

    python server.py            ->  http://127.0.0.1:8000   (opens your browser)

Everything runs on this computer. Uploads are kept in a temporary folder and deleted after an hour.
API (used by web/app.js):
    GET  /api/info
    POST /api/image/inspect     file                         -> noise level, size, time estimate
    POST /api/image             file, mode, strength, grain, sigma  -> job
    POST /api/audio             file, strength, hum          -> job
    GET  /api/jobs/{id}         status, progress, result
    GET  /api/files/{id}/{name}
"""
from __future__ import annotations
import os, shutil, tempfile, threading, time, uuid, webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.io import wavfile
from scipy.signal import resample_poly

from reviveai import __version__
from reviveai.denoise_image import denoise_image, analyze_noise
from reviveai.enhance_photo import restore_old_photo
from reviveai import ai_denoise
from reviveai.denoise_audio import denoise_audio, noise_floor_dbfs
from reviveai.viz import spectrogram_png

ROOT = Path(__file__).parent
WORK = Path(tempfile.mkdtemp(prefix="reviveai_"))
MAX_UPLOAD = 60 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_AUDIO_S = 300
TTL = 3600
SEC_PER_MP_STUDIO = 60.0          # rough single-core figure used for the time estimate shown in the UI

app = FastAPI(title="ReviveAI", version=__version__)
pool = ThreadPoolExecutor(max_workers=max(1, min(2, (os.cpu_count() or 1))))
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


# ---------------------------------------------------------------- job plumbing
def _gc():
    now = time.time()
    with LOCK:
        for jid in [j for j, v in JOBS.items() if now - v["created"] > TTL]:
            shutil.rmtree(WORK / jid, ignore_errors=True)
            JOBS.pop(jid, None)


def _new_job(kind: str) -> tuple[str, Path]:
    _gc()
    jid = uuid.uuid4().hex[:12]
    d = WORK / jid
    d.mkdir(parents=True)
    with LOCK:
        JOBS[jid] = dict(id=jid, kind=kind, status="queued", progress=0.0, message="Queued", result=None, created=time.time())
    return jid, d


def _set(jid, **kw):
    with LOCK:
        JOBS[jid].update(kw)


async def _save_upload(file: UploadFile, d: Path, stem: str) -> Path:
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That file is larger than 60 MB.")
    p = d / (stem + (Path(file.filename or "").suffix.lower()[:8] or ".bin"))
    p.write_bytes(data)
    return p


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("This file could not be read as an image. Use JPG, PNG, WebP or BMP.")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise ValueError(f"This image is larger than {MAX_PIXELS // 1_000_000} megapixels. Resize it first.")
    return img[..., ::-1].copy()


def _png(path: Path, rgb: np.ndarray):
    cv2.imwrite(str(path), rgb[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    """-> float32 array (samples, channels) in [-1, 1], sample rate."""
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        try:
            sr, x = wavfile.read(path)
        except Exception:
            raise ValueError("This file could not be read as audio. Use WAV, MP3, FLAC or OGG.")
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1)
        x = x.astype(np.float32)
        x = x[:, None] if x.ndim == 1 else x
    if len(x) / sr > MAX_AUDIO_S:
        raise ValueError(f"This recording is longer than {MAX_AUDIO_S // 60} minutes. Trim it first.")
    return x, int(sr)


def _wav(path: Path, x: np.ndarray, sr: int):
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


# ---------------------------------------------------------------- workers
def _run_image(jid: str, d: Path, src: Path, mode: str, strength: float, grain: float, sigma: float | None, task: str = "denoise", rp: dict | None = None):
    try:
        _set(jid, status="running", message="Reading photo")
        rgb = _read_image(src)
        _png(d / "before.png", rgb)

        def prog(done, total):
            _set(jid, progress=0.05 + 0.9 * done / total, message=f"Denoising, section {done} of {total}")

        if task == "restore":
            _set(jid, progress=0.3, message="Restoring photo")
            out, info = restore_old_photo(rgb, **(rp or {}))
            _png(d / "after.png", out)
            _set(jid, status="done", progress=1.0, message="Done",
                 result=dict(before=f"/api/files/{jid}/before.png", after=f"/api/files/{jid}/after.png", info=info))
            return
        out, info = denoise_image(rgb, mode, strength, grain, sigma, progress=prog)
        _png(d / "after.png", out)
        _set(jid, progress=0.97, message="Checking the result")
        info["before"] = analyze_noise(rgb)
        info["after"] = analyze_noise(out)
        _set(jid, status="done", progress=1.0, message="Done",
             result=dict(before=f"/api/files/{jid}/before.png", after=f"/api/files/{jid}/after.png", info=info))
    except Exception as e:  # surfaced to the UI as a plain message
        _set(jid, status="error", message=str(e))


def _run_audio(jid: str, d: Path, src: Path, strength: float, hum: str):
    try:
        t0 = time.time()
        _set(jid, status="running", message="Reading audio")
        x, sr = _read_audio(src)
        ch = x.shape[1]

        def prog(f):
            _set(jid, progress=0.05 + 0.8 * f, message="Denoising")
        y, dinfo = denoise_audio(x if ch > 1 else x[:, 0], sr, strength, "auto" if hum == "auto" else None, progress=prog)
        y = y if ch > 1 else y[:, None]
        steps, snr = dinfo["steps"], [dinfo["input_snr_db"]]
        _set(jid, progress=0.88, message="Drawing spectrograms")
        _wav(d / "before.wav", x, sr)
        _wav(d / "after.wav", y, sr)
        mono_b, mono_a = x.mean(1), y.mean(1)
        pb, ref = spectrogram_png(mono_b, sr)
        pa, _ = spectrogram_png(mono_a, sr, ref_db=ref)
        (d / "spec_before.png").write_bytes(pb)
        (d / "spec_after.png").write_bytes(pa)
        nb, na = noise_floor_dbfs(mono_b, sr), noise_floor_dbfs(mono_a, sr)
        info = dict(steps=steps, duration=round(len(x) / sr, 1), sample_rate=sr, channels=ch,
                    input_snr_db=round(float(np.mean(snr)), 1), noise_floor_before=round(nb, 1),
                    noise_floor_after=round(na, 1), reduction_db=round(nb - na, 1),
                    seconds=round(time.time() - t0, 1))
        base = f"/api/files/{jid}/"
        _set(jid, status="done", progress=1.0, message="Done",
             result=dict(before=base + "before.wav", after=base + "after.wav", spec_before=base + "spec_before.png",
                         spec_after=base + "spec_after.png", info=info))
    except Exception as e:
        _set(jid, status="error", message=str(e))


# ---------------------------------------------------------------- routes
@app.get("/api/info")
def info():
    return dict(version=__version__, cpus=os.cpu_count(), sec_per_mp_studio=SEC_PER_MP_STUDIO, ai=ai_denoise.status(), custom=ai_denoise.status("custom"),
                limits=dict(upload_mb=MAX_UPLOAD // 2 ** 20, megapixels=MAX_PIXELS // 10 ** 6, audio_minutes=MAX_AUDIO_S // 60))


@app.post("/api/image/inspect")
async def image_inspect(file: UploadFile = File(...)):
    d = Path(tempfile.mkdtemp(dir=WORK))
    try:
        p = await _save_upload(file, d, "probe")
        try:
            rgb = _read_image(p)
        except ValueError as e:
            raise HTTPException(400, str(e))
        a = analyze_noise(rgb)
        h, w = rgb.shape[:2]
        mp_cost = 0.4 if a["gray"] else 1.0
        return dict(a, width=w, height=h, studio_seconds=round(w * h / 1e6 * SEC_PER_MP_STUDIO * mp_cost),
                    suggest="restore" if a["kind"] == "none" else "denoise")
    finally:
        shutil.rmtree(d, ignore_errors=True)


@app.post("/api/image")
async def image_job(file: UploadFile = File(...), mode: str = Form("studio"), strength: float = Form(0.5),
                    grain: float = Form(0.1), sigma: str = Form(""), task: str = Form("denoise"),
                    contrast: float = Form(0.6), repair: float = Form(0.5), sharpen: float = Form(0.5),
                    clean: float = Form(0.5), tone: str = Form("keep")):
    if mode not in ("studio", "fast", "ai", "custom"):
        raise HTTPException(400, "mode must be studio, fast, ai or custom")
    if mode in ("ai", "custom") and task == "denoise":
        st = ai_denoise.status("custom" if mode == "custom" else "scunet")
        if not st["available"]:
            raise HTTPException(400, st["reason"])
    jid, d = _new_job("image")
    src = await _save_upload(file, d, "input")
    sg = float(sigma) if sigma.strip() else None
    if task not in ("denoise", "restore") or tone not in ("keep", "neutral", "bw"):
        raise HTTPException(400, "unknown task or tone")
    rp = dict(contrast=float(np.clip(contrast, 0, 1)), repair=float(np.clip(repair, 0, 1)),
              sharpen=float(np.clip(sharpen, 0, 1)), clean=float(np.clip(clean, 0, 1)), tone=tone)
    pool.submit(_run_image, jid, d, src, mode, float(np.clip(strength, 0, 1)), float(np.clip(grain, 0, 1)), sg, task, rp)
    return {"job": jid}


def _run_install(jid: str):
    try:
        _set(jid, status="running", message="Downloading the AI model")
        ai_denoise.download(lambda f: _set(jid, progress=f, message=f"Downloading the AI model, {int(f * 100)}%"))
        _set(jid, status="done", progress=1.0, message="Installed", result=dict(ai=ai_denoise.status()))
    except Exception as e:
        _set(jid, status="error", message=str(e))


@app.post("/api/ai/install")
def ai_install():
    st = ai_denoise.status()
    if st["available"]:
        return {"job": None, "already": True}
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise HTTPException(400, st["reason"])
    jid, _d = _new_job("install")
    pool.submit(_run_install, jid)
    return {"job": jid}


@app.post("/api/audio")
async def audio_job(file: UploadFile = File(...), strength: float = Form(0.4), hum: str = Form("auto")):
    jid, d = _new_job("audio")
    src = await _save_upload(file, d, "input")
    pool.submit(_run_audio, jid, d, src, float(np.clip(strength, 0, 1)), hum)
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            raise HTTPException(404, "Unknown or expired job.")
        return {k: j[k] for k in ("id", "kind", "status", "progress", "message", "result")}


@app.get("/api/files/{jid}/{name}")
def files(jid: str, name: str):
    if jid not in JOBS or "/" in name or ".." in name or name.startswith("input") or name.startswith("probe"):
        raise HTTPException(404)
    p = WORK / jid / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-store"})


app.mount("/samples", StaticFiles(directory=ROOT / "assets" / "samples"), name="samples")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", 8000))
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
