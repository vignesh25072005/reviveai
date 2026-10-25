"""Diffusion engines (need a GPU + trained checkpoints; imported lazily).

Images : Stable Diffusion + the ControlNet you trained with scripts/train_image_controlnet.py
Audio  : SGMSE+ (https://github.com/sp-uhh/sgmse) run through its own enhancement.py

NOTE: written against diffusers>=0.27 / SGMSE+ main, NOT executed in the authoring sandbox
(no GPU there). If an API has drifted, the fix is usually a one-line change.
"""
from __future__ import annotations
import os, subprocess, tempfile
from pathlib import Path
import numpy as np

BASE_MODEL = os.environ.get("REVIVEAI_BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
CONTROLNET_DIR = os.environ.get("REVIVEAI_CONTROLNET", "checkpoints/controlnet")
SGMSE_DIR = os.environ.get("SGMSE_DIR", "")                    # path to a clone of sp-uhh/sgmse
SGMSE_CKPT = os.environ.get("SGMSE_CKPT", "checkpoints/sgmse.ckpt")


def image_engine_available() -> tuple[bool, str]:
    try:
        import torch, diffusers  # noqa: F401
    except ImportError:
        return False, "install torch + diffusers (requirements-train.txt)"
    if not torch.cuda.is_available():
        return False, "no CUDA GPU detected"
    if not Path(CONTROLNET_DIR).exists():
        return False, f"trained ControlNet not found at {CONTROLNET_DIR}"
    return True, "ready"


def audio_engine_available() -> tuple[bool, str]:
    if not SGMSE_DIR or not Path(SGMSE_DIR, "enhancement.py").exists():
        return False, "set SGMSE_DIR to your sgmse clone"
    if not Path(SGMSE_CKPT).exists():
        return False, f"SGMSE+ checkpoint not found at {SGMSE_CKPT}"
    return True, "ready"


_pipe = None


def _load_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
        cn = ControlNetModel.from_pretrained(CONTROLNET_DIR, torch_dtype=torch.float16)
        _pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            BASE_MODEL, controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
        _pipe.scheduler = UniPCMultistepScheduler.from_config(_pipe.scheduler.config)
        _pipe.enable_attention_slicing()
    return _pipe


def restore_image_diffusion(img_rgb: np.ndarray, strength: float = 0.45, steps: int = 30,
                            control_scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """strength: 0 = keep input, 1 = ignore input. Low values keep faces/structure faithful."""
    import torch
    from PIL import Image
    pipe = _load_pipe()
    h, w = img_rgb.shape[:2]
    W, H = (w // 8) * 8, (h // 8) * 8
    im = Image.fromarray(img_rgb).resize((W, H))
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe("a restored photograph", image=im, control_image=im, strength=strength,
               num_inference_steps=steps, controlnet_conditioning_scale=control_scale,
               guidance_scale=1.5, generator=g).images[0]
    return np.array(out.resize((w, h)))


def restore_audio_diffusion(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Runs SGMSE+ enhancement.py on a temp wav (16 kHz mono) and returns the enhanced audio."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    ok, why = audio_engine_available()
    if not ok:
        raise RuntimeError(why)
    x16 = x if sr == 16000 else resample_poly(x, 16000, sr).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "in").mkdir(); (Path(d) / "out").mkdir()
        wavfile.write(Path(d) / "in" / "a.wav", 16000, (np.clip(x16, -1, 1) * 32767).astype(np.int16))
        subprocess.run(["python", "enhancement.py", "--test_dir", str(Path(d) / "in"),
                        "--enhanced_dir", str(Path(d) / "out"), "--ckpt", str(Path(SGMSE_CKPT).resolve())],
                       cwd=SGMSE_DIR, check=True)
        r, y = wavfile.read(Path(d) / "out" / "a.wav")
    y = y.astype(np.float32) / (32768.0 if y.dtype == np.int16 else 1.0)
    return y, r


if __name__ == "__main__":     # batch mode: python -m reviveai.restore_diffusion IN_DIR OUT_DIR
    import sys, cv2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.png")):
        im = cv2.imread(str(f))[..., ::-1]
        cv2.imwrite(str(dst / f.name), restore_image_diffusion(im)[..., ::-1])
        print("restored", f.name)
