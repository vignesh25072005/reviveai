"""ReviveAI web app.   python app.py   ->  http://127.0.0.1:7860

Engines
  fast       classical restoration; runs on any laptop, no downloads
  diffusion  your trained ControlNet (photos) / SGMSE+ (audio); needs a GPU + checkpoints
The UI falls back to `fast` (and says so) when a diffusion engine is not available.
"""
from __future__ import annotations
import json, os, tempfile, time, zipfile
from pathlib import Path
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from scipy import signal

from reviveai.restore_image_classic import restore_image
from reviveai.restore_audio_classic import restore_audio, to_mono_float
from reviveai.degrade_image import degrade_image, ImageDamageConfig
from reviveai.degrade_audio import degrade_audio, AudioDamageConfig
from reviveai.synth_audio import synth_clean_audio
from reviveai.samples import builtin_images, random_crop
from reviveai import restore_diffusion as rd

ROOT = Path(__file__).parent
SAMPLES = ROOT / "assets" / "samples"
MAX_SIDE, MAX_AUDIO_S = 1024, 60
SOURCES = builtin_images()
RECONSTRUCTION_NOTE = ("**Restored output is a reconstruction.** Where damage destroyed detail, the "
                       "software fills in what is plausible. It may not match the original, so do not use "
                       "restored copies as evidence of what a person, place or recording really looked or sounded like.")

# ------------------------------------------------------------------ helpers
def _fit(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _read_editor(val):
    """ImageEditor value -> (rgb uint8, user_mask uint8 or None)."""
    if val is None:
        return None, None
    if isinstance(val, np.ndarray):
        return val[..., :3].astype(np.uint8), None
    base = val.get("background")
    if base is None or np.asarray(base).size == 0 or (np.asarray(base).ndim == 3 and np.asarray(base).shape[2] == 4
                                                       and np.asarray(base)[..., 3].max() == 0):
        base = val.get("composite")
    if base is None:
        return None, None
    rgb = np.asarray(base)[..., :3].astype(np.uint8)
    mask = None
    for layer in val.get("layers") or []:
        L = np.asarray(layer)
        if L.ndim == 3 and L.shape[2] == 4:
            m = L[..., 3] > 10
            mask = m if mask is None else (mask | m)
    if mask is not None:
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        mask = mask.astype(np.uint8) * 255
        if not mask.any():
            mask = None
    return rgb, mask


def spectrogram(x, sr, w=560, h=150):
    _, _, Z = signal.stft(x, sr, nperseg=512, noverlap=384)
    S = 20 * np.log10(np.abs(Z) + 1e-6)
    S = np.clip(S, S.max() - 80, S.max())
    S = ((S - S.min()) / (S.max() - S.min() + 1e-9))[::-1]
    rgb = cv2.applyColorMap((S * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)[..., ::-1]
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def _pcm(x):
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


# ------------------------------------------------------------------ photo tab
def restore_photo(editor, engine, sensitivity, denoise, tone, strength):
    rgb, user_mask = _read_editor(editor)
    if rgb is None:
        raise gr.Error("Upload a photograph first, or pick one of the samples.")
    if max(rgb.shape[:2]) > MAX_SIDE:
        rgb = _fit(rgb)
        if user_mask is not None:
            user_mask = cv2.resize(user_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    t0, note = time.time(), ""
    if engine == "diffusion":
        ok, why = rd.image_engine_available()
        if ok:
            out = rd.restore_image_diffusion(rgb, strength=strength)
            mask = user_mask if user_mask is not None else np.zeros(rgb.shape[:2], np.uint8)
        else:
            note, engine = f"The diffusion engine is not available ({why}), so the fast engine was used. ", "fast"
    if engine == "fast":
        out, mask = restore_image(rgb, user_mask, sensitivity, denoise, tone, 0.0)
    marked = rgb.copy()
    marked[mask > 0] = (230, 57, 70)
    cover = 100 * float((mask > 0).mean())
    status = (f"{note}Done in {time.time() - t0:.1f} s. Damage found and repaired: {cover:.1f}% of pixels "
              f"(shown in red on the right). Drag the slider to compare.")
    return (rgb, out), marked, status


def load_photo(i):
    return np.array(Image.open(SAMPLES / f"photo_{i}.png").convert("RGB"))


# ------------------------------------------------------------------ audio tab
def restore_audio_fn(audio, engine, hum, clicks, denoise):
    if audio is None:
        raise gr.Error("Upload or record audio first, or pick one of the samples.")
    sr, x = audio
    x = to_mono_float(x)
    note = ""
    if len(x) > MAX_AUDIO_S * sr:
        x, note = x[: MAX_AUDIO_S * sr], f"Only the first {MAX_AUDIO_S} s were processed. "
    t0 = time.time()
    if engine == "diffusion":
        ok, why = rd.audio_engine_available()
        if ok:
            y, sr_out = rd.restore_audio_diffusion(x, sr)
            steps = ["SGMSE+ diffusion enhancement"]
        else:
            note += f"The diffusion engine is not available ({why}), so the fast engine was used. "
            engine = "fast"
    if engine == "fast":
        y, steps = restore_audio(x, sr, hum="auto" if hum else None, clicks=clicks, denoise=denoise)
        sr_out = sr
    status = f"{note}Done in {time.time() - t0:.1f} s. Applied: {', '.join(steps) or 'nothing'}."
    return (sr, _pcm(x)), (sr_out, _pcm(y)), spectrogram(x, sr), spectrogram(y, sr_out), status


# ------------------------------------------------------------------ training-data tab
def image_variants(source, severity, seed, count):
    rng = np.random.default_rng(int(seed))
    clean = random_crop(SOURCES[source], 256, rng)
    cfg = ImageDamageConfig(severity=severity)
    items = []
    for k in range(int(count)):
        d, _, ops = degrade_image(clean, np.random.default_rng(int(seed) * 1000 + k + 1), cfg)
        items.append((d, ", ".join(ops) or "no damage"))
    return clean, items


def audio_example(seed, severity):
    rng = np.random.default_rng(int(seed))
    clean = synth_clean_audio(rng, 4.0, 16000)
    noisy, ops = degrade_audio(clean, 16000, rng, AudioDamageConfig(severity=severity))
    return (16000, _pcm(clean)), (16000, _pcm(noisy)), spectrogram(clean, 16000), spectrogram(noisy, 16000), \
        "Damage added: " + (", ".join(ops) or "none")


# ------------------------------------------------------------------ dataset builder
def build_dataset(files, per_image, severity, size, seed):
    paths = [getattr(f, "name", f) for f in (files or [])]
    if not paths:
        raise gr.Error("Add at least one clean photograph.")
    paths = paths[:30]
    per_image = int(min(per_image, 600 // len(paths)))
    rng = np.random.default_rng(int(seed))
    cfg = ImageDamageConfig(severity=severity)
    tmp = Path(tempfile.mkdtemp())
    for sub in ("clean", "damaged", "mask"):
        (tmp / sub).mkdir()
    meta, preview = open(tmp / "metadata.jsonl", "w"), []
    n = 0
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        im, stem = im[..., ::-1], Path(p).stem
        for i in range(per_image):
            clean = random_crop(im, int(size), rng)
            dam, mask, ops = degrade_image(clean, rng, cfg)
            fn = f"{stem}_{i:03d}.png"
            cv2.imwrite(str(tmp / "clean" / fn), clean[..., ::-1])
            cv2.imwrite(str(tmp / "damaged" / fn), dam[..., ::-1])
            cv2.imwrite(str(tmp / "mask" / fn), mask)
            meta.write(json.dumps({"image": f"clean/{fn}", "conditioning_image": f"damaged/{fn}", "mask": f"mask/{fn}",
                                   "text": "a restored photograph", "ops": ops}) + "\n")
            n += 1
            if len(preview) < 8:
                preview.append((dam, "damaged"))
    meta.close()
    zp = tmp / "reviveai_pairs.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for f in tmp.rglob("*"):
            if f.is_file() and f != zp:
                z.write(f, f.relative_to(tmp))
    return str(zp), preview, (f"Made {n} pairs from {len(paths)} photographs. To train, split by source photograph "
                              f"(not by crop) so test images never appear in training. `scripts/make_dataset.py` does this for you.")


# ------------------------------------------------------------------ UI
CSS = """
.rv-head{padding:18px 4px 14px;border-bottom:2px solid #0F6E75;margin-bottom:10px}
.rv-head h1{font-family:'Source Serif 4',Georgia,serif;font-size:2.1rem;letter-spacing:-.01em;margin:0;color:#14262E}
.rv-head p{margin:.35rem 0 0;color:#46606B;font-size:1rem;max-width:64ch}
.dark .rv-head h1{color:#E8F0F2}.dark .rv-head p{color:#A9BDC5}
.rv-note{border-left:3px solid #0F6E75;padding:.55rem .85rem;background:#E9F1F1;color:#1B3A41;border-radius:0 6px 6px 0;font-size:.92rem}
.dark .rv-note{background:#15292E;color:#CFE3E6}
h2,h3{font-family:'Source Serif 4',Georgia,serif}
"""
HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" '
        'href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap">')
THEME = gr.themes.Base(primary_hue="teal", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Source Sans 3"), "system-ui", "sans-serif"])


def build_ui():
    ok_img, why_img = rd.image_engine_available()
    ok_aud, why_aud = rd.audio_engine_available()
    eng_img = [("Fast (runs anywhere)", "fast"),
               ("Diffusion (GPU + trained model)" if ok_img else f"Diffusion - unavailable: {why_img}", "diffusion")]
    eng_aud = [("Fast (runs anywhere)", "fast"),
               ("Diffusion (GPU + trained model)" if ok_aud else f"Diffusion - unavailable: {why_aud}", "diffusion")]

    with gr.Blocks(title="ReviveAI") as demo:
        gr.HTML('<div class="rv-head"><h1>ReviveAI</h1><p>Clean up scratched, faded and torn photographs, and '
                'noisy old recordings. Files stay on the machine running this app.</p></div>')
        gr.HTML(f'<div class="rv-note">{RECONSTRUCTION_NOTE}</div>')

        with gr.Tab("Photos"):
            with gr.Row():
                with gr.Column(scale=5):
                    editor = gr.ImageEditor(type="numpy", label="Your photograph (paint over any tears or holes)",
                                            sources=["upload", "clipboard"], height=420,
                                            brush=gr.Brush(default_size=14, colors=["#E63946"], color_mode="fixed"),
                                            transforms=[])
                    with gr.Row():
                        s_btns = [gr.Button(f"Sample {i}", size="sm") for i in range(1, 5)]
                    engine = gr.Radio(eng_img, value="fast", label="Engine")
                    with gr.Accordion("Settings", open=False):
                        sens = gr.Slider(0, 1, 0.5, step=0.05, label="Scratch and dust detection",
                                         info="Higher finds fainter damage but may repair real detail.")
                        den = gr.Slider(0, 20, 10, step=1, label="Grain removal")
                        tone = gr.Slider(0, 1, 0.3, step=0.05, label="Restore faded contrast and colour",
                                         info="Helps washed-out scans. Lower it for photos that were dark to begin with.")
                        strg = gr.Slider(0.1, 0.9, 0.45, step=0.05, label="Diffusion strength",
                                         info="Low keeps faces and structure faithful; high repaints more.")
                    go = gr.Button("Restore photo", variant="primary")
                with gr.Column(scale=5):
                    slider = gr.ImageSlider(label="Before / after", type="numpy", height=420)
                    status = gr.Markdown()
                    marked = gr.Image(label="Damage that was repaired (red)", interactive=False, height=240)
            go.click(restore_photo, [editor, engine, sens, den, tone, strg], [slider, marked, status])
            for i, b in enumerate(s_btns, 1):
                b.click(lambda i=i: load_photo(i), None, editor)

        with gr.Tab("Audio"):
            with gr.Row():
                with gr.Column(scale=5):
                    a_in = gr.Audio(type="numpy", sources=["upload", "microphone"], label="Your recording")
                    with gr.Row():
                        a_btns = [gr.Button(f"Sample {i}", size="sm") for i in (1, 2)]
                    a_eng = gr.Radio(eng_aud, value="fast", label="Engine")
                    with gr.Accordion("Settings", open=False):
                        a_hum = gr.Checkbox(True, label="Remove mains hum (50 or 60 Hz, found automatically)")
                        a_clk = gr.Checkbox(True, label="Remove clicks and pops")
                        a_dn = gr.Slider(0, 1, 0.3, step=0.05, label="Hiss reduction",
                                         info="High values start to dull the music or voice.")
                    a_go = gr.Button("Restore audio", variant="primary")
                with gr.Column(scale=5):
                    a_before = gr.Audio(label="Before", interactive=False)
                    a_after = gr.Audio(label="After", interactive=False)
                    with gr.Row():
                        sp_b = gr.Image(label="Before (spectrogram)", interactive=False, height=150)
                        sp_a = gr.Image(label="After (spectrogram)", interactive=False, height=150)
                    a_status = gr.Markdown()
            a_go.click(restore_audio_fn, [a_in, a_eng, a_hum, a_clk, a_dn], [a_before, a_after, sp_b, sp_a, a_status])
            for i, b in enumerate(a_btns, 1):
                b.click(lambda i=i: str(SAMPLES / f"audio_{i}.wav"), None, a_in)

        with gr.Tab("Training data"):
            gr.Markdown("### See what the models learn from\nEach training example is a clean photo or clip that was damaged "
                        "on purpose. The model learns to undo the damage, so the more varied the damage, the better "
                        "it copes with real old media.")
            with gr.Row():
                with gr.Column(scale=4):
                    src = gr.Dropdown(list(SOURCES), value=list(SOURCES)[0], label="Clean photo")
                    sev = gr.Slider(0, 2, 1.0, step=0.1, label="Damage level")
                    seed = gr.Number(7, precision=0, label="Random seed")
                    cnt = gr.Slider(2, 12, 8, step=1, label="Variants to show")
                    mk = gr.Button("Generate examples", variant="primary")
                with gr.Column(scale=6):
                    clean_out = gr.Image(label="Clean target", interactive=False, height=260)
                    gal = gr.Gallery(label="Damaged inputs (same photo, different damage)", columns=4, height="auto")
            mk.click(image_variants, [src, sev, seed, cnt], [clean_out, gal])
            gr.Markdown("### Audio examples")
            with gr.Row():
                with gr.Column(scale=4):
                    a_seed = gr.Number(3, precision=0, label="Random seed")
                    a_sev = gr.Slider(0, 2, 1.0, step=0.1, label="Damage level")
                    a_mk = gr.Button("Generate audio example", variant="primary")
                    a_ops = gr.Markdown()
                with gr.Column(scale=6):
                    c_au = gr.Audio(label="Clean target", interactive=False)
                    d_au = gr.Audio(label="Damaged input", interactive=False)
                    with gr.Row():
                        c_sp = gr.Image(label="Clean", interactive=False, height=150)
                        d_sp = gr.Image(label="Damaged", interactive=False, height=150)
            a_mk.click(audio_example, [a_seed, a_sev], [c_au, d_au, c_sp, d_sp, a_ops])

        with gr.Tab("Make a dataset"):
            gr.Markdown("### Build training pairs from your own clean photographs\nUpload sharp, undamaged photos "
                        "(family archives, public-domain images). You get a ZIP with `clean/`, `damaged/`, `mask/` and "
                        "`metadata.jsonl`, ready for `scripts/train_image_controlnet.py`.")
            with gr.Row():
                with gr.Column(scale=4):
                    up = gr.File(file_count="multiple", file_types=["image"], label="Clean photographs (up to 30)")
                    pp = gr.Slider(1, 50, 20, step=1, label="Pairs per photograph")
                    d_sev = gr.Slider(0, 2, 1.0, step=0.1, label="Damage level")
                    d_size = gr.Dropdown([256, 384, 512], value=256, label="Crop size (pixels)")
                    d_seed = gr.Number(0, precision=0, label="Random seed")
                    bd = gr.Button("Build dataset", variant="primary")
                with gr.Column(scale=6):
                    zf = gr.File(label="Download (ZIP)")
                    pv = gr.Gallery(label="Preview of damaged inputs", columns=4, height="auto")
                    bd_status = gr.Markdown()
            bd.click(build_dataset, [up, pp, d_sev, d_size, d_seed], [zf, pv, bd_status])

        with gr.Tab("About"):
            gr.Markdown(
                "**Fast engine** uses classical image and signal processing. It repairs thin scratches, dust, grain, "
                "hum, clicks and hiss, but it cannot rebuild lost detail, torn regions or missing high frequencies.\n\n"
                "**Diffusion engine** uses models you train with the scripts in this project "
                "(Stable Diffusion + ControlNet for photos, SGMSE+ for audio). It can reconstruct much more, "
                "which is also why it can invent details that were never there.\n\n"
                "**Measuring quality.** Photos: PSNR, SSIM and LPIPS. Audio: SI-SDR, STOI and PESQ. "
                "Run `scripts/evaluate.py` on held-out data. There is no single 'accuracy' for restoration, "
                "so report these numbers and show failure cases next to successes.")
    return demo


if __name__ == "__main__":
    build_ui().launch(theme=THEME, css=CSS, head=HEAD, server_name=os.environ.get("HOST", "127.0.0.1"),
                      server_port=int(os.environ.get("PORT", 7860)))
