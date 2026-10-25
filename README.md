# ReviveAI

Remove noise from any photograph or recording, and restore faded old prints. A local web app plus the engines behind it.

```bash
pip install -r requirements.txt
python server.py            # opens http://127.0.0.1:8000
```
Everything runs on your computer. Uploads sit in a temp folder and are deleted after an hour.

## The simple way to use it
Drop a photo or recording, press the big button. ReviveAI measures the problem and picks the right clean-up. Not happy? **Gentler** and **Stronger**
re-run it; **Download** saves the result. Every slider lives under **Fine-tune** and you can ignore it.

## Train your own model on 1,000+ images
This is the route if your project needs a model *you trained*. It needs PyTorch (a free Colab or Kaggle GPU is enough).
```bash
pip install torch opencv-python-headless numpy onnxruntime
python scripts/train_denoiser.py --data path/to/your_clean_photos --iters 20000
```
Put 1,000 or more varied clean photos in that folder (free sets: DIV2K, Flickr2K, BSD500; check licences). The script makes noisy/clean pairs on the
fly, keeps 3% of the photos aside to test on, trains a DnCNN-style network, and writes `models/custom_denoiser.onnx`. Restart ReviveAI and a **Mine**
option appears under Quality; its hint shows how many images it was trained on and its held-out score.

**Tested end to end here** (CPU only, tiny network, 800 iterations, 3 minutes, 1,009 images that were all crops of just 6 photos, so this only proves the
pipeline works): on 31 held-out photos PSNR rose 24.9 -> 29.9 dB (noise level 15), 20.6 -> 28.0 (level 25), 15.1 -> 23.9 (level 50). On a fresh photo at level
25 it reached 29.1 dB, **below** Studio (BM3D) at 32.2 dB. A proper GPU run on 1,000+ *varied* photos is what is meant to reach or pass Studio; that run has
not been done here, so do not quote a score until you have measured your own.
**"100% accuracy" is not a thing for denoising** (there is no single right answer to match). Report PSNR/SSIM on held-out photos, as above.

## What it handles

| Photos | How |
|---|---|
| Grain, sensor and high-ISO noise | BM3D (Studio) or non-local means (Fast), noise level measured automatically |
| Speckle, salt-and-pepper, "snow" | Found pixel by pixel, repaired from clean neighbours, then the remaining grain is removed |
| Colour (chroma) noise | Colour channels smoothed hard, brightness and detail left alone |
| Black-and-white photos | Detected automatically, processed as one channel |
| Mixtures of the above | Speckle first, then grain |
| Faded, soft, stained old prints | "Restore old photo" mode (contrast, dust and scratches, sharpening, colour tone) |

Audio: WAV, MP3, FLAC, OGG, mono or stereo, any sample rate, up to 5 minutes.
The photo mode is chosen for you from what is found in the image. **Strength** also controls how hard stubborn speckle is chased.

## AI mode: the closest thing to "any photo" (optional)
No denoiser is perfect on every file, but a trained neural network copes with far more kinds of real noise than a hand-made filter.
ReviveAI can use **SCUNet**, a pretrained *blind* denoiser (it needs no noise level and was trained on many mixed noise types).
```bash
pip install onnxruntime        # one small library, then restart ReviveAI
```
Then Photos -> Quality -> **AI** -> **Install the AI model** (one time, about 75 MB, needs internet). Or run `python -m reviveai.ai_denoise --download`.
Speckle is removed first (the network was not trained for salt-and-pepper), then the network handles the rest.
It runs on the CPU, so it is slower than Fast and large photos can take minutes.
**Status:** the surrounding code (download, tiling, odd sizes, blending, errors, the one-click installer) is tested with a stand-in model. The real
SCUNet weights could not be downloaded where this was built, so its real-world quality is **unmeasured here**. Check your first result yourself.
The model files are a community ONNX export (huggingface.co/Heliosoph/scunet-onnx, Apache-2.0) of the authors' weights.
There is no equivalent verified general model for music or audio; speech-cleaning networks tend to damage music.

## Measured results (synthetic noise, noise level found automatically). Reproduce: `python scripts/evaluate_denoise.py`

**Photos, Studio mode, 6 different photographs per row** ("removed" = 1 - 10^(-gain/10); 90% = a 10 dB gain):

| Noise | PSNR in -> out | SSIM in -> out | Noise removed |
|---|---|---|---|
| none (clean photo) | 60.0 -> 60.0 | 1.00 -> 1.00 | untouched |
| grain, sigma 25 | 20.6 -> 32.0 | 0.34 -> 0.88 | 93% |
| salt & pepper 5% | 17.9 -> 25.5 | 0.38 -> 0.87 | 83% |
| salt & pepper 20% | 11.9 -> 24.1 | 0.12 -> 0.77 | 94% |
| snow specks 8% | 15.4 -> 25.5 | 0.45 -> 0.87 | 90% |
| grain 15 + speckle 5% | 17.2 -> 27.2 | 0.28 -> 0.83 | 90% |
| colour noise | 30.4 -> 37.8 | 0.79 -> 0.97 | 82% |
| multiplicative speckle (radar/ultrasound style) | 19.3 -> 24.3 | 0.43 -> 0.69 | 68% |
| B&W + grain 25 | 20.5 -> 32.0 | 0.38 -> 0.89 | 93% |
| B&W + VERY heavy grain 50 | 14.9 -> 26.7 | 0.17 -> 0.71 | 93% (but SSIM 0.71: fine detail is lost) |
| B&W + salt & pepper 10% | 15.1 -> 24.9 | 0.24 -> 0.82 | 90% |

So most noise types reach about 90%; light speckle (5%), colour noise and signal-dependent speckle fall short. Fast mode is several dB weaker.

**Audio, default strength 0.4.** 48 kHz stereo rock-style music with strong bass (the case that exposed an earlier bug):

| Case | SI-SDR gain | Bass level (0 = untouched) | Treble | Hiss in quiet intro |
|---|---|---|---|---|
| quiet intro, hiss 15 dB below music | +2.3 dB | -0.2 dB | -6.2 dB | -13 dB |
| quiet intro, hiss 10 dB below music | +5.2 dB | -0.4 dB | -4.2 dB | -12 dB |
| continuous music, hiss 15 dB below | +2.0 dB | -0.4 dB | -3.7 dB | |
| continuous music, hiss 10 dB below | +5.2 dB | -0.5 dB | -4.2 dB | |

16 kHz mono music: 56-76% of the added noise energy removed. **Audio does not reach 90%.** Bass is kept at every setting; the cost of
removing more hiss is softer treble, so raise Strength only as far as you need.

## Why very noisy photos still look smooth
At very heavy noise (level ~40 and above) the picture's fine detail is buried under the noise, so the result is smooth and painterly with *any*
method: Studio matches what BM3D gets even when told the true noise level. The AI mode may look more natural; lowering Strength keeps more
texture (with more leftover noise) and raising Keep film grain adds a natural look back.

## Limits
- Test noise is synthetic and the photos/music are bundled samples. Real noise and real recordings will differ. Try your own files.
- Studio takes about 1 minute per megapixel on one CPU core (less on multi-core, and about 3x faster for black-and-white photos).
- Audio assumes steady hiss. Crackle, wow/flutter and clipping are not noise-removal problems.
- "Restore old photo" is classical image processing judged by eye, with no accuracy score. It cannot recover detail that was never captured.
- The interface was tested through jsdom and HTTP calls plus screenshots you sent, not by an automated real browser.

## Tests
```bash
python tests/test_smoke.py
python server.py &  node tests/ui.test.js      # needs `npm i jsdom`; drives the real UI against the live server
```

## Also included
The earlier restoration toolkit: damage simulators, dataset generator (240 sample photo pairs in `data/`), a classical baseline, ControlNet and
SGMSE+ training scripts (written, never run: no GPU was available) and a Gradio demo (`python app.py`).
