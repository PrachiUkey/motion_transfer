# LTX-2 Motion Transfer

Transfer motion from a reference video onto a still image, producing a new short video where the image moves like the reference. Built on top of [Lightricks LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) using the **ICLoraPipeline** with IC-LoRA conditioning.

```
   subject image                  reference video                  output video
   ┌──────────┐                   ┌──────────────┐               ┌─────────────┐
   │   🧑      │   +               │  ↻ motion ↺  │   ───────▶    │  🧑 + motion │
   └──────────┘                   └──────────────┘               └─────────────┘
                                                                   (silent .mp4)
```

---

## How it works (layman's version)

1. **Subject image** — locks in *who/what* appears in the output (the person, clothes, background style).
2. **Reference video** — supplies *how things move* (motion, pose, camera path).
3. **Text prompt** — guides the overall scene description.
4. **The model** — combines all three into a new video, in two stages: a fast low-res pass to lock in motion + composition, then a refinement pass that upsamples to the target resolution.

The output is a silent MP4 (audio is stripped automatically).

---

## Quick start

```bash
# 1. Clone
git clone git@github.com:PrachiUkey/motion_transfer.git
cd motion_transfer

# 2. Install Python deps (pick ONE of the two options below — see "Install" section)
uv sync --frozen && source .venv/bin/activate    # option A — uv (recommended)
# -- or --
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129 \
  && pip install -e packages/ltx-core packages/ltx-pipelines   # option B — pip

# 3. Download model weights (~67 GB). Needs a HuggingFace token; see "Models" section.
python download_models.py

# 4. Generate
python main.py path/to/your_image.png
# → outputs/motion_transfer_<image_stem>.mp4
```

The default reference video is `assets/idle_avatar_15_reverse.mp4` (a 5-second idle-avatar clip). Override it with `--video your_motion.mp4`.

---

## Requirements

| | |
|---|---|
| **Python** | ≥ 3.10 |
| **GPU** | NVIDIA with ≥ 24 GB VRAM (e.g. RTX 4090, A100, H100). FP8 quantization is used to fit the 22B model in 24 GB. |
| **CUDA** | 12.9 (PyTorch wheels are pinned to this) |
| **RAM** | ≥ 32 GB (Gemma 3 text encoder runs on CPU to keep VRAM free) |
| **Disk** | ~70 GB for model weights + ~1 MB per output video |

---

## Install

You can use either **uv** (faster, single command, recommended) or plain **pip**.

### Option A — uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --frozen
source .venv/bin/activate
```

`uv sync --frozen` reads `uv.lock` and installs the exact resolved versions (including the right PyTorch CUDA 12.9 wheel and the local `ltx-core` + `ltx-pipelines` workspace packages).

### Option B — pip

```bash
# Create and activate a venv (recommended)
python3 -m venv .venv && source .venv/bin/activate

# Install all third-party deps, using PyTorch's CUDA 12.9 index for torch wheels
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu129

# Install the two local packages in editable mode
pip install -e packages/ltx-core packages/ltx-pipelines
```

`requirements.txt` is auto-generated from `uv.lock` (`uv export --no-dev --no-emit-workspace`) so the pinned versions stay in sync with the uv flow. The `--extra-index-url` is required so `pip` picks the CUDA 12.9 PyTorch wheel rather than the CPU-only default.

---

## Model weights

`download_models.py` fetches four sets of weights into `./models/`:

| Subdir | File | Source | Size |
|---|---|---|---|
| `distilled/` | `ltx-2.3-22b-distilled.safetensors` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | ~43 GB |
| `upscaler/` | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | ~1 GB |
| `ic-lora/` | `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors` | [Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) | ~0.3 GB |
| `gemma/` | Gemma 3 12B (5 shards + tokenizer) | [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) — **gated** | ~23 GB |

### HuggingFace token (one-time setup)

Gemma 3 is a gated Google model. Before running `download_models.py`:

1. Sign in at https://huggingface.co.
2. Visit https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized, click **"Acknowledge license"**, and submit the form.
3. Create a **Read** token at https://huggingface.co/settings/tokens.

Provide the token any of these ways (script checks in order):
- `export HF_TOKEN=hf_...`
- `export HUGGINGFACE_HUB_TOKEN=hf_...`
- `huggingface-cli login` (caches token in `~/.cache/huggingface/token`)
- Interactive prompt (the script asks if none of the above are set)

The token stays on your machine. It is **not** committed to the repo.

---

## Using `main.py`

```bash
python main.py IMAGE [--video VIDEO] [--prompt PROMPT] [--output PATH] [options]
```

| Flag | Default | Purpose |
|---|---|---|
| `IMAGE` | *(required)* | Subject image (PNG / JPG). Convert paletted/grayscale PNGs to RGB first. |
| `--video` | `assets/idle_avatar_15_reverse.mp4` | Reference motion video. |
| `--prompt` | "A person facing the camera with subtle head and shoulder movement…" | Text prompt for the output. |
| `--output` | `outputs/motion_transfer_<image_stem>.mp4` | Where to write the silent MP4. |
| `--height` / `--width` | `768` / `768` | Output resolution (must be multiples of 64). |
| `--num-frames` | `121` | Frame count (≈ 5 s at 25 fps). |
| `--frame-rate` | `25.0` | Output fps. |
| `--seed` | `42` | Reproducibility. |
| `--lora-strength` | `0.8` | How strongly the IC-LoRA influences output. Try 0.6–1.0. |
| `--video-strength` | `1.0` | How strongly the reference motion drives output. |
| `--image-strength` | `1.0` | How strongly the subject image anchors appearance. |

Internally, `main.py`:
1. Sets `LTX_TEXT_ENCODER_CPU=1` so Gemma runs on CPU (needed on 24 GB GPUs).
2. Invokes `python -m ltx_pipelines.ic_lora` with FP8 quantization.
3. Strips the audio stream losslessly via PyAV (no `ffmpeg` binary required).

---

## Tuning tips

- **Motion too weak** → raise `--video-strength` (try 1.0–1.5) or `--lora-strength`.
- **Motion too rigid / face distorts** → lower `--lora-strength` to 0.5–0.7.
- **Subject drifts from the image** → raise `--image-strength`, keep the image anchored at frame 0.
- **Out of VRAM** → lower `--height` / `--width` to 512, or `--num-frames` to 81. FP8 quantization is already on.
- **Faster preview** → temporarily set `--num-frames 49 --height 512 --width 512` for a 2-second 512² draft.

---

## Repo layout

```
.
├── main.py                            # generate a single silent video
├── download_models.py                 # fetch all weights into models/
├── assets/
│   └── idle_avatar_15_reverse.mp4     # default reference motion
├── models/                            # downloaded weights (gitignored)
├── outputs/                           # generated MP4s (gitignored)
└── packages/
    ├── ltx-core/                      # LTX-2 model implementation
    └── ltx-pipelines/                 # ICLoraPipeline + shared utils
```

---

## Troubleshooting

- **`GatedRepoError: 403`** during Gemma download → your account hasn't accepted the Gemma 3 license yet. See the [HF token](#huggingface-token-one-time-setup) section.
- **`ValueError: Expected numpy array with ndim 3`** during run → your input image is grayscale or palette-indexed. Convert to RGB first:
  ```python
  from PIL import Image; Image.open("img.png").convert("RGB").save("img.png")
  ```
- **`CUBLAS_STATUS_ALLOC_FAILED`** → out of GPU memory. Confirm no other processes are holding the card with `nvidia-smi`. Try lower resolution.

---

## License & credits

- This project builds on [Lightricks LTX-2](https://huggingface.co/Lightricks/LTX-2.3). Source code in `packages/` is governed by the [LTX-2 Community License Agreement](LICENSE).
- The Gemma 3 text encoder is © Google, used under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- Pipeline: `ICLoraPipeline` from [packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py).
