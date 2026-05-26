# Pipeline Flow

What happens to your image and reference video, step by step, from `python main.py photo.png` to a finished silent MP4. Every section links to the line in source where it happens.

---

## 1. Bird's-eye diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   photo.png ──┐                                                              │
│               │                                                              │
│               ▼                                                              │
│      VideoEncoder (VAE)  ─►  image latent ─┐                                 │
│                                            │                                 │
│   reference.mp4 ──►  VideoEncoder (VAE) ─► reference latent (IC-LoRA cond)─┐ │
│                                                                            │ │
│   "a person smiling" ─► Gemma 3 12B  ─►  hidden states ─► EmbeddingsProc ─┐│ │
│   (text prompt)         (text encoder)                  (LTX side)        ▼▼ │
│                                                                              │
│   ┌──────── STAGE 1 (384x384) ────────┐    ┌──────── STAGE 2 (768x768) ───┐  │
│   │                                   │    │                              │  │
│   │ noise ──► DiT 22B + IC-LoRA  ──┐  │    │ stage_1_latent               │  │
│   │          (distilled, FP8)     │  │    │     │                        │  │
│   │   8x Euler steps              ▼  │    │     ▼                        │  │
│   │                       stage_1_lat│    │ spatial upsampler (2x)       │  │
│   │                                   │    │     │                        │  │
│   │                                   │───►│     ▼                        │  │
│   │                                   │    │ DiT 22B (distilled, no LoRA) │  │
│   │                                   │    │   3x Euler refine steps      │  │
│   │                                   │    │     │                        │  │
│   │                                   │    │     ▼                        │  │
│   │                                   │    │   final_latent               │  │
│   └───────────────────────────────────┘    └──────┬───────────────────────┘  │
│                                                   │                          │
│                                                   ▼                          │
│                            Video VAE decoder ─► RGB frames (121 x 768 x 768) │
│                            Audio VAE + Vocoder ─► waveform (discarded later) │
│                                                   │                          │
│                                                   ▼                          │
│                            PyAV encoder ─► motion_transfer.with-audio.mp4    │
│                                                   │                          │
│                                                   ▼                          │
│                            PyAV remux  (-an) ─► motion_transfer.mp4 (silent) │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-step trace

### Step 0 — `main.py` orchestration
[main.py](main.py)

`main.py` is a thin shell. It validates inputs, sets two env vars, and shells out to the actual pipeline:

| Env var | Why |
|---|---|
| `LTX_TEXT_ENCODER_CPU=1` | Tells our patched `ModelLedger.text_encoder()` to load Gemma 3 on CPU so it doesn't eat 24 GB of VRAM. |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | Reduces VRAM fragmentation when the 22B model is loaded under FP8 quantization. |

The actual invocation is `python -m ltx_pipelines.ic_lora ...` with `--quantization fp8-cast`. After the subprocess returns, `main.py` calls `strip_audio()` to remux the file with the audio stream dropped.

### Step 1 — Prompt → embedding (Gemma 3, on CPU)
[packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py:171](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py#L171)

`encode_prompts()` (in `utils/helpers.py`) does three things in order:

1. **Load Gemma 3 12B text encoder** onto CPU via `ModelLedger.text_encoder()` (our patched branch).
2. **Tokenize** the prompt string and run a forward pass → `(hidden_states, attention_mask)`. This is the most CPU-bound part — typically 5–30 seconds.
3. **Move hidden states to GPU** (also our patch), free Gemma from RAM, then load the **LTX embeddings processor** ([utils/helpers.py:80](packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py#L80)) which projects Gemma's outputs into the latent space the DiT expects.

Result: `ctx_p.video_encoding` (text guidance for the video branch) and `ctx_p.audio_encoding` (for the audio branch).

### Step 2 — Image → latent (Video VAE encoder)
[ic_lora.py:191](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py#L191) → [utils/helpers.py:89 (`combined_image_conditionings`)](packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py#L89)

For each `--image` you pass (just one by default):

1. **Load image** via PIL, resize to stage-1 dimensions (here `384 × 384`), convert to RGB, normalize to `[-1, 1]`.
2. **Re-encode it as a 1-frame video** through PyAV at the requested CRF (default 33). This matches how the training data was processed.
3. **Run `video_encoder(image)`** → a small latent tensor that represents the image in the VAE's latent space.
4. **Wrap as a `VideoConditionByLatentIndex`** with `latent_idx=0` and the requested `strength=1.0`. This tells the diffuser "replace the latent at output frame 0 with my image's latent."

### Step 3 — Reference video → latent (Video VAE encoder)
[ic_lora.py:_create_conditionings (around line 320)](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py#L320)

The same `video_encoder` is reused. The reference video:

1. **Loaded via PyAV** ([utils/media_io.py](packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py) `load_video_conditioning`), capped to `num_frames` (121), resized to the stage-1 ref dimensions.
2. **Optionally downscaled further** by `reference_downscale_factor` — the Motion-Track-Control LoRA was trained at half-resolution references (the `ref0.5` in its filename), so the LoRA's metadata says "feed me ref at 0.5×". The pipeline auto-reads this and resizes the reference video to `ref_height × ref_width`.
3. **Encoded** through `video_encoder(video)` → a latent tensor of shape `(B, C, T_lat, H_lat, W_lat)`.
4. **Wrapped as a `VideoConditionByReferenceLatent`** with `strength=1.0` and the downscale_factor. This is the **IC-LoRA conditioning hook** — at every DiT block that has IC-LoRA weights, the model attends across both the noisy latents *and* this reference latent. That's how the motion is injected.

### Step 4 — Stage 1: low-resolution denoising
[ic_lora.py:202–234](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py#L202-L234)

1. **Free the video encoder** to reclaim VRAM.
2. **Load the LTX-2.3 22B distilled transformer** (DiT) with FP8 quantization (`--quantization fp8-cast` casts the safetensors weights to FP8 E4M3 at load time, halving VRAM relative to bf16). The Motion-Track-Control IC-LoRA is applied to the transformer here.
3. **Sample initial Gaussian noise** at the stage-1 latent shape derived from `384 × 384 × 121 frames`.
4. **Run `denoise_audio_video()`** with the distilled sigma schedule (`DISTILLED_SIGMA_VALUES`, 8 values) for 8 Euler diffusion steps. Each step:
   - Concatenate the noisy latent with the IC-LoRA reference latent + image conditioning.
   - Forward pass through DiT (with the text/audio embeddings as cross-attention).
   - Compute the noise prediction → Euler update → less-noisy latent.
   - Audio latent is denoised in the same call (synced timesteps).
5. **Free the transformer** at the end of stage 1.

Output of stage 1: `stage_1_latent` (low-res video latent) + `stage_1_audio_latent`.

### Step 5 — Stage 2: upsample + refine
[ic_lora.py:249–301](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py#L249-L301)

1. **Reload the VAE encoder** briefly.
2. **Spatial upsampler** ([upsampler.py](packages/ltx-core/src/ltx_core/model/upsampler.py)) takes the stage-1 latent and produces a `768 × 768` latent (2× spatial upscale, still in latent space).
3. **Load the LTX-2.3 22B transformer again, *without* IC-LoRA** (stage 2 uses the same base checkpoint but with `loras=[]` per [ic_lora.py:88](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py#L88) — IC-LoRA has done its job, now we just refine textures).
4. **Re-encode the original image at full resolution** (768×768) so the subject's appearance is anchored cleanly during refinement.
5. **Add a small amount of noise** to the upscaled latent and run **3 Euler refinement steps** with `STAGE_2_DISTILLED_SIGMA_VALUES` (a truncated, low-noise schedule).
6. **Free the transformer.**

### Step 6 — Latent → pixel video (Video VAE decoder)
[ic_lora.py:307](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py#L307)

Load the **video VAE decoder** (same VAE family as the encoder, decoder half), call `vae_decode_video(latent, decoder, tiling_config, generator)`. The decoder is tiled spatio-temporally so memory stays manageable. Output: an **iterator** of pixel-space frames at 768 × 768.

### Step 7 — Latent → audio waveform (Audio VAE + Vocoder)
Same call as step 6: `vae_decode_audio(audio_latent, audio_decoder, vocoder)`. Goes through the audio VAE decoder, then a small **vocoder** that converts the spectrogram-style latent to a 1D waveform.

### Step 8 — Mux to MP4
[utils/media_io.py](packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py) `encode_video()`

PyAV opens the output path, adds a video stream (H.264) and an audio stream (AAC), iterates the frame iterator from step 6, encodes each frame, interleaves the audio packets, and closes the file.

At this point the temp file is `motion_transfer_<stem>.with-audio.mp4`.

### Step 9 — Strip audio
[main.py:`strip_audio()`](main.py)

`main.py` opens the temp file with PyAV again, demuxes only the video stream, and re-muxes packet-by-packet into the final `outputs/motion_transfer_<stem>.mp4`. No re-encode → lossless and very fast (≈ 0.3 s). The temp file is then deleted.

---

## 3. Models touched, in order

| # | Model | Where | Purpose | Lives on |
|--|---|---|---|---|
| 1 | **Gemma 3 12B** | `models/gemma/` | Text → hidden states | **CPU** (our patch) |
| 2 | **LTX embeddings processor** | inside the 22B checkpoint | Project Gemma hidden states into LTX latent dim | GPU |
| 3 | **Video VAE encoder** | inside the 22B checkpoint | Image / reference video → latent | GPU |
| 4 | **LTX-2.3 22B DiT** (+ Motion-Track IC-LoRA) | `models/distilled/` + `models/ic-lora/` | Stage 1 denoising | GPU (FP8) |
| 5 | **Spatial upsampler** | `models/upscaler/` | 2× latent upsample between stages | GPU |
| 6 | **LTX-2.3 22B DiT** (no LoRA) | `models/distilled/` (reloaded) | Stage 2 refinement | GPU (FP8) |
| 7 | **Video VAE decoder** | inside the 22B checkpoint | Latent → RGB frames | GPU |
| 8 | **Audio VAE decoder + Vocoder** | inside the 22B checkpoint | Latent → audio waveform | GPU |

Note: the "inside the 22B checkpoint" entries (VAE encoder/decoder, audio decoder, vocoder, embeddings processor) are different sub-networks all packed inside the single `ltx-2.3-22b-distilled.safetensors` file. `ModelLedger` builds each one selectively using filename-key filters defined in [packages/ltx-core/src/ltx_core/model/](packages/ltx-core/src/ltx_core/model/).

---

## 4. Memory choreography (why FP8 + Gemma-on-CPU)

Peak-VRAM moments and how they're kept under 24 GB on an RTX 4090:

| Phase | Naive bf16 VRAM | What we do | Actual VRAM |
|---|---|---|---|
| Gemma 3 12B loaded | ~24 GB (alone) | Load on CPU instead (`LTX_TEXT_ENCODER_CPU=1`) | ~0 GB on GPU |
| DiT 22B loaded | ~44 GB | FP8 cast at load time (`--quantization fp8-cast`) | ~22 GB |
| DiT 22B + reference latent + image latent + noise + activations | overflow | Free encoder before loading DiT; `expandable_segments` reduces fragmentation | ~23 GB peak |
| Spatial upsampler | small | – | <1 GB |
| Stage 2 DiT reload | ~22 GB | DiT from stage 1 is freed first | ~22 GB |

`ModelLedger` is built so each model is **lazily constructed when its method is called**, and the caller frees it with `del` + `cleanup_memory()` (see [utils/helpers.py:`cleanup_memory()`](packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py)) as soon as it's no longer needed.

---

## 5. File map

| Concern | File |
|---|---|
| User entry point | [main.py](main.py) |
| Pipeline orchestration | [packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py) |
| `encode_prompts`, `combined_image_conditionings`, `denoise_audio_video`, `cleanup_memory` | [packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py](packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py) |
| Lazy model loaders + LoRA application + quantization wiring | [packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py](packages/ltx-pipelines/src/ltx_pipelines/utils/model_ledger.py) |
| Image / video I/O (PyAV) | [packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py](packages/ltx-pipelines/src/ltx_pipelines/utils/media_io.py) |
| Sigma schedules + default LTX-2.3 params | [packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py](packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py) |
| CLI argument parser | [packages/ltx-pipelines/src/ltx_pipelines/utils/args.py](packages/ltx-pipelines/src/ltx_pipelines/utils/args.py) |
| Gemma encoder loader | [packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py](packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py) |
| Video VAE | [packages/ltx-core/src/ltx_core/model/video_vae.py](packages/ltx-core/src/ltx_core/model/video_vae.py) |
| Audio VAE + vocoder | [packages/ltx-core/src/ltx_core/model/audio_vae.py](packages/ltx-core/src/ltx_core/model/audio_vae.py) |
| Spatial upsampler | [packages/ltx-core/src/ltx_core/model/upsampler.py](packages/ltx-core/src/ltx_core/model/upsampler.py) |
| DiT transformer | [packages/ltx-core/src/ltx_core/model/transformer/](packages/ltx-core/src/ltx_core/model/transformer/) |
| Conditioning types (`VideoConditionByReferenceLatent`, etc.) | [packages/ltx-core/src/ltx_core/conditioning/](packages/ltx-core/src/ltx_core/conditioning/) |
