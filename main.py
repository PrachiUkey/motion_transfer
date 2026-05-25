#!/usr/bin/env python3
"""Generate a silent motion-transferred video from a single image.

The image supplies the subject/appearance; a reference video supplies the
motion. Output is an MP4 with the audio stream stripped.

Examples
--------
    python main.py assets/images/avatar_11.png
    python main.py path/to/photo.jpg --video assets/idle_avatar_15_reverse.mp4
    python main.py path/to/photo.jpg --prompt "a person nodding"

Models are expected under ./models/ (see BUILD.md for download instructions).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

DEFAULT_VIDEO = ROOT / "assets" / "idle_avatar_15_reverse.mp4"
DEFAULT_PROMPT = (
    "A person facing the camera with subtle head and shoulder movement, "
    "professional headshot lighting, neutral background, photorealistic"
)

DISTILLED = MODELS / "distilled" / "ltx-2.3-22b-distilled.safetensors"
UPSAMPLER = MODELS / "upscaler" / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA = MODELS / "gemma"
LORA = MODELS / "ic-lora" / "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", help="Path to the conditioning image (PNG/JPG).")
    p.add_argument("--video", default=str(DEFAULT_VIDEO), help=f"Reference motion video. Default: {DEFAULT_VIDEO.relative_to(ROOT)}")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt for the output.")
    p.add_argument("--output", default=None, help="Output mp4 path. Default: outputs/motion_transfer_<image_stem>.mp4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--frame-rate", type=float, default=25.0)
    p.add_argument("--lora-strength", type=float, default=0.8)
    p.add_argument("--video-strength", type=float, default=1.0)
    p.add_argument("--image-strength", type=float, default=1.0)
    return p.parse_args()


def require(path: Path, label: str) -> None:
    if not path.exists():
        sys.exit(f"error: missing {label}: {path}\nSee BUILD.md for download instructions.")


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).resolve()
    video_path = Path(args.video).resolve()
    require(image_path, "input image")
    require(video_path, "reference video")
    for p, label in [(DISTILLED, "distilled checkpoint"), (UPSAMPLER, "spatial upsampler"),
                     (GEMMA, "Gemma encoder dir"), (LORA, "IC-LoRA weights")]:
        require(p, label)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).resolve() if args.output else OUTPUTS / f"motion_transfer_{image_path.stem}.mp4"
    tmp_path = output_path.with_suffix(".with-audio.mp4")

    env = os.environ.copy()
    env["LTX_TEXT_ENCODER_CPU"] = "1"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable, "-m", "ltx_pipelines.ic_lora",
        "--distilled-checkpoint-path", str(DISTILLED),
        "--spatial-upsampler-path", str(UPSAMPLER),
        "--gemma-root", str(GEMMA),
        "--lora", str(LORA), str(args.lora_strength),
        "--prompt", args.prompt,
        "--image", str(image_path), "0", str(args.image_strength), "33",
        "--video-conditioning", str(video_path), str(args.video_strength),
        "--num-frames", str(args.num_frames),
        "--frame-rate", str(args.frame_rate),
        "--height", str(args.height),
        "--width", str(args.width),
        "--seed", str(args.seed),
        "--quantization", "fp8-cast",
        "--output-path", str(tmp_path),
    ]

    print(f"[main] Generating: {image_path.name} -> {output_path.name}")
    subprocess.run(cmd, env=env, check=True)

    print("[main] Stripping audio...")
    strip_audio(tmp_path, output_path)
    tmp_path.unlink(missing_ok=True)
    print(f"[main] Saved: {output_path}")


def strip_audio(src: Path, dst: Path) -> None:
    """Re-mux *src* into *dst* keeping only the video stream (no re-encode)."""
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        in_video = inp.streams.video[0]
        out_video = out.add_stream_from_template(in_video)
        for packet in inp.demux(in_video):
            if packet.dts is None:
                continue
            packet.stream = out_video
            out.mux(packet)


if __name__ == "__main__":
    main()
