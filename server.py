"""FastAPI server that exposes the motion-transfer pipeline.

Run locally:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000

UI:  open http://localhost:8000/ in a browser.
API: POST /generate (multipart), then GET /jobs/{id} to poll, then GET /jobs/{id}/result for the mp4.
"""

import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
STATIC = ROOT / "static"
DEFAULT_VIDEO = ROOT / "assets" / "idle_avatar_15_reverse.mp4"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LTX-2 Motion Transfer")

# In-memory job registry. Persists only for the server's lifetime.
JOBS: dict[str, dict] = {}
_JOB_LOCK = threading.Lock()
# Serialize generations: the pipeline pins the whole GPU, so we run one at a time.
_GPU_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set(job_id: str, **fields) -> None:
    with _JOB_LOCK:
        JOBS[job_id].update(fields)


def run_job(job_id: str, image_path: Path, video_path: Path, prompt: str | None) -> None:
    """Run main.py as a subprocess and capture status."""
    _set(job_id, status="waiting_for_gpu")
    with _GPU_LOCK:
        _set(job_id, status="running", started_at=_now())
        output_path = OUTPUTS / f"api_{job_id}.mp4"
        cmd = [
            sys.executable, str(ROOT / "main.py"),
            str(image_path),
            "--video", str(video_path),
            "--output", str(output_path),
        ]
        if prompt:
            cmd.extend(["--prompt", prompt])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0 and output_path.exists():
                _set(job_id, status="done", finished_at=_now(),
                     result=str(output_path.relative_to(ROOT)))
            else:
                _set(job_id, status="failed", finished_at=_now(),
                     error=(result.stderr or result.stdout or "")[-2000:])
        except subprocess.TimeoutExpired:
            _set(job_id, status="failed", finished_at=_now(), error="timed out after 30 min")
        except Exception as e:
            _set(job_id, status="failed", finished_at=_now(), error=repr(e))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/generate")
async def generate(
    image: UploadFile = File(..., description="Subject image (PNG or JPG)"),
    video: UploadFile | None = File(None, description="Reference motion video (optional; defaults to assets/idle_avatar_15_reverse.mp4)"),
    prompt: str | None = Form(None, description="Text prompt (optional; main.py has a sensible default)"),
):
    job_id = uuid.uuid4().hex[:8]

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "empty image upload")
    image_path = UPLOADS / f"{job_id}_image_{image.filename}"
    image_path.write_bytes(image_bytes)

    if video is not None and video.filename:
        video_bytes = await video.read()
        if not video_bytes:
            raise HTTPException(400, "empty video upload")
        video_path = UPLOADS / f"{job_id}_video_{video.filename}"
        video_path.write_bytes(video_bytes)
        used_default = False
    else:
        video_path = DEFAULT_VIDEO
        used_default = True

    with _JOB_LOCK:
        JOBS[job_id] = {
            "status": "pending",
            "image": image.filename,
            "video": video_path.name,
            "used_default_video": used_default,
            "prompt": prompt,
            "submitted_at": _now(),
        }

    threading.Thread(target=run_job, args=(job_id, image_path, video_path, prompt), daemon=True).start()
    return {"job_id": job_id, **JOBS[job_id]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    return JOBS[job_id]


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    if JOBS[job_id].get("status") != "done":
        raise HTTPException(409, f"job not done (status={JOBS[job_id].get('status')})")
    return FileResponse(ROOT / JOBS[job_id]["result"], media_type="video/mp4", filename=f"motion_transfer_{job_id}.mp4")


@app.get("/jobs")
def list_jobs():
    return JOBS


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    bar = "=" * 60
    display_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"\n{bar}\n  LTX-2 Motion Transfer server\n"
          f"  ➜ UI:  http://{display_host}:{port}/\n"
          f"  ➜ API: http://{display_host}:{port}/generate (POST multipart)\n"
          f"  Bound to {host}:{port}.  Ctrl-C to quit.\n{bar}\n", flush=True)

    uvicorn.run(app, host=host, port=port, log_level="info")
