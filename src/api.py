"""FastAPI layer: submit a search, watch it over SSE, fetch the frame.

    POST /api/find                -> 202 + job_id
    GET  /api/jobs/{id}           -> current state (polling alternative)
    GET  /api/jobs/{id}/events    -> Server-Sent Events until terminal
    GET  /api/frame/{id}.png      -> the extracted PNG
    GET  /api/health              -> liveness + ffmpeg presence
    GET  /                        -> the frontend

The pipeline runs on a worker thread because it is synchronous and long; the
event loop only ever reads job state.

    python -m uvicorn src.api:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src import config
from src.core import ffmpeg
from src.core.normalize import normalize_text
from src.core.resolve import _validate_url
from src.errors import InvalidInputError, DialogueFrameError, ResolveError, UnsupportedMediaError
from src.jobs import InMemoryJobStore, Job
from src.service import find_dialogue

app = FastAPI(
    title="DialogueFrame",
    description="Find the video frame where a line of dialogue is spoken.",
    version="0.1.0",
)

# The single place the backend is chosen; a shared store later means changing
# this line and nothing else.
STORE = InMemoryJobStore()



class FindRequest(BaseModel):
    """Pydantic enforces the shape; the semantic checks reuse the same
    functions the CLI uses, so both front ends reject identical input."""

    url: str = Field(..., description="Video URL (a single video, not a playlist)")
    text: str = Field(..., description="The line of dialogue to locate")
    force: bool = Field(False, description="Ignore caches and recompute from scratch")


def http_status_for(exc: Exception) -> int:
    """400 malformed request | 422 media we refuse | 502 upstream | 500 bug."""
    if isinstance(exc, InvalidInputError):
        return 400
    if isinstance(exc, UnsupportedMediaError):
        return 422
    if isinstance(exc, ResolveError):
        return 502
    return 500


def error_body(exc: Exception) -> dict[str, Any]:
    """One error shape everywhere, so the frontend has a single render path."""
    return {"status": "error", "error": str(exc), "error_type": type(exc).__name__}


@app.exception_handler(DialogueFrameError)
async def dialogueframe_error_handler(request: Request, exc: DialogueFrameError) -> JSONResponse:
    """A DialogueFrameError raised anywhere reaches the client as a readable message
    rather than a 500 with a traceback."""
    return JSONResponse(status_code=http_status_for(exc), content=error_body(exc))


def validate_find_request(payload: FindRequest) -> tuple[str, str]:
    """Reject bad input at submit time.

    Without this the client gets 202 Accepted, waits, polls, and only then
    learns it made a typo.
    """
    url = (payload.url or "").strip()
    text = (payload.text or "").strip()

    try:
        _validate_url(url)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=error_body(exc)) from exc

    if not text:
        exc = InvalidInputError("Dialogue text is required and must not be empty.")
        raise HTTPException(status_code=400, detail=error_body(exc)) from exc

    # Same rule matching.py enforces, checked before any network work.
    normalized = normalize_text(text)
    if len(normalized) < config.MIN_QUERY_CHARS:
        exc = InvalidInputError(
            f"Dialogue text {text!r} normalizes to {normalized!r}, shorter than the "
            f"{config.MIN_QUERY_CHARS}-character minimum."
        )
        raise HTTPException(status_code=400, detail=error_body(exc)) from exc

    return url, text


def _run_job(job: Job, force: bool) -> None:
    """Worker thread body.

    Records BOTH expected failures and genuine bugs on the job: a crashed
    worker leaving the job stuck at "running" forever would be worse for the
    client than an honest error.
    """
    job.mark_running()
    try:
        # Job.record_progress matches ProgressCallback, so the pipeline streams
        # straight into the job's event history.
        result = find_dialogue(job.url, job.text, force=force,
                               progress_callback=job.record_progress)
        job.mark_done(result.to_dict())
    except DialogueFrameError as exc:
        job.mark_error(str(exc), type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 - never leave a job stuck running
        job.mark_error(f"Unexpected {type(exc).__name__}: {exc}", type(exc).__name__)


def _require_job(job_id: str) -> Job:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={
            "status": "error",
            "error": f"No job with id {job_id!r}. It may have expired.",
            "error_type": "JobNotFound",
        })
    return job


@app.post("/api/find", status_code=202)
async def create_find_job(payload: FindRequest) -> dict[str, Any]:
    """Start a search and return its job_id immediately."""
    url, text = validate_find_request(payload)
    job = STORE.create(url, text)
    threading.Thread(
        target=_run_job, args=(job, payload.force),
        name=f"dialogueframe-job-{job.job_id[:8]}",
        daemon=True,  # never block interpreter shutdown on an in-flight job
    ).start()
    return {
        "job_id": job.job_id, "status": job.status, "url": url, "text": text,
        "events_url": f"/api/jobs/{job.job_id}/events",
        "status_url": f"/api/jobs/{job.job_id}",
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    """Returns 200 even for a failed job -- reading the job succeeded; the
    failure is in the payload."""
    return _require_job(job_id).snapshot()


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request) -> StreamingResponse:
    """Stream progress as SSE.

        event: progress   {seq, stage, percent, message}
        event: done       the job snapshot including `result`
        event: error      {error, error_type}
        : keepalive       during long silences

    Polling a lock-guarded list avoids cross-thread asyncio primitives entirely
    -- the pipeline runs on a worker thread while this is served on the event
    loop. Replays from seq 0, so a client connecting late still sees the bar
    fill from the beginning.
    """
    job = _require_job(job_id)

    async def event_stream():
        next_seq = 0
        last_send = time.monotonic()
        while True:
            if await request.is_disconnected():
                return

            for event in job.events_since(next_seq):
                next_seq = event.seq + 1
                yield f"event: progress\ndata: {json.dumps(event.to_dict())}\n\n"
                last_send = time.monotonic()

            if job.is_terminal:
                snapshot = job.snapshot()
                if job.status == "error":
                    payload = {"error": snapshot["error"], "error_type": snapshot["error_type"]}
                    yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                else:
                    yield f"event: done\ndata: {json.dumps(snapshot)}\n\n"
                return

            # ASR can run for minutes between events on a long video.
            if time.monotonic() - last_send > config.SSE_KEEPALIVE_SECONDS:
                yield ": keepalive\n\n"
                last_send = time.monotonic()
            await asyncio.sleep(config.SSE_POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Tells nginx not to buffer, which would defeat streaming entirely.
        "X-Accel-Buffering": "no",
    })


@app.get("/api/frame/{job_id}.png")
async def get_frame(job_id: str) -> FileResponse:
    """Serve the PNG for a finished job.

    The path comes from the job's own server-generated result, then is verified
    to sit inside OUTPUT_DIR anyway -- so a bug upstream cannot become an
    arbitrary file read.
    """
    job = _require_job(job_id)
    snapshot = job.snapshot()

    if job.status != "done" or not snapshot.get("result"):
        raise HTTPException(status_code=409, detail={
            "status": "error",
            "error": f"Job {job_id} is {job.status}, so no frame is available yet.",
            "error_type": "JobNotFinished",
        })

    image_path = (snapshot["result"] or {}).get("image_path")
    if not image_path:
        raise HTTPException(status_code=404, detail={
            "status": "error",
            "error": "This job produced no frame (the dialogue was not found).",
            "error_type": "NoFrame",
        })

    path = Path(image_path).resolve()
    try:
        path.relative_to(config.OUTPUT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail={
            "status": "error",
            "error": "Refusing to serve a file outside the data directory.",
            "error_type": "ForbiddenPath",
        }) from exc

    if not path.exists():
        raise HTTPException(status_code=404, detail={
            "status": "error",
            "error": f"The extracted frame is no longer on disk: {path.name}",
            "error_type": "FrameMissing",
        })
    return FileResponse(path, media_type="image/png")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Includes ffmpeg presence: otherwise the server starts happily without it
    on PATH and fails on the first real request."""
    tools: dict[str, Any] = {}
    for name in ("ffmpeg", "ffprobe"):
        try:
            tools[name] = ffmpeg.binary(name)
        except DialogueFrameError as exc:
            tools[name] = f"MISSING: {exc}"
    return {
        "status": "ok",
        "version": app.version,
        "asr_model": config.ASR_MODEL,
        "asr_device": f"{config.ASR_DEVICE}/{config.ASR_COMPUTE_TYPE}",
        "index_version": config.INDEX_VERSION,
        "output_dir": str(config.OUTPUT_DIR),
        "tools": tools,
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = config.ASSETS_DIR / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DialogueFrame API</h1><p>Frontend not installed. See /docs.</p>")
