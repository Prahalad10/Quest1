"""FastAPI backend: run the dialogue search as a background job over HTTP.

WHY JOBS RATHER THAN A PLAIN REQUEST
    Indexing a video takes minutes on first use. A synchronous endpoint would
    exceed browser and proxy timeouts and show the user nothing while it worked.
    So POST /api/find starts a worker thread and returns a job_id at once; the
    browser watches progress on the SSE endpoint (or polls, if it prefers).

ENDPOINTS
    POST /api/find                  {url, text} -> 202 {job_id, ...}
    GET  /api/jobs/{job_id}         current stage/percent/message + result
    GET  /api/jobs/{job_id}/events  same progress as an SSE stream
    GET  /api/frame/{job_id}.png    the extracted frame image
    GET  /api/health                liveness + dependency check
    GET  /                          the frontend (Step 9), if present

ERROR HANDLING
    Input problems are caught SYNCHRONOUSLY in POST /api/find and returned as
    4xx with a readable message -- no job is created for a URL that cannot work.
    Failures that only surface during processing are recorded on the job and
    reported through the job endpoints, since the request has long since
    returned. Either way the client sees a sentence, never a stack trace.

RUN IT
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import config
from app.core.resolve import _validate_url
from app.errors import (
    InvalidInputError,
    Quest1Error,
    ResolveError,
    UnsupportedMediaError,
)
from app.jobs import InMemoryJobStore, Job, JobStore
from app.service import find_dialogue

# The single place the backend is chosen. Swapping in a shared store later means
# changing this line and nothing else.
STORE: JobStore = InMemoryJobStore()

# Directory holding the static frontend added in Step 9.
WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(
    title="Quest1",
    description="Find the video frame where a line of dialogue is spoken.",
    version="0.1.0",
)


# --------------------------------------------------------------------------- #
# Request/response models
# --------------------------------------------------------------------------- #

class FindRequest(BaseModel):
    """Body of POST /api/find.

    Pydantic enforces the shape (both fields present, both strings); the
    semantic checks -- is this a usable URL, is this query long enough -- are
    done by the same functions the CLI uses, so both front ends agree.
    """

    url: str = Field(..., description="Video URL (a single video, not a playlist)")
    text: str = Field(..., description="The line of dialogue to locate")
    force: bool = Field(False, description="Ignore caches and recompute from scratch")


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #

def http_status_for(exc: Exception) -> int:
    """Map a Quest1Error to the HTTP status that describes it honestly.

    400 the caller sent something malformed
    422 the request was well-formed but the media is one we refuse
    502 an upstream failure (yt-dlp could not reach or parse the site)
    500 anything else

    USED BY: validate_find_request and the job error reporting below.
    """
    if isinstance(exc, InvalidInputError):
        return 400
    if isinstance(exc, UnsupportedMediaError):
        return 422
    if isinstance(exc, ResolveError):
        return 502
    return 500


def error_body(exc: Exception) -> dict[str, Any]:
    """Uniform JSON error shape.

    Every error the client can receive looks the same, so the frontend has one
    rendering path rather than several.

    USED BY: the exception handlers and the 4xx responses below.
    """
    return {
        "status": "error",
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


@app.exception_handler(Quest1Error)
async def quest1_error_handler(request: Request, exc: Quest1Error) -> JSONResponse:
    """Turn any uncaught Quest1Error into a clean JSON response.

    A safety net: the endpoints below catch these explicitly, but this ensures
    that a Quest1Error raised anywhere still reaches the client as a readable
    message rather than a 500 with a traceback.
    """
    return JSONResponse(status_code=http_status_for(exc), content=error_body(exc))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_find_request(payload: FindRequest) -> tuple[str, str]:
    """Check the URL and text BEFORE creating a job, raising HTTPException on failure.

    WHY SYNCHRONOUSLY: a malformed URL is knowable instantly. Creating a job for
    it would mean the client gets 202 Accepted, waits, polls, and only then
    learns it made a typo. Failing at submit time is both faster and clearer.

    Deliberately reuses resolve._validate_url and the same minimum-length rule
    as matching.py, so the API and the CLI reject exactly the same inputs.

    USED BY: POST /api/find.
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

    # Cheap pre-check of the same rule matching.py enforces, so an obviously
    # useless query is refused before any network work happens.
    from app.core.normalize import normalize_text

    normalized = normalize_text(text)
    if not normalized:
        exc = InvalidInputError(
            f"Dialogue text {text!r} contains no matchable characters."
        )
        raise HTTPException(status_code=400, detail=error_body(exc)) from exc
    if len(normalized) < config.MIN_QUERY_CHARS:
        exc = InvalidInputError(
            f"Dialogue text {text!r} normalizes to {normalized!r}, shorter than the "
            f"{config.MIN_QUERY_CHARS}-character minimum. Such a short query would "
            f"match almost anywhere and its timestamp would be meaningless."
        )
        raise HTTPException(status_code=400, detail=error_body(exc)) from exc

    return url, text


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def _run_job(job: Job, force: bool) -> None:
    """Run the pipeline for one job. THE BODY OF THE WORKER THREAD.

    Catches Quest1Error (expected failures, message shown to the user) and any
    other exception (a genuine bug) separately, but records both on the job --
    a crashed worker that left the job stuck at "running" forever would be far
    worse for the client than an honest error.

    USED BY: POST /api/find, via threading.Thread.
    """
    job.mark_running()
    try:
        result = find_dialogue(
            job.url,
            job.text,
            force=force,
            # Job.record_progress matches the ProgressCallback signature, so the
            # pipeline streams straight into the job's event history.
            progress_callback=job.record_progress,
        )
        job.mark_done(result.to_dict())
    except Quest1Error as exc:
        job.mark_error(str(exc), type(exc).__name__)
    except Exception as exc:  # noqa: BLE001 - never leave a job stuck running
        job.mark_error(
            f"Unexpected {type(exc).__name__}: {exc}",
            type(exc).__name__,
        )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.post("/api/find", status_code=202)
async def create_find_job(payload: FindRequest) -> dict[str, Any]:
    """Start a dialogue search and return its job_id immediately.

    Returns 202 Accepted, not 200: the work has been accepted, not completed.
    The client then watches /api/jobs/{job_id}/events.

    Raises 400/422 synchronously for input that cannot possibly work.
    """
    url, text = validate_find_request(payload)

    job = STORE.create(url, text)
    thread = threading.Thread(
        target=_run_job,
        args=(job, payload.force),
        name=f"quest1-job-{job.job_id[:8]}",
        daemon=True,   # never block interpreter shutdown on an in-flight job
    )
    thread.start()

    return {
        "job_id": job.job_id,
        "status": job.status,
        "url": url,
        "text": text,
        "events_url": f"/api/jobs/{job.job_id}/events",
        "status_url": f"/api/jobs/{job.job_id}",
    }


def _require_job(job_id: str) -> Job:
    """Fetch a job or raise a clean 404.

    USED BY: every job-scoped endpoint below.
    """
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "error": f"No job with id {job_id!r}. It may have expired.",
                "error_type": "JobNotFound",
            },
        )
    return job


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    """Current state of one job, including the full result once finished.

    The polling alternative to the SSE stream. Returns 200 even for a failed
    job -- the request to READ the job succeeded; the failure is in the payload.
    """
    return _require_job(job_id).snapshot()


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request) -> StreamingResponse:
    """Stream this job's progress as Server-Sent Events.

    STREAM SHAPE
        event: progress   one per pipeline event, data = {seq, stage, percent, message}
        event: done       final, data = the job snapshot including `result`
        event: error      final, data = {error, error_type}
        : keepalive       a comment line during long silences

    WHY POLLING THE EVENT LIST RATHER THAN AN ASYNC QUEUE: the pipeline runs on
    a worker thread and the stream is served on the event loop. Polling a
    lock-guarded list every SSE_POLL_INTERVAL_SECONDS avoids cross-thread
    asyncio primitives entirely, and 200ms is well below the threshold at which
    a progress bar stops feeling live.

    Replays events from seq 0, so a client that connects after work started
    still sees the whole history rather than joining mid-bar.
    """
    job = _require_job(job_id)

    async def event_stream():
        """Yield SSE frames until the job reaches a terminal state."""
        next_seq = 0
        last_send = time.monotonic()

        while True:
            # Stop promptly if the browser navigated away or closed the tab.
            if await request.is_disconnected():
                return

            for event in job.events_since(next_seq):
                next_seq = event.seq + 1
                yield f"event: progress\ndata: {json.dumps(event.to_dict())}\n\n"
                last_send = time.monotonic()

            if job.is_terminal:
                snapshot = job.snapshot()
                if job.status == "error":
                    payload = {
                        "error": snapshot["error"],
                        "error_type": snapshot["error_type"],
                    }
                    yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                else:
                    yield f"event: done\ndata: {json.dumps(snapshot)}\n\n"
                return

            # Keep the connection alive through a long ASR stage, which can run
            # for minutes between progress events on a big video.
            if time.monotonic() - last_send > config.SSE_KEEPALIVE_SECONDS:
                yield ": keepalive\n\n"
                last_send = time.monotonic()

            await asyncio.sleep(config.SSE_POLL_INTERVAL_SECONDS)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx not to buffer, which would defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/frame/{job_id}.png")
async def get_frame(job_id: str) -> FileResponse:
    """Serve the PNG extracted for a finished job.

    SECURITY: the path is taken from the job's own result and then verified to
    sit inside DATA_DIR before being served. The job result is server-generated
    so this should always hold -- the check is there so that a bug upstream
    cannot turn into an arbitrary file read.
    """
    job = _require_job(job_id)
    snapshot = job.snapshot()

    if job.status != "done" or not snapshot.get("result"):
        raise HTTPException(
            status_code=409,
            detail={
                "status": "error",
                "error": f"Job {job_id} is {job.status}, so no frame is available yet.",
                "error_type": "JobNotFinished",
            },
        )

    image_path = (snapshot["result"] or {}).get("image_path")
    if not image_path:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "error": "This job produced no frame (the dialogue was not found).",
                "error_type": "NoFrame",
            },
        )

    path = Path(image_path).resolve()
    try:
        path.relative_to(config.DATA_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "status": "error",
                "error": "Refusing to serve a file outside the data directory.",
                "error_type": "ForbiddenPath",
            },
        ) from exc

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "error": f"Frame image is missing from disk: {path.name}",
                "error_type": "FrameMissing",
            },
        )

    return FileResponse(path, media_type="image/png", filename=path.name)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness plus a check that the external tools are actually present.

    WHY IT PROBES ffmpeg: the server starts perfectly well without ffmpeg on
    PATH and then fails on the first real request. This makes that visible
    immediately instead.
    """
    from app.core import ffmpeg as ffmpeg_mod

    tools: dict[str, Any] = {}
    for name, getter in (("ffmpeg", ffmpeg_mod.ffmpeg_binary),
                         ("ffprobe", ffmpeg_mod.ffprobe_binary)):
        try:
            tools[name] = getter()
        except Quest1Error as exc:
            tools[name] = f"MISSING: {exc}"

    return {
        "status": "ok",
        "version": app.version,
        "asr_model": config.ASR_MODEL,
        "asr_device": f"{config.ASR_DEVICE}/{config.ASR_COMPUTE_TYPE}",
        "index_version": config.INDEX_VERSION,
        "data_dir": str(config.DATA_DIR),
        "tools": tools,
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the frontend if it exists, otherwise a short API pointer.

    Step 9 adds app/web/index.html; until then this returns a plain page naming
    the endpoints, so hitting the server in a browser is never a blank 404.
    """
    page = WEB_DIR / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Quest1 API</h1>"
        "<p>The frontend is not installed yet (Step 9).</p>"
        "<ul>"
        "<li>POST /api/find</li>"
        "<li>GET /api/jobs/{job_id}</li>"
        "<li>GET /api/jobs/{job_id}/events</li>"
        "<li>GET /api/frame/{job_id}.png</li>"
        "<li>GET /api/health</li>"
        "<li>GET /docs</li>"
        "</ul>"
    )
