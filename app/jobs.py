"""Background job tracking for the web layer.

WHY THIS MODULE EXISTS
    Indexing a video can take minutes. An HTTP request that blocked for that
    long would time out in the browser, in any proxy in front of it, and would
    give the user no indication of progress. So POST /api/find starts the work
    on a background thread and returns a job_id immediately; the browser then
    polls or streams progress against that id.

WHY THE STORE IS BEHIND AN INTERFACE
    Right now jobs live in a dict inside this process, which is correct for a
    single local server and requires no external services. It does NOT survive a
    restart and does NOT work across multiple worker processes.

    Swapping in Redis later means writing one more JobStore subclass and
    changing the one line in app/api.py that constructs the store. Nothing else
    in the codebase touches job state directly.

THREADING MODEL
    The pipeline runs on a worker thread and calls Job.record_progress from that
    thread. The API reads job state from the event loop thread. Every mutation
    is therefore guarded by a per-store lock, and progress events are appended to
    a list that the SSE endpoint polls -- no cross-thread asyncio primitives,
    which keeps the failure modes boring.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app import config

# Job lifecycle states.
STATUS_QUEUED = "queued"     # created, worker thread not yet started
STATUS_RUNNING = "running"   # pipeline in progress
STATUS_DONE = "done"         # finished; `result` is populated
STATUS_ERROR = "error"       # failed; `error` and `error_type` are populated

TERMINAL_STATUSES = {STATUS_DONE, STATUS_ERROR}


@dataclass
class ProgressEvent:
    """One progress update, as delivered to a browser over SSE.

    USED BY: Job.record_progress (appends) and the SSE endpoint (replays).
    """

    seq: int              # monotonic per job, so a client can resume after a gap
    stage: str            # resolve | audio | probe | asr | index | match | frame | done
    percent: float        # overall 0-100, never decreases within a job
    message: str
    at: float             # unix timestamp

    def to_dict(self) -> dict[str, Any]:
        """JSON form sent in the SSE data payload."""
        return {
            "seq": self.seq,
            "stage": self.stage,
            "percent": self.percent,
            "message": self.message,
            "at": self.at,
        }


@dataclass
class Job:
    """One dialogue search, its live progress, and its eventual outcome.

    Holds BOTH the latest state (stage/percent/message, for a polling client)
    and the full event history (for an SSE client that connects late and needs
    to catch up). Cheap either way -- a job produces tens of events, not
    thousands.
    """

    job_id: str
    url: str
    text: str
    status: str = STATUS_QUEUED
    stage: str = "queued"
    percent: float = 0.0
    message: str = "queued"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    events: list[ProgressEvent] = field(default_factory=list)

    # Guards this job's own mutable state. Held only for the duration of a field
    # update, never across pipeline work.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_progress(self, stage: str, percent: Optional[float], message: str) -> None:
        """Append a progress event and update the latest-state fields.

        Matches the ProgressCallback signature from app/progress.py exactly, so
        it can be handed straight to find_dialogue as its progress_callback.

        CALLED FROM: the worker thread, many times per job.
        """
        with self._lock:
            self.stage = stage
            if percent is not None:
                self.percent = float(percent)
            self.message = message
            self.updated_at = time.time()
            self.events.append(ProgressEvent(
                seq=len(self.events),
                stage=stage,
                percent=self.percent,
                message=message,
                at=self.updated_at,
            ))

    def mark_running(self) -> None:
        """Transition to running. CALLED FROM: the worker thread on entry."""
        with self._lock:
            self.status = STATUS_RUNNING
            self.updated_at = time.time()

    def mark_done(self, result: dict[str, Any]) -> None:
        """Store a successful result. CALLED FROM: the worker thread."""
        with self._lock:
            self.status = STATUS_DONE
            self.result = result
            self.percent = 100.0
            self.stage = "done"
            self.message = result.get("status", "done")
            self.updated_at = self.finished_at = time.time()

    def mark_error(self, error: str, error_type: str) -> None:
        """Store a failure.

        The message is expected to be a Quest1Error string, which is written to
        be shown to a user verbatim. An unexpected exception is stored with its
        class name so the frontend can still say something useful.

        CALLED FROM: the worker thread.
        """
        with self._lock:
            self.status = STATUS_ERROR
            self.error = error
            self.error_type = error_type
            self.stage = "error"
            self.message = error
            self.updated_at = self.finished_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        """Consistent view of the job for a polling client.

        USED BY: GET /api/jobs/{job_id}.
        """
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "stage": self.stage,
                "percent": round(self.percent, 1),
                "message": self.message,
                "url": self.url,
                "text": self.text,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "finished_at": self.finished_at,
                "result": self.result,
                "error": self.error,
                "error_type": self.error_type,
                "event_count": len(self.events),
            }

    def events_since(self, seq: int) -> list[ProgressEvent]:
        """Every event with seq >= `seq`.

        WHY: an SSE client that connects after work has started must receive the
        events it missed, otherwise its progress bar starts from wherever it
        happened to join.

        USED BY: GET /api/jobs/{job_id}/events.
        """
        with self._lock:
            return [e for e in self.events if e.seq >= seq]

    @property
    def is_terminal(self) -> bool:
        """True once the job can produce no further events.

        USED BY: the SSE endpoint, to close the stream.
        """
        return self.status in TERMINAL_STATUSES


class JobStore:
    """Interface every job backend must implement.

    Deliberately tiny. Everything the API needs is create/get/prune, so a Redis
    or database implementation later has a very small surface to satisfy.
    """

    def create(self, url: str, text: str) -> Job:
        """Register a new job and return it."""
        raise NotImplementedError

    def get(self, job_id: str) -> Optional[Job]:
        """Return a job by id, or None if unknown or already pruned."""
        raise NotImplementedError

    def prune(self) -> int:
        """Drop finished jobs past retention. Returns how many were removed."""
        raise NotImplementedError


class InMemoryJobStore(JobStore):
    """Job state in a process-local dict.

    CORRECT FOR: a single local uvicorn process, which is what Steps 8-10 run.
    NOT CORRECT FOR: multiple workers (each would have its own dict, so a poll
    could land on a process that has never heard of the job) or any deployment
    where jobs must survive a restart. Swap in a shared store before scaling
    past one worker.

    USED BY: app/api.py.
    """

    def __init__(self) -> None:
        """Create an empty store with its own lock."""
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, url: str, text: str) -> Job:
        """Register a new job under a fresh uuid4 and prune old ones.

        Pruning here rather than on a timer keeps the store bounded without a
        background thread, at the cost of doing the work on the request path --
        which is negligible for a dict of a few hundred entries.
        """
        job = Job(job_id=uuid.uuid4().hex, url=url, text=text)
        with self._lock:
            self._jobs[job.job_id] = job
        self.prune()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Look up a job. Returns None for unknown or pruned ids -> HTTP 404."""
        with self._lock:
            return self._jobs.get(job_id)

    def prune(self) -> int:
        """Remove finished jobs past retention, then enforce the count cap.

        Running jobs are NEVER pruned regardless of age -- a slow ASR pass on a
        long video could legitimately outlive the retention window, and dropping
        it would strand the browser watching it.
        """
        now = time.time()
        removed = 0
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.finished_at and (now - job.finished_at) > config.JOB_RETENTION_SECONDS:
                    del self._jobs[job_id]
                    removed += 1

            if len(self._jobs) > config.JOB_MAX_RETAINED:
                finished = sorted(
                    (j for j in self._jobs.values() if j.finished_at),
                    key=lambda j: j.finished_at or 0.0,
                )
                excess = len(self._jobs) - config.JOB_MAX_RETAINED
                for job in finished[:excess]:
                    del self._jobs[job.job_id]
                    removed += 1
        return removed
