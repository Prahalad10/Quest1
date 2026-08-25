"""Job state for the web layer: one dialogue search, its progress, its outcome.

A job holds BOTH the latest state (for a polling client) and the full event
history (for an SSE client that connects late and must catch up). Cheap either
way -- a job produces tens of events, not thousands.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app import config

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
TERMINAL_STATUSES = (STATUS_DONE, STATUS_ERROR)


@dataclass
class ProgressEvent:
    seq: int          # monotonic per job, so a client can resume after a gap
    stage: str
    percent: float    # overall 0-100, never decreases within a job
    message: str
    at: float

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "stage": self.stage, "percent": self.percent,
                "message": self.message, "at": self.at}


@dataclass
class Job:
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
    # Held only for a field update, never across pipeline work.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_progress(self, stage: str, percent: Optional[float], message: str) -> None:
        """Matches the ProgressCallback signature exactly, so it can be handed
        straight to find_dialogue. Called from the worker thread."""
        with self._lock:
            self.stage = stage
            if percent is not None:
                self.percent = float(percent)
            self.message = message
            self.updated_at = time.time()
            self.events.append(ProgressEvent(seq=len(self.events), stage=stage,
                                             percent=self.percent, message=message,
                                             at=self.updated_at))

    def mark_running(self) -> None:
        with self._lock:
            self.status = STATUS_RUNNING
            self.updated_at = time.time()

    def mark_done(self, result: dict[str, Any]) -> None:
        with self._lock:
            self.status = STATUS_DONE
            self.result = result
            self.percent = 100.0
            self.stage = "done"
            self.message = result.get("status", "done")
            self.updated_at = self.finished_at = time.time()

    def mark_error(self, error: str, error_type: str) -> None:
        with self._lock:
            self.status = STATUS_ERROR
            self.error = error
            self.error_type = error_type
            self.stage = "error"
            self.message = error
            self.updated_at = self.finished_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.job_id, "status": self.status, "stage": self.stage,
                "percent": round(self.percent, 1), "message": self.message,
                "url": self.url, "text": self.text,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "finished_at": self.finished_at, "result": self.result,
                "error": self.error, "error_type": self.error_type,
                "event_count": len(self.events),
            }

    def events_since(self, seq: int) -> list[ProgressEvent]:
        """Lets a late or reconnecting client replay what it missed."""
        with self._lock:
            return [e for e in self.events if e.seq >= seq]

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class InMemoryJobStore:
    """Job state in a process-local dict.

    Correct for a single local uvicorn process. NOT correct for multiple
    workers -- each would have its own dict, so a poll could land on a process
    that has never heard of the job -- nor for jobs that must survive a
    restart. Swap in a shared store before scaling past one worker.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, url: str, text: str) -> Job:
        """Pruning here rather than on a timer keeps the store bounded without a
        background thread, at negligible cost for a dict of a few hundred."""
        job = Job(job_id=uuid.uuid4().hex, url=url, text=text)
        with self._lock:
            self._jobs[job.job_id] = job
        self.prune()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """None for unknown or pruned ids -> HTTP 404."""
        with self._lock:
            return self._jobs.get(job_id)

    def prune(self) -> int:
        """Drop finished jobs past retention, then enforce the count cap.

        Running jobs are NEVER pruned regardless of age: a slow ASR pass on a
        long video can outlive the retention window, and dropping it would
        strand the browser watching it.
        """
        now = time.time()
        removed = 0
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.finished_at and (now - job.finished_at) > config.JOB_RETENTION_SECONDS:
                    del self._jobs[job_id]
                    removed += 1
            if len(self._jobs) > config.JOB_MAX_RETAINED:
                finished = sorted((j for j in self._jobs.values() if j.finished_at),
                                  key=lambda j: j.finished_at or 0.0)
                for job in finished[:len(self._jobs) - config.JOB_MAX_RETAINED]:
                    del self._jobs[job.job_id]
                    removed += 1
        return removed
