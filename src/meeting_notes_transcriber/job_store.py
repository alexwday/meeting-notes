from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


INTERRUPTED_JOB_MESSAGE = "A previous app session ended before this job finished. Submit the file again to retry."


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class JobRecord:
    job_id: str
    original_filename: str
    stored_path: str
    status: str
    status_message: str
    settings: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None = None
    progress_percent: float | None = None
    error: str | None = None
    transcript_preview: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self, jobs_root: Path):
        self._jobs_root = jobs_root
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._load_existing_jobs()

    def create(
        self,
        *,
        job_id: str,
        original_filename: str,
        stored_path: str,
        settings: dict[str, Any],
    ) -> JobRecord:
        job = JobRecord(
            job_id=job_id,
            original_filename=original_filename,
            stored_path=stored_path,
            status="queued",
            status_message="Queued for transcription",
            settings=settings,
            created_at=utc_now(),
            updated_at=utc_now(),
            progress_percent=0.0,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._write_snapshot(job)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str, message: str, *, progress_percent: float | None = None) -> JobRecord:
        return self._update(
            job_id,
            status="running",
            status_message=message,
            error=None,
            progress_percent=_normalize_progress(progress_percent),
        )

    def update_progress(self, job_id: str, *, message: str, progress_percent: float) -> JobRecord:
        return self._update(
            job_id,
            status="running",
            status_message=message,
            error=None,
            progress_percent=_normalize_progress(progress_percent),
        )

    def mark_completed(
        self,
        job_id: str,
        *,
        message: str,
        transcript_preview: str,
        metadata: dict[str, Any],
        outputs: dict[str, str],
    ) -> JobRecord:
        return self._update(
            job_id,
            status="completed",
            status_message=message,
            progress_percent=100.0,
            transcript_preview=transcript_preview,
            metadata=metadata,
            outputs=outputs,
        )

    def mark_failed(self, job_id: str, *, message: str, error: str) -> JobRecord:
        return self._update(job_id, status="failed", status_message=message, error=error)

    def _update(self, job_id: str, **changes: Any) -> JobRecord:
        with self._lock:
            job = self._jobs[job_id]
            if changes.get("status") == "running" and job.started_at is None:
                job.started_at = utc_now()
            for key, value in changes.items():
                if value is not None or key in {"error", "transcript_preview", "metadata", "outputs"}:
                    setattr(job, key, value)
            job.updated_at = utc_now()
            self._write_snapshot(job)
            return job

    def _load_existing_jobs(self) -> None:
        if not self._jobs_root.exists():
            return

        for snapshot in self._jobs_root.glob("*/job.json"):
            payload = json.loads(snapshot.read_text())
            job = JobRecord(**payload)
            changed = False
            if job.status in {"queued", "running"}:
                job.status = "failed"
                job.status_message = INTERRUPTED_JOB_MESSAGE
                job.error = INTERRUPTED_JOB_MESSAGE
                job.updated_at = utc_now()
                if job.progress_percent is None:
                    job.progress_percent = 0.0
                changed = True
            elif job.progress_percent is None:
                if job.status == "completed":
                    job.progress_percent = 100.0
                    changed = True
                elif job.status == "failed":
                    job.progress_percent = 0.0
                    changed = True
            self._jobs[job.job_id] = job
            if changed:
                self._write_snapshot(job)

    def _write_snapshot(self, job: JobRecord) -> None:
        job_dir = self._jobs_root / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2))


def _normalize_progress(progress_percent: float | None) -> float | None:
    if progress_percent is None:
        return None
    return max(0.0, min(100.0, round(float(progress_percent), 1)))
