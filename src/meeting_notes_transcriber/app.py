from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any
import os

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig, load_config
from .diarization import LocalSpeakerDiarizer, resolve_hf_token
from .job_store import JobStore
from .ssl import SSLState, ensure_enterprise_ssl
from .system import RuntimeProfile, detect_runtime_profile
from .transcriber import (
    TranscriptionService,
    TranscriptionSettings,
    UnsupportedConfigurationError,
    list_model_presets,
    resolve_settings,
    summarize_assigned_speakers,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobTask:
    job_id: str
    source_path: Path
    output_root: Path
    settings: TranscriptionSettings
    hf_token: str | None


class TranscriptionWorker:
    def __init__(self, job_store: JobStore, service: TranscriptionService):
        self._job_store = job_store
        self._service = service
        self._queue: Queue[JobTask] = Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="transcription-worker")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def submit(self, task: JobTask) -> None:
        self._queue.put(task)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.25)
            except Empty:
                continue

            try:
                last_progress = -1.0
                last_message = ""

                def report_progress(progress_percent: float, message: str) -> None:
                    nonlocal last_progress, last_message
                    if progress_percent < 99.0 and message == last_message and progress_percent - last_progress < 1.5:
                        return
                    last_progress = progress_percent
                    last_message = message
                    self._job_store.update_progress(
                        task.job_id,
                        message=message,
                        progress_percent=progress_percent,
                    )

                self._job_store.mark_running(
                    task.job_id,
                    "Preparing transcription",
                    progress_percent=1.0,
                )
                transcript, outputs = self._service.transcribe(
                    source_path=task.source_path,
                    output_root=task.output_root,
                    settings=task.settings,
                    hf_token=task.hf_token,
                    progress_callback=report_progress,
                )
                preview = transcript["text"][:3000]
                self._job_store.mark_completed(
                    task.job_id,
                    message="Transcription finished",
                    transcript_preview=preview,
                    metadata={
                        "language": transcript["language"],
                        "language_probability": transcript["language_probability"],
                        "duration": transcript["duration"],
                        "duration_after_vad": transcript["duration_after_vad"],
                        "device": transcript["device"],
                        "compute_type": transcript["compute_type"],
                        "model": transcript["model"],
                        "model_label": transcript["model_label"],
                        "diarization_enabled": transcript["diarization_enabled"],
                        "num_speakers_detected": transcript["num_speakers_detected"],
                        "speaker_labels": transcript["speaker_labels"],
                        "diarization_speaker_count": transcript["diarization_speaker_count"],
                        "diarization_speaker_labels": transcript["diarization_speaker_labels"],
                    },
                    outputs=outputs,
                )
                self._cleanup_source_file(task.source_path)
            except Exception as exc:
                logger.exception("Job %s failed", task.job_id)
                self._job_store.mark_failed(
                    task.job_id,
                    message="Transcription failed",
                    error=str(exc),
                )
            finally:
                self._queue.task_done()

    @staticmethod
    def _cleanup_source_file(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove uploaded source file '%s' after job completion.", source_path)


@dataclass(frozen=True)
class AppServices:
    config: AppConfig
    runtime_profile: RuntimeProfile
    ssl_state: SSLState
    job_store: JobStore
    worker: TranscriptionWorker
    diarizer: LocalSpeakerDiarizer


def build_services(config: AppConfig) -> AppServices:
    config.ensure_directories()
    os.environ.setdefault("MPLCONFIGDIR", str(config.matplotlib_root))
    ssl_state = ensure_enterprise_ssl()
    runtime_profile = detect_runtime_profile()
    job_store = JobStore(config.jobs_root)
    diarizer = LocalSpeakerDiarizer(
        config.pyannote_cache_root,
        bundled_model_root=config.bundled_pyannote_model_root,
        device=runtime_profile.device,
    )
    transcription_service = TranscriptionService(
        config.model_cache_root,
        bundled_models_root=config.bundled_models_root,
        device=runtime_profile.device,
        compute_type=runtime_profile.compute_type,
        diarizer=diarizer,
    )
    worker = TranscriptionWorker(job_store, transcription_service)
    worker.start()
    return AppServices(
        config=config,
        runtime_profile=runtime_profile,
        ssl_state=ssl_state,
        job_store=job_store,
        worker=worker,
        diarizer=diarizer,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    services = build_services(config)
    app.state.services = services
    yield
    services.worker.stop()


app = FastAPI(title="Meeting Notes Transcriber", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(load_config().static_root)), name="static")


def services_for(request: Request) -> AppServices:
    return request.app.state.services


def derive_speaker_metadata_from_transcript(transcript: dict[str, Any]) -> dict[str, Any]:
    assigned_labels = summarize_assigned_speakers(transcript.get("segments", []))

    raw_labels = transcript.get("diarization_speaker_labels")
    if raw_labels is None and transcript.get("diarization_enabled"):
        raw_labels = transcript.get("speaker_labels", [])

    raw_count = transcript.get("diarization_speaker_count")
    if raw_count is None and transcript.get("diarization_enabled"):
        if raw_labels:
            raw_count = len(raw_labels)
        else:
            raw_count = int(transcript.get("num_speakers_detected") or 0)

    return {
        "speaker_labels": assigned_labels,
        "num_speakers_detected": len(assigned_labels),
        "diarization_speaker_labels": list(raw_labels or []),
        "diarization_speaker_count": int(raw_count or 0),
    }


def load_job_transcript_metadata(job: Any) -> dict[str, Any]:
    json_path = job.outputs.get("json")
    if not json_path:
        return {}

    path = Path(json_path)
    if not path.exists():
        return {}

    try:
        transcript = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    return derive_speaker_metadata_from_transcript(transcript)


def job_to_response(job: Any) -> dict[str, Any]:
    payload = job.to_dict()
    transcript_metadata = load_job_transcript_metadata(job)
    if transcript_metadata:
        payload["metadata"] = {**payload.get("metadata", {}), **transcript_metadata}
    payload["downloads"] = {
        key: f"/api/jobs/{job.job_id}/download/{key}"
        for key in job.outputs
    }
    return payload


@app.get("/")
async def index(request: Request) -> FileResponse:
    return FileResponse(services_for(request).config.static_root / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/system")
async def system_info(request: Request) -> dict[str, Any]:
    services = services_for(request)
    return {
        "runtime": services.runtime_profile.to_dict(),
        "ssl": services.ssl_state.to_dict(),
        "models": list_model_presets(),
        "diarization": services.diarizer.status(),
    }


@app.post("/api/jobs")
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    model_key: str = Form("distil-large-v3"),
    custom_model_id: str | None = Form(default=None),
    language: str | None = Form(default="en"),
    task: str = Form("transcribe"),
    beam_size: int = Form(5),
    temperature: float = Form(0.0),
    vad_filter: bool = Form(False),
    word_timestamps: bool = Form(False),
    initial_prompt: str | None = Form(default=None),
    enable_diarization: bool = Form(False),
    diarization_exclusive: bool = Form(False),
    num_speakers: int | None = Form(default=None),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    hf_token: str | None = Form(default=None),
) -> JSONResponse:
    services = services_for(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file is required.")

    normalized_hf_token = (hf_token or "").strip() or None
    settings = TranscriptionSettings(
        model_key=model_key,
        custom_model_id=custom_model_id,
        language=language,
        task=task,
        beam_size=beam_size,
        temperature=temperature,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        initial_prompt=initial_prompt,
        enable_diarization=enable_diarization,
        diarization_exclusive=diarization_exclusive,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    job_id = uuid.uuid4().hex
    safe_name = Path(file.filename).name
    stored_path = services.config.upload_root / f"{job_id}-{safe_name}"
    job_root = services.config.jobs_root / job_id
    job_root.mkdir(parents=True, exist_ok=True)

    with stored_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    await file.close()

    try:
        resolve_settings(settings)
        if settings.enable_diarization and not (
            services.diarizer.is_loaded()
            or services.diarizer.has_local_model()
            or resolve_hf_token(normalized_hf_token)
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Diarization requires a Hugging Face token the first time unless the model already exists locally."
                ),
            )
    except UnsupportedConfigurationError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise

    job = services.job_store.create(
        job_id=job_id,
        original_filename=safe_name,
        stored_path=str(stored_path),
        settings=settings.to_dict(),
    )

    services.worker.submit(
        JobTask(
            job_id=job_id,
            source_path=stored_path,
            output_root=job_root,
            settings=settings,
            hf_token=normalized_hf_token,
        )
    )
    return JSONResponse(job_to_response(job), status_code=202)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, Any]:
    job = services_for(request).job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_to_response(job)


@app.get("/api/jobs/{job_id}/download/{fmt}")
async def download_output(job_id: str, fmt: str, request: Request) -> FileResponse:
    job = services_for(request).job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    output_path = job.outputs.get(fmt)
    if output_path is None:
        raise HTTPException(status_code=404, detail="Output not available.")
    path = Path(output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file is missing.")
    media_types = {
        "txt": "text/plain",
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
        "json": "application/json",
    }
    return FileResponse(path, filename=path.name, media_type=media_types.get(fmt))
