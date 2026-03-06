from __future__ import annotations

import json
import logging
import shutil
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .diarization import (
    DiarizationSettings,
    LocalSpeakerDiarizer,
    assign_speakers_to_segments,
    validate_diarization_settings,
)
from .ssl import ensure_enterprise_ssl


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPreset:
    key: str
    label: str
    model_id: str
    bundled_dir_name: str | None
    description: str
    english_only: bool
    supports_translation: bool
    recommended_for: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_PRESETS = {
    "turbo": ModelPreset(
        key="turbo",
        label="Whisper Turbo",
        model_id="turbo",
        bundled_dir_name="turbo",
        description="Best default for multilingual transcription on laptop-class hardware.",
        english_only=False,
        supports_translation=False,
        recommended_for="Fast multilingual transcription",
    ),
    "large-v3": ModelPreset(
        key="large-v3",
        label="Whisper Large v3",
        model_id="large-v3",
        bundled_dir_name="large-v3",
        description="Highest-quality multilingual option in this build. Use when translation quality matters.",
        english_only=False,
        supports_translation=True,
        recommended_for="Best multilingual quality and translation",
    ),
    "distil-large-v3": ModelPreset(
        key="distil-large-v3",
        label="Distil-Whisper Large v3",
        model_id="distil-large-v3",
        bundled_dir_name="distil-large-v3",
        description="English-only distilled checkpoint with the best speed profile in this app.",
        english_only=True,
        supports_translation=False,
        recommended_for="Fast English-only transcription",
    ),
    "custom": ModelPreset(
        key="custom",
        label="Custom model",
        model_id="",
        bundled_dir_name=None,
        description="Load a custom CTranslate2-compatible Hugging Face model id or local path.",
        english_only=False,
        supports_translation=True,
        recommended_for="Advanced experimentation",
    ),
}


@dataclass(frozen=True)
class TranscriptionSettings:
    model_key: str
    custom_model_id: str | None
    language: str | None
    task: str
    beam_size: int
    temperature: float
    vad_filter: bool
    word_timestamps: bool
    initial_prompt: str | None
    enable_diarization: bool
    diarization_exclusive: bool
    num_speakers: int | None
    min_speakers: int | None
    max_speakers: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedSettings:
    model_id: str
    display_label: str
    bundled_model_dir_name: str | None
    language: str | None
    task: str
    beam_size: int
    temperature: float
    vad_filter: bool
    word_timestamps: bool
    initial_prompt: str | None
    english_only: bool
    condition_on_previous_text: bool
    diarization: DiarizationSettings


class UnsupportedConfigurationError(ValueError):
    pass


def list_model_presets() -> list[dict[str, Any]]:
    return [preset.to_dict() for preset in MODEL_PRESETS.values()]


def resolve_settings(settings: TranscriptionSettings) -> ResolvedSettings:
    if settings.model_key not in MODEL_PRESETS:
        raise UnsupportedConfigurationError(f"Unknown model preset: {settings.model_key}")

    preset = MODEL_PRESETS[settings.model_key]
    model_id = preset.model_id
    if settings.model_key == "custom":
        if not settings.custom_model_id:
            raise UnsupportedConfigurationError("A custom model id or local model path is required.")
        model_id = settings.custom_model_id.strip()

    if settings.task not in {"transcribe", "translate"}:
        raise UnsupportedConfigurationError("Task must be either 'transcribe' or 'translate'.")

    if settings.task == "translate" and not preset.supports_translation:
        raise UnsupportedConfigurationError(f"{preset.label} is not recommended for translation. Use large-v3 instead.")

    language = (settings.language or "").strip().lower() or None
    if preset.english_only and language not in {None, "en", "english"}:
        raise UnsupportedConfigurationError(f"{preset.label} is English-only. Use English or switch models.")
    if preset.english_only and language is None:
        language = "en"

    if settings.beam_size < 1 or settings.beam_size > 10:
        raise UnsupportedConfigurationError("Beam size must be between 1 and 10.")

    if settings.temperature < 0.0 or settings.temperature > 1.0:
        raise UnsupportedConfigurationError("Temperature must be between 0.0 and 1.0.")

    diarization = DiarizationSettings(
        enabled=settings.enable_diarization,
        exclusive=settings.diarization_exclusive,
        num_speakers=settings.num_speakers,
        min_speakers=settings.min_speakers,
        max_speakers=settings.max_speakers,
    )
    try:
        validate_diarization_settings(diarization)
    except ValueError as exc:
        raise UnsupportedConfigurationError(str(exc)) from exc

    return ResolvedSettings(
        model_id=model_id,
        display_label=preset.label,
        bundled_model_dir_name=preset.bundled_dir_name,
        language=language,
        task=settings.task,
        beam_size=settings.beam_size,
        temperature=settings.temperature,
        vad_filter=settings.vad_filter,
        word_timestamps=settings.word_timestamps,
        initial_prompt=(settings.initial_prompt or "").strip() or None,
        english_only=preset.english_only,
        condition_on_previous_text=not preset.english_only,
        diarization=diarization,
    )


class TranscriptionService:
    def __init__(
        self,
        model_cache_root: Path,
        *,
        bundled_models_root: Path | None = None,
        device: str,
        compute_type: str,
        diarizer: LocalSpeakerDiarizer | None = None,
    ):
        self._model_cache_root = model_cache_root
        self._bundled_models_root = bundled_models_root
        self._device = device
        self._compute_type = compute_type
        self._diarizer = diarizer
        self._lock = threading.Lock()
        self._models: dict[str, Any] = {}

    def transcribe(
        self,
        *,
        source_path: Path,
        output_root: Path,
        settings: TranscriptionSettings,
        hf_token: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        resolved = resolve_settings(settings)
        _emit_progress(progress_callback, 4.0, "Loading transcription model")
        model = self._load_model(resolved.model_id, resolved.bundled_model_dir_name)
        _emit_progress(progress_callback, 10.0, "Transcribing audio")
        segments_iter, info = model.transcribe(
            str(source_path),
            task=resolved.task,
            language=resolved.language,
            beam_size=resolved.beam_size,
            temperature=resolved.temperature,
            vad_filter=resolved.vad_filter,
            word_timestamps=resolved.word_timestamps,
            initial_prompt=resolved.initial_prompt,
            condition_on_previous_text=resolved.condition_on_previous_text,
        )
        duration_hint = float(getattr(info, "duration", 0.0) or 0.0)
        last_transcription_progress = 10.0

        segments: list[dict[str, Any]] = []
        for index, segment in enumerate(segments_iter, start=1):
            words: list[dict[str, Any]] = []
            for word in getattr(segment, "words", []) or []:
                words.append(
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": getattr(word, "probability", None),
                    }
                )
            segments.append(
                {
                    "id": getattr(segment, "id", index),
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "compression_ratio": getattr(segment, "compression_ratio", None),
                    "no_speech_prob": getattr(segment, "no_speech_prob", None),
                    "words": words,
                    }
                )
            if duration_hint > 0:
                segment_end = float(getattr(segment, "end", 0.0) or 0.0)
                transcription_progress = 10.0 + min(max(segment_end / duration_hint, 0.0), 1.0) * 72.0
                if transcription_progress >= last_transcription_progress + 2.0 or segment_end >= duration_hint:
                    last_transcription_progress = transcription_progress
                    _emit_progress(progress_callback, transcription_progress, "Transcribing audio")

        _emit_progress(progress_callback, max(last_transcription_progress, 84.0), "Formatting transcript")

        diarization_payload = {
            "diarization": [],
            "exclusive_diarization": [],
            "speaker_labels": [],
            "speaker_count": 0,
        }
        if resolved.diarization.enabled:
            if self._diarizer is None:
                raise UnsupportedConfigurationError("Diarization is enabled but the diarization service is unavailable.")
            _emit_progress(progress_callback, 86.0, "Running speaker diarization")
            diarization_payload = self._diarizer.diarize(
                source_path,
                settings=resolved.diarization,
                hf_token=hf_token,
            )
            diarization_turns = (
                diarization_payload["exclusive_diarization"]
                if resolved.diarization.exclusive
                else diarization_payload["diarization"]
            )
            segments = assign_speakers_to_segments(segments, diarization_turns)
            _emit_progress(progress_callback, 94.0, "Speaker diarization finished")

        transcript_speaker_labels = summarize_assigned_speakers(segments)
        raw_diarization_labels = diarization_payload["speaker_labels"]
        unassigned_diarization_labels = [
            label for label in raw_diarization_labels if label not in transcript_speaker_labels
        ]
        plain_text = build_plain_text(segments)
        transcript_text = render_transcript_text(segments)
        _emit_progress(progress_callback, 97.0, "Writing output files")
        payload = {
            "model": resolved.model_id,
            "model_label": resolved.display_label,
            "task": resolved.task,
            "language": resolved.language or getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "duration_after_vad": getattr(info, "duration_after_vad", None),
            "device": self._device,
            "compute_type": self._compute_type,
            "english_only": resolved.english_only,
            "plain_text": plain_text,
            "text": transcript_text,
            "segments": segments,
            "diarization_enabled": resolved.diarization.enabled,
            "diarization": diarization_payload["diarization"],
            "exclusive_diarization": diarization_payload["exclusive_diarization"],
            "speaker_labels": transcript_speaker_labels,
            "num_speakers_detected": len(transcript_speaker_labels),
            "diarization_speaker_labels": raw_diarization_labels,
            "diarization_speaker_count": diarization_payload["speaker_count"],
            "unassigned_diarization_speaker_labels": unassigned_diarization_labels,
        }

        outputs = write_output_files(output_root=output_root, transcript=payload)
        return payload, outputs

    def _load_model(self, model_id: str, bundled_model_dir_name: str | None) -> Any:
        with self._lock:
            model_reference = self._resolve_model_reference(model_id, bundled_model_dir_name)
            cache_key = str(model_reference)
            cached = self._models.get(cache_key)
            if cached is not None:
                return cached

            ensure_enterprise_ssl()
            from faster_whisper import WhisperModel

            logger.info("Loading model '%s' on %s (%s).", model_reference, self._device, self._compute_type)
            try:
                model = self._instantiate_model(WhisperModel, model_reference)
            except Exception as exc:
                if self._repair_model_cache_and_retry(exc, model_reference, model_id):
                    logger.warning("Retrying model load for '%s' after clearing its local download cache.", model_reference)
                    try:
                        model = self._instantiate_model(WhisperModel, model_reference)
                    except Exception as retry_exc:
                        raise RuntimeError(self._format_model_load_error(model_reference, retry_exc)) from retry_exc
                else:
                    raise RuntimeError(self._format_model_load_error(model_reference, exc)) from exc
            self._models[cache_key] = model
            return model

    def _instantiate_model(self, whisper_model_cls: Any, model_reference: str) -> Any:
        return whisper_model_cls(
            model_reference,
            device=self._device,
            compute_type=self._compute_type,
            download_root=str(self._model_cache_root),
        )

    def _repair_model_cache_and_retry(self, error: Exception, model_reference: str, model_id: str) -> bool:
        if not self._is_retryable_model_cache_error(error):
            return False

        repo_id = self._resolve_remote_repo_id(model_reference, model_id)
        if repo_id is None:
            return False

        self._clear_model_download_cache(repo_id)
        return True

    def _resolve_remote_repo_id(self, model_reference: str, model_id: str) -> str | None:
        reference_path = Path(model_reference)
        if reference_path.exists():
            return None

        if "/" in model_reference:
            return model_reference

        try:
            from faster_whisper.utils import _MODELS  # type: ignore[attr-defined]
        except Exception:
            _MODELS = {}

        return _MODELS.get(model_reference) or _MODELS.get(model_id)

    def _clear_model_download_cache(self, repo_id: str) -> None:
        cache_dir_name = f"models--{repo_id.replace('/', '--')}"
        repo_cache_dir = self._model_cache_root / cache_dir_name
        lock_dir = self._model_cache_root / ".locks" / cache_dir_name

        if repo_cache_dir.exists():
            shutil.rmtree(repo_cache_dir, ignore_errors=True)
        if lock_dir.exists():
            shutil.rmtree(lock_dir, ignore_errors=True)

    @staticmethod
    def _is_retryable_model_cache_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "snapshot folder" in message
            or "specified revision" in message
            or "consistency check failed" in message
            or "file should be of size" in message
        )

    @staticmethod
    def _format_model_load_error(model_reference: str, error: Exception) -> str:
        message = str(error)
        lowered = message.lower()
        if "ssl" in lowered or "certificate" in lowered:
            return (
                f"Failed to download Whisper model '{model_reference}'. SSL validation failed while contacting "
                "Hugging Face. Confirm the enterprise SSL environment is active and retry."
            )
        if (
            "snapshot folder" in lowered
            or "specified revision" in lowered
            or "consistency check failed" in lowered
            or "file should be of size" in lowered
        ):
            return (
                f"Failed to load Whisper model '{model_reference}' because the local Hugging Face download is incomplete "
                "or corrupted. The app attempted a cache repair. If the problem continues, your network path may be "
                "returning a small proxy/error response instead of the model file. Delete .data/models and retry after "
                "confirming enterprise access to Hugging Face large file downloads."
            )
        return f"Failed to load Whisper model '{model_reference}': {message}"

    def _resolve_model_reference(self, model_id: str, bundled_model_dir_name: str | None) -> str:
        bundled_model_path = self._resolve_bundled_model_path(bundled_model_dir_name)
        if bundled_model_path is not None:
            return str(bundled_model_path)
        return model_id

    def _resolve_bundled_model_path(self, bundled_model_dir_name: str | None) -> Path | None:
        if not bundled_model_dir_name or self._bundled_models_root is None:
            return None

        candidate = self._bundled_models_root / "whisper" / bundled_model_dir_name
        if candidate.is_dir() and (candidate / "model.bin").exists():
            return candidate
        return None


def write_output_files(*, output_root: Path, transcript: dict[str, Any]) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)

    text_path = output_root / "transcript.txt"
    srt_path = output_root / "transcript.srt"
    vtt_path = output_root / "transcript.vtt"
    json_path = output_root / "transcript.json"

    text_path.write_text(transcript["text"])
    srt_path.write_text(render_srt(transcript["segments"]))
    vtt_path.write_text(render_vtt(transcript["segments"]))
    json_path.write_text(json.dumps(transcript, indent=2))

    return {
        "txt": str(text_path),
        "srt": str(srt_path),
        "vtt": str(vtt_path),
        "json": str(json_path),
    }


def summarize_assigned_speakers(segments: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker or speaker in seen:
            continue
        seen.add(speaker)
        labels.append(speaker)
    return labels


def _emit_progress(
    progress_callback: Callable[[float, str], None] | None,
    progress_percent: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(progress_percent, message)


def render_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_timestamp(segment['start'], srt=True)} --> {format_timestamp(segment['end'], srt=True)}",
                    display_segment_text(segment),
                ]
            )
        )
    return "\n\n".join(blocks).strip() + "\n"


def render_vtt(segments: list[dict[str, Any]]) -> str:
    blocks = ["WEBVTT\n"]
    for segment in segments:
        blocks.append(
            "\n".join(
                [
                    f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}",
                    display_segment_text(segment),
                    "",
                ]
            )
        )
    return "\n".join(blocks).strip() + "\n"


def build_plain_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(segment["text"] for segment in segments if segment["text"]).strip()


def render_transcript_text(segments: list[dict[str, Any]]) -> str:
    if not segments:
        return "No speech detected."

    if not any(segment.get("speaker") for segment in segments):
        plain_text = build_plain_text(segments)
        return plain_text or "No speech detected."

    lines: list[str] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue

        speaker = segment.get("speaker")
        if not speaker:
            if current_parts:
                lines.append(f"[{current_speaker}] {' '.join(current_parts)}")
                current_parts = []
                current_speaker = None
            lines.append(text)
            continue

        if current_speaker != speaker and current_parts:
            lines.append(f"[{current_speaker}] {' '.join(current_parts)}")
            current_parts = []

        current_speaker = speaker
        current_parts.append(text)

    if current_parts:
        lines.append(f"[{current_speaker}] {' '.join(current_parts)}")

    rendered = "\n".join(lines).strip()
    return rendered or "No speech detected."


def display_segment_text(segment: dict[str, Any]) -> str:
    text = segment["text"].strip()
    speaker = segment.get("speaker")
    if speaker and text:
        return f"[{speaker}] {text}"
    return text


def format_timestamp(value: float, *, srt: bool = False) -> str:
    milliseconds = round(value * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"
