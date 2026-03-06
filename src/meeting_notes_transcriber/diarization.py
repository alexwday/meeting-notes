from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ssl import ensure_enterprise_ssl


logger = logging.getLogger(__name__)

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"
PYANNOTE_CACHE_REPO_DIR = f"models--{PYANNOTE_MODEL_ID.replace('/', '--')}"
_FFMPEG_LOCK = threading.Lock()
_FFMPEG_PATH: Path | None = None
_FFMPEG_SOURCE: str | None = None


class DiarizationSetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiarizationSettings:
    enabled: bool
    exclusive: bool
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pyannote_available() -> bool:
    return importlib.util.find_spec("pyannote.audio") is not None


def ffmpeg_available() -> bool:
    return resolve_ffmpeg_path()[0] is not None


def resolve_ffmpeg_path() -> tuple[Path | None, str | None]:
    global _FFMPEG_PATH, _FFMPEG_SOURCE

    with _FFMPEG_LOCK:
        if _FFMPEG_PATH is not None and _FFMPEG_PATH.exists():
            _apply_ffmpeg_environment(_FFMPEG_PATH)
            return _FFMPEG_PATH, _FFMPEG_SOURCE

        system_path = shutil.which("ffmpeg")
        if system_path:
            _FFMPEG_PATH = Path(system_path)
            _FFMPEG_SOURCE = "system"
            _apply_ffmpeg_environment(_FFMPEG_PATH)
            return _FFMPEG_PATH, _FFMPEG_SOURCE

        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError:
            return None, None

        try:
            bundled_path = Path(get_ffmpeg_exe())
        except Exception as exc:
            logger.warning("Unable to resolve bundled ffmpeg runtime: %s", exc)
            return None, None

        if not bundled_path.exists():
            return None, None

        _FFMPEG_PATH = bundled_path
        _FFMPEG_SOURCE = "imageio-ffmpeg"
        _apply_ffmpeg_environment(_FFMPEG_PATH)
        return _FFMPEG_PATH, _FFMPEG_SOURCE


def _apply_ffmpeg_environment(ffmpeg_path: Path) -> None:
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", str(ffmpeg_path))
    path_prefix = str(ffmpeg_path.parent)
    current_path = os.getenv("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if path_prefix not in path_entries:
        os.environ["PATH"] = f"{path_prefix}{os.pathsep}{current_path}" if current_path else path_prefix


def _reset_ffmpeg_resolution_cache() -> None:
    global _FFMPEG_PATH, _FFMPEG_SOURCE
    with _FFMPEG_LOCK:
        _FFMPEG_PATH = None
        _FFMPEG_SOURCE = None


def resolve_hf_token(request_token: str | None) -> str | None:
    token = (request_token or "").strip()
    if token:
        return token

    for env_name in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
        env_token = (os.getenv(env_name) or "").strip()
        if env_token:
            return env_token
    return None


def validate_diarization_settings(settings: DiarizationSettings) -> None:
    if not settings.enabled:
        return

    if settings.num_speakers is not None and (
        settings.min_speakers is not None or settings.max_speakers is not None
    ):
        raise ValueError("Use either an exact speaker count or a min/max range, not both.")

    for value, label in (
        (settings.num_speakers, "Exact speaker count"),
        (settings.min_speakers, "Minimum speakers"),
        (settings.max_speakers, "Maximum speakers"),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{label} must be at least 1.")

    if (
        settings.min_speakers is not None
        and settings.max_speakers is not None
        and settings.min_speakers > settings.max_speakers
    ):
        raise ValueError("Minimum speakers cannot be greater than maximum speakers.")


def diarization_environment(
    *,
    pipeline_loaded: bool,
    local_model_path: Path | None,
    local_model_source: str | None,
) -> dict[str, Any]:
    ffmpeg_path, ffmpeg_source = resolve_ffmpeg_path()
    return {
        "available": pyannote_available(),
        "ffmpeg_available": ffmpeg_path is not None,
        "ffmpeg_path": str(ffmpeg_path) if ffmpeg_path else None,
        "ffmpeg_source": ffmpeg_source,
        "hf_token_configured": resolve_hf_token(None) is not None,
        "pipeline_loaded": pipeline_loaded,
        "model_id": PYANNOTE_MODEL_ID,
        "local_model_ready": local_model_path is not None,
        "local_model_path": str(local_model_path) if local_model_path else None,
        "local_model_source": local_model_source,
    }


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]],
    *,
    fallback_gap: float = 0.6,
) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for segment in segments:
        speaker = _best_matching_speaker(segment, diarization_turns, fallback_gap=fallback_gap)
        enriched = dict(segment)
        enriched["speaker"] = speaker
        assigned.append(enriched)
    return assigned


def _best_matching_speaker(
    segment: dict[str, Any],
    diarization_turns: list[dict[str, Any]],
    *,
    fallback_gap: float,
) -> str | None:
    start = float(segment["start"])
    end = float(segment["end"])

    overlaps: dict[str, float] = {}
    for turn in diarization_turns:
        overlap = min(end, float(turn["end"])) - max(start, float(turn["start"]))
        if overlap > 0:
            speaker = str(turn["speaker"])
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap

    if overlaps:
        return max(overlaps.items(), key=lambda item: (item[1], item[0]))[0]

    closest_speaker: str | None = None
    closest_gap: float | None = None
    for turn in diarization_turns:
        turn_start = float(turn["start"])
        turn_end = float(turn["end"])
        if end <= turn_start:
            gap = turn_start - end
        elif start >= turn_end:
            gap = start - turn_end
        else:
            gap = 0.0

        if gap <= fallback_gap and (closest_gap is None or gap < closest_gap):
            closest_gap = gap
            closest_speaker = str(turn["speaker"])

    return closest_speaker


class LocalSpeakerDiarizer:
    def __init__(self, cache_root: Path, *, bundled_model_root: Path, device: str):
        self._cache_root = cache_root
        self._bundled_model_root = bundled_model_root
        self._device = device
        self._lock = threading.Lock()
        self._pipeline: Any | None = None

    def status(self) -> dict[str, Any]:
        local_model_path, local_model_source = self._resolve_local_model_path()
        return diarization_environment(
            pipeline_loaded=self._pipeline is not None,
            local_model_path=local_model_path,
            local_model_source=local_model_source,
        )

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def has_local_model(self) -> bool:
        local_model_path, _ = self._resolve_local_model_path()
        return local_model_path is not None

    def diarize(
        self,
        source_path: Path,
        *,
        settings: DiarizationSettings,
        hf_token: str | None,
    ) -> dict[str, Any]:
        validate_diarization_settings(settings)
        if not settings.enabled:
            return {
                "diarization": [],
                "exclusive_diarization": [],
                "speaker_labels": [],
                "speaker_count": 0,
            }

        pipeline = self._load_pipeline(hf_token)
        kwargs: dict[str, Any] = {}
        if settings.num_speakers is not None:
            kwargs["num_speakers"] = settings.num_speakers
        else:
            if settings.min_speakers is not None:
                kwargs["min_speakers"] = settings.min_speakers
            if settings.max_speakers is not None:
                kwargs["max_speakers"] = settings.max_speakers

        logger.info("Running speaker diarization for %s", source_path.name)
        output = pipeline(str(source_path), **kwargs)
        serialized = output.serialize() if hasattr(output, "serialize") else {
            "diarization": [],
            "exclusive_diarization": [],
        }

        speaker_labels = sorted(
            {
                turn["speaker"]
                for key in ("exclusive_diarization", "diarization")
                for turn in serialized.get(key, [])
            }
        )
        serialized["speaker_labels"] = speaker_labels
        serialized["speaker_count"] = len(speaker_labels)
        return serialized

    def _load_pipeline(self, hf_token: str | None) -> Any:
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            if not pyannote_available():
                raise DiarizationSetupError(
                    "pyannote.audio is not installed. Re-run the project setup to install diarization support."
                )
            ffmpeg_path, ffmpeg_source = resolve_ffmpeg_path()
            if ffmpeg_path is None:
                raise DiarizationSetupError(
                    "ffmpeg is required for diarization, but no system install or bundled runtime was found."
                )

            from pyannote.audio import Pipeline

            local_model_path, local_model_source = self._resolve_local_model_path()
            if local_model_path is None:
                token = resolve_hf_token(hf_token)
                if not token:
                    raise DiarizationSetupError(
                        "Diarization requires a Hugging Face token the first time unless the model already exists locally."
                    )
                local_model_path = self._download_model(token)
                local_model_source = "cache"

            ensure_enterprise_ssl()
            logger.info(
                "Loading speaker diarization pipeline from %s (%s) with ffmpeg from %s (%s).",
                local_model_path,
                local_model_source or "local",
                ffmpeg_path,
                ffmpeg_source or "unknown",
            )
            pipeline = Pipeline.from_pretrained(
                str(local_model_path),
                cache_dir=str(self._cache_root),
            )
            if pipeline is None:
                raise DiarizationSetupError("pyannote.audio failed to load the speaker diarization pipeline.")

            if self._device == "cuda":
                import torch

                pipeline.to(torch.device("cuda"))

            self._pipeline = pipeline
            return self._pipeline

    def _download_model(self, token: str) -> Path:
        ensure_enterprise_ssl()

        from huggingface_hub import snapshot_download

        logger.info("Downloading diarization model '%s' into %s.", PYANNOTE_MODEL_ID, self._cache_root)
        snapshot_path = Path(
            snapshot_download(
                repo_id=PYANNOTE_MODEL_ID,
                token=token,
                cache_dir=str(self._cache_root),
            )
        )
        if not self._is_model_dir(snapshot_path):
            raise DiarizationSetupError(
                "The diarization model download completed, but the local snapshot is missing config.yaml."
            )
        return snapshot_path

    def _resolve_local_model_path(self) -> tuple[Path | None, str | None]:
        if self._is_model_dir(self._bundled_model_root):
            return self._bundled_model_root, "bundled"

        snapshot_path = self._resolve_cached_snapshot_path()
        if snapshot_path is not None:
            return snapshot_path, "cache"

        return None, None

    def _resolve_cached_snapshot_path(self) -> Path | None:
        repo_cache_root = self._cache_root / PYANNOTE_CACHE_REPO_DIR
        snapshots_root = repo_cache_root / "snapshots"
        ref_path = repo_cache_root / "refs" / "main"

        if ref_path.exists():
            revision = ref_path.read_text().strip()
            if revision:
                candidate = snapshots_root / revision
                if self._is_model_dir(candidate):
                    return candidate

        if not snapshots_root.exists():
            return None

        for candidate in sorted(snapshots_root.iterdir(), reverse=True):
            if self._is_model_dir(candidate):
                return candidate

        return None

    @staticmethod
    def _is_model_dir(path: Path) -> bool:
        return path.is_dir() and (path / "config.yaml").exists()
