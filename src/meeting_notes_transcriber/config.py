from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


@dataclass(frozen=True)
class AppConfig:
    data_root: Path
    upload_root: Path
    jobs_root: Path
    model_cache_root: Path
    pyannote_cache_root: Path
    bundled_models_root: Path
    bundled_pyannote_model_root: Path
    matplotlib_root: Path
    static_root: Path
    host: str
    port: int

    def ensure_directories(self) -> None:
        for path in (
            self.data_root,
            self.upload_root,
            self.jobs_root,
            self.model_cache_root,
            self.pyannote_cache_root,
            self.matplotlib_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    data_root = Path(os.getenv("MEETING_NOTES_DATA_DIR", PROJECT_ROOT / ".data"))
    return AppConfig(
        data_root=data_root,
        upload_root=data_root / "uploads",
        jobs_root=data_root / "jobs",
        model_cache_root=data_root / "models",
        pyannote_cache_root=data_root / "pyannote",
        bundled_models_root=PROJECT_ROOT / "models",
        bundled_pyannote_model_root=PROJECT_ROOT / "models" / "pyannote-speaker-diarization-community-1",
        matplotlib_root=data_root / "matplotlib",
        static_root=PACKAGE_ROOT / "static",
        host=os.getenv("MEETING_NOTES_HOST", "127.0.0.1"),
        port=int(os.getenv("MEETING_NOTES_PORT", "8765")),
    )
