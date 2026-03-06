from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeProfile:
    os_name: str
    os_version: str
    machine: str
    python_version: str
    accelerator: str
    device: str
    compute_type: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_runtime_profile() -> RuntimeProfile:
    notes: list[str] = []
    accelerator = "CPU"
    device = "cpu"
    compute_type = "int8"

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            accelerator = "CUDA"
            device = "cuda"
            compute_type = "float16"
            notes.append("CUDA detected; GPU inference will be preferred.")
        else:
            notes.append("No CUDA device detected; using CPU int8 inference.")
    except Exception:
        notes.append("CTranslate2 runtime details unavailable; defaulting to CPU int8 inference.")

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        notes.append("Apple Silicon is supported, but this first build keeps one shared faster-whisper backend for both Mac and Windows.")

    return RuntimeProfile(
        os_name=platform.system(),
        os_version=platform.version(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        accelerator=accelerator,
        device=device,
        compute_type=compute_type,
        notes=notes,
    )
