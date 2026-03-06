from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import textwrap
import venv
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
MARKER_FILE = VENV_DIR / ".install-fingerprint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap dependencies and launch the meeting-notes transcriber web app."
    )
    parser.add_argument("--host", default=os.getenv("MEETING_NOTES_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.getenv("MEETING_NOTES_PORT", "8765"))
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def try_run_command(command: list[str]) -> bool:
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def project_fingerprint() -> str:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_bytes()
    return hashlib.sha256(pyproject).hexdigest()


def ensure_virtualenv() -> Path:
    python_path = venv_python()
    if python_path.exists():
        return python_path

    print("Creating virtual environment...")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(VENV_DIR)
    return python_path


def ensure_dependencies(python_path: Path) -> None:
    fingerprint = project_fingerprint()
    current = MARKER_FILE.read_text().strip() if MARKER_FILE.exists() else None
    if current == fingerprint:
        ensure_optional_rbc_security(python_path)
        return

    print("Installing project dependencies...")
    run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(python_path), "-m", "pip", "install", "-e", "."])
    ensure_optional_rbc_security(python_path)
    MARKER_FILE.write_text(fingerprint)


def ensure_optional_rbc_security(python_path: Path) -> None:
    if os.getenv("MEETING_NOTES_SKIP_RBC_SECURITY_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
        return

    if module_available(python_path, "rbc_security"):
        return

    print("Attempting optional install of rbc_security...")
    installed = try_run_command([str(python_path), "-m", "pip", "install", "rbc_security"])
    if installed:
        print("Installed rbc_security.")
    else:
        print("rbc_security not available in the current pip sources. Will retry on the next launch.")


def module_available(python_path: Path, module_name: str) -> bool:
    result = subprocess.run(
        [str(python_path), "-c", f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{module_name}') else 1)"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    args = parse_args()

    if sys.version_info < (3, 11):
        message = textwrap.dedent(
            """
            Python 3.11 or newer is required.
            Install Python 3.11+ and re-run `python launch.py`.
            """
        ).strip()
        print(message)
        return 1

    python_path = ensure_virtualenv()
    ensure_dependencies(python_path)

    host = str(args.host)
    port = str(args.port)

    if not args.no_browser:
        webbrowser.open(f"http://{host}:{port}")

    command = [
        str(python_path),
        "-m",
        "uvicorn",
        "meeting_notes_transcriber.app:app",
        "--app-dir",
        "src",
        "--host",
        host,
        "--port",
        port,
    ]
    run_command(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
