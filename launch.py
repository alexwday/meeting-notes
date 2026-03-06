from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
import venv
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
MARKER_FILE = VENV_DIR / ".install-fingerprint"
DEFAULT_BOOTSTRAP_CONFIG = PROJECT_ROOT / "launch.local.json"


@dataclass(frozen=True)
class PipBootstrapSettings:
    index_urls: tuple[str, ...] = ()
    trusted_hosts: tuple[str, ...] = ()

    def fingerprint_payload(self) -> str:
        payload = {
            "index_urls": list(self.index_urls),
            "trusted_hosts": list(self.trusted_hosts),
        }
        return json.dumps(payload, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap dependencies and launch the meeting-notes transcriber web app."
    )
    parser.add_argument("--host", default=os.getenv("MEETING_NOTES_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.getenv("MEETING_NOTES_PORT", "8765"))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--bootstrap-config",
        default=os.getenv("MEETING_NOTES_BOOTSTRAP_CONFIG", str(DEFAULT_BOOTSTRAP_CONFIG)),
        help="Optional JSON file with pip bootstrap settings such as index_urls and trusted_hosts.",
    )
    parser.add_argument(
        "--index-url",
        dest="index_urls",
        action="append",
        default=[],
        help="Repeat to provide one primary pip index URL plus any extra index URLs.",
    )
    parser.add_argument(
        "--trusted-host",
        dest="trusted_hosts",
        action="append",
        default=[],
        help="Repeat to provide trusted hosts for pip installs.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def try_run_command(command: list[str]) -> bool:
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


def wait_for_server_ready(
    health_url: str,
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    *,
    timeout_seconds: float = 20.0,
    poll_interval_seconds: float = 0.25,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib_request.urlopen(health_url, timeout=1.0) as response:
                if getattr(response, "status", 200) == 200:
                    return True
        except (urllib_error.URLError, OSError):
            pass
        time.sleep(poll_interval_seconds)
    return False


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def project_fingerprint(pip_settings: PipBootstrapSettings) -> str:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_bytes()
    digest = hashlib.sha256()
    digest.update(pyproject)
    digest.update(pip_settings.fingerprint_payload().encode("utf-8"))
    return digest.hexdigest()


def ensure_virtualenv() -> Path:
    python_path = venv_python()
    if python_path.exists():
        return python_path

    print("Creating virtual environment...")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(VENV_DIR)
    return python_path


def ensure_dependencies(python_path: Path, pip_settings: PipBootstrapSettings) -> None:
    fingerprint = project_fingerprint(pip_settings)
    current = MARKER_FILE.read_text().strip() if MARKER_FILE.exists() else None
    if current == fingerprint:
        ensure_optional_rbc_security(python_path, pip_settings)
        return

    print("Installing project dependencies...")
    run_command(build_pip_install_command(python_path, pip_settings, "--upgrade", "pip"))
    run_command(build_pip_install_command(python_path, pip_settings, "-e", "."))
    ensure_optional_rbc_security(python_path, pip_settings)
    MARKER_FILE.write_text(fingerprint)


def ensure_optional_rbc_security(python_path: Path, pip_settings: PipBootstrapSettings) -> None:
    if os.getenv("MEETING_NOTES_SKIP_RBC_SECURITY_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
        return

    if module_available(python_path, "rbc_security"):
        return

    print("Attempting optional install of rbc_security...")
    installed = try_run_command(build_pip_install_command(python_path, pip_settings, "rbc_security"))
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


def load_pip_bootstrap_settings(args: argparse.Namespace) -> PipBootstrapSettings:
    config_path = Path(args.bootstrap_config).expanduser()
    file_payload = load_bootstrap_config_file(config_path)

    index_urls = resolve_bootstrap_list(
        cli_values=args.index_urls,
        env_name="MEETING_NOTES_PIP_INDEX_URLS",
        file_values=file_payload.get("index_urls"),
    )
    trusted_hosts = resolve_bootstrap_list(
        cli_values=args.trusted_hosts,
        env_name="MEETING_NOTES_PIP_TRUSTED_HOSTS",
        file_values=file_payload.get("trusted_hosts"),
    )
    return PipBootstrapSettings(
        index_urls=tuple(index_urls),
        trusted_hosts=tuple(trusted_hosts),
    )


def load_bootstrap_config_file(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}

    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Bootstrap config at {config_path} must contain a JSON object.")
    return payload


def resolve_bootstrap_list(
    *,
    cli_values: list[str],
    env_name: str,
    file_values: object,
) -> list[str]:
    if cli_values:
        return dedupe_preserve_order(cli_values)

    env_value = (os.getenv(env_name) or "").strip()
    if env_value:
        return dedupe_preserve_order(parse_bootstrap_values(env_value))

    if file_values:
        return dedupe_preserve_order(parse_bootstrap_values(file_values))

    return []


def parse_bootstrap_values(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [value.strip() for value in raw.replace("\r", "\n").replace(",", "\n").split("\n") if value.strip()]
    if isinstance(raw, list):
        values: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
        return values
    return []


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def build_pip_install_command(
    python_path: Path,
    pip_settings: PipBootstrapSettings,
    *packages_or_flags: str,
) -> list[str]:
    command = [str(python_path), "-m", "pip", "install"]
    if pip_settings.index_urls:
        command.extend(["--index-url", pip_settings.index_urls[0]])
        for extra_index_url in pip_settings.index_urls[1:]:
            command.extend(["--extra-index-url", extra_index_url])
    for trusted_host in pip_settings.trusted_hosts:
        command.extend(["--trusted-host", trusted_host])
    command.extend(packages_or_flags)
    return command


def launch_server(command: list[str], *, host: str, port: str, open_browser: bool) -> int:
    if not open_browser:
        run_command(command)
        return 0

    browser_url = f"http://{host}:{port}"
    health_url = f"{browser_url}/api/health"
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)

    try:
        if wait_for_server_ready(health_url, process):
            webbrowser.open(browser_url)
        else:
            print(f"Server did not become ready before browser launch timeout. Open {browser_url} manually if needed.")
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 130


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

    try:
        pip_settings = load_pip_bootstrap_settings(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Bootstrap configuration error: {exc}")
        return 1

    python_path = ensure_virtualenv()
    ensure_dependencies(python_path, pip_settings)

    host = str(args.host)
    port = str(args.port)

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
    return launch_server(command, host=host, port=port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
