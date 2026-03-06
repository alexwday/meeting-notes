from __future__ import annotations

import logging
import os
import ssl
import threading
from dataclasses import asdict, dataclass
from typing import Any

import httpx


logger = logging.getLogger(__name__)
_SSL_LOCK = threading.Lock()
_SSL_STATE: "SSLState | None" = None


@dataclass(frozen=True)
class SSLState:
    enabled: bool
    provider: str
    detail: str | None
    ssl_cert_file: str | None
    requests_ca_bundle: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_enterprise_ssl() -> SSLState:
    global _SSL_STATE

    with _SSL_LOCK:
        if _SSL_STATE is not None:
            return _SSL_STATE

        try:
            import rbc_security  # type: ignore
        except ImportError:
            _SSL_STATE = _build_local_ssl_state("rbc_security not installed")
            logger.info(
                "rbc_security is unavailable; using the local CA bundle fallback (%s).",
                _SSL_STATE.provider,
            )
            return _SSL_STATE

        try:
            rbc_security.enable_certs()
            _configure_huggingface_networking()
            _SSL_STATE = SSLState(
                enabled=True,
                provider="rbc_security",
                detail=_build_ssl_detail("enterprise certificates enabled"),
                ssl_cert_file=os.getenv("SSL_CERT_FILE"),
                requests_ca_bundle=os.getenv("REQUESTS_CA_BUNDLE"),
            )
            logger.info("Enterprise SSL certificates enabled via rbc_security.")
            return _SSL_STATE
        except Exception as exc:  # pragma: no cover - depends on enterprise runtime
            _SSL_STATE = _build_local_ssl_state(f"rbc_security failed: {exc}")
            logger.warning("rbc_security failed; falling back to the local CA bundle: %s", exc)
            return _SSL_STATE


def _build_local_ssl_state(detail: str) -> SSLState:
    cert_path: str | None = None
    provider = "system"

    try:
        import certifi

        cert_path = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", cert_path)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
        os.environ.setdefault("CURL_CA_BUNDLE", cert_path)
        provider = "certifi"
        detail = f"{detail}; using certifi CA bundle"
    except Exception as exc:  # pragma: no cover - certifi is expected but optional
        logger.warning("Local certifi bundle unavailable; relying on the system certificate store: %s", exc)

    _configure_huggingface_networking()
    return SSLState(
        enabled=True,
        provider=provider,
        detail=_build_ssl_detail(detail),
        ssl_cert_file=os.getenv("SSL_CERT_FILE"),
        requests_ca_bundle=os.getenv("REQUESTS_CA_BUNDLE"),
    )


def _build_ssl_detail(base_detail: str) -> str:
    if _should_relax_x509_strict():
        return f"{base_detail}; OpenSSL strict X.509 checks relaxed for enterprise CA compatibility{_hf_xet_detail()}"
    return f"{base_detail}{_hf_xet_detail()}"


def _hf_xet_detail() -> str:
    if _should_disable_hf_xet():
        return "; huggingface hub xet downloads disabled"
    return ""


def _should_disable_hf_xet() -> bool:
    return os.getenv("MEETING_NOTES_ALLOW_HF_XET", "").strip().lower() not in {"1", "true", "yes"}


def _should_relax_x509_strict() -> bool:
    return os.getenv("MEETING_NOTES_KEEP_X509_STRICT", "").strip().lower() not in {"1", "true", "yes"}


def _configure_huggingface_networking() -> None:
    try:
        from huggingface_hub import set_async_client_factory, set_client_factory
        from huggingface_hub import constants as hf_constants
    except Exception:
        return

    set_client_factory(lambda: create_secure_httpx_client())
    set_async_client_factory(lambda: create_secure_async_httpx_client())

    disable_xet = _should_disable_hf_xet()
    os.environ["HF_HUB_DISABLE_XET"] = "1" if disable_xet else "0"
    try:
        hf_constants.HF_HUB_DISABLE_XET = disable_xet
    except Exception:
        pass

    if disable_xet:
        logger.info("Configured huggingface_hub to use Python HTTP downloads with enterprise SSL; hf_xet disabled.")


def create_secure_httpx_client(**kwargs: Any) -> httpx.Client:
    kwargs.setdefault("verify", create_secure_ssl_context())
    return httpx.Client(**kwargs)


def create_secure_async_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
    kwargs.setdefault("verify", create_secure_ssl_context())
    return httpx.AsyncClient(**kwargs)


def create_secure_ssl_context() -> ssl.SSLContext:
    state = ensure_enterprise_ssl()
    cafile = state.requests_ca_bundle or state.ssl_cert_file
    context = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()

    # Some enterprise CA bundles fail OpenSSL's strict RFC validation in Python 3.13+ even though
    # they are otherwise trusted in the corporate environment. Keep verification enabled while
    # relaxing only the strict X.509 flag unless explicitly opted out.
    strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
    if strict_flag and _should_relax_x509_strict():
        context.verify_flags &= ~strict_flag

    return context
