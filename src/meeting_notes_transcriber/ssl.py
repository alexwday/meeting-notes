from __future__ import annotations

import logging
import os
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
            _SSL_STATE = SSLState(
                enabled=True,
                provider="rbc_security",
                detail="enterprise certificates enabled",
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

    return SSLState(
        enabled=True,
        provider=provider,
        detail=detail,
        ssl_cert_file=os.getenv("SSL_CERT_FILE"),
        requests_ca_bundle=os.getenv("REQUESTS_CA_BUNDLE"),
    )


def create_secure_httpx_client(**kwargs: Any) -> httpx.Client:
    state = ensure_enterprise_ssl()
    kwargs.setdefault("verify", state.requests_ca_bundle or state.ssl_cert_file or True)
    return httpx.Client(**kwargs)


def create_secure_async_httpx_client(**kwargs: Any) -> httpx.AsyncClient:
    state = ensure_enterprise_ssl()
    kwargs.setdefault("verify", state.requests_ca_bundle or state.ssl_cert_file or True)
    return httpx.AsyncClient(**kwargs)
