import os
import ssl
import unittest
from unittest import mock

from meeting_notes_transcriber import ssl as ssl_module
from meeting_notes_transcriber.ssl import create_secure_ssl_context, ensure_enterprise_ssl


class SSLTests(unittest.TestCase):
    def tearDown(self) -> None:
        ssl_module._SSL_STATE = None

    def test_ssl_wrapper_falls_back_when_package_missing(self) -> None:
        state = ensure_enterprise_ssl()
        self.assertTrue(state.enabled)
        self.assertIn(state.provider, {"certifi", "system", "rbc_security"})

    def test_ssl_wrapper_configures_huggingface_networking_and_disables_xet_by_default(self) -> None:
        ssl_module._SSL_STATE = None
        with mock.patch.dict(os.environ, {}, clear=False):
            with mock.patch("huggingface_hub.set_client_factory") as set_client_factory:
                with mock.patch("huggingface_hub.set_async_client_factory") as set_async_client_factory:
                    state = ensure_enterprise_ssl()
                    self.assertEqual(os.environ.get("HF_HUB_DISABLE_XET"), "1")

        self.assertTrue(state.enabled)
        set_client_factory.assert_called_once()
        set_async_client_factory.assert_called_once()
        self.assertIn("huggingface hub xet downloads disabled", state.detail or "")

    def test_secure_ssl_context_relaxes_strict_x509_by_default(self) -> None:
        ssl_module._SSL_STATE = ssl_module.SSLState(
            enabled=True,
            provider="rbc_security",
            detail="enterprise certificates enabled",
            ssl_cert_file=None,
            requests_ca_bundle=None,
        )

        context = create_secure_ssl_context()
        strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)

        if strict_flag:
            self.assertEqual(context.verify_flags & strict_flag, 0)

    def test_secure_ssl_context_can_keep_strict_x509_when_requested(self) -> None:
        ssl_module._SSL_STATE = ssl_module.SSLState(
            enabled=True,
            provider="rbc_security",
            detail="enterprise certificates enabled",
            ssl_cert_file=None,
            requests_ca_bundle=None,
        )

        strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
        if not strict_flag:
            self.skipTest("Python/OpenSSL does not expose VERIFY_X509_STRICT")

        with mock.patch.dict(os.environ, {"MEETING_NOTES_KEEP_X509_STRICT": "1"}, clear=False):
            context = create_secure_ssl_context()

        self.assertNotEqual(context.verify_flags & strict_flag, 0)


if __name__ == "__main__":
    unittest.main()
