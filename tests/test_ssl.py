import os
import unittest
from unittest import mock

from meeting_notes_transcriber import ssl as ssl_module
from meeting_notes_transcriber.ssl import ensure_enterprise_ssl


class SSLTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
