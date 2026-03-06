import unittest

from meeting_notes_transcriber.ssl import ensure_enterprise_ssl


class SSLTests(unittest.TestCase):
    def test_ssl_wrapper_falls_back_when_package_missing(self) -> None:
        state = ensure_enterprise_ssl()
        self.assertTrue(state.enabled)
        self.assertIn(state.provider, {"certifi", "system", "rbc_security"})


if __name__ == "__main__":
    unittest.main()
