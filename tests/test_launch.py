import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import launch


class LaunchTests(unittest.TestCase):
    def test_optional_rbc_security_install_retries_on_each_launch_when_missing(self) -> None:
        python_path = Path("/tmp/python")
        pip_settings = launch.PipBootstrapSettings()

        with mock.patch.dict(os.environ, {"MEETING_NOTES_SKIP_RBC_SECURITY_INSTALL": ""}, clear=False):
            with mock.patch.object(launch, "module_available", return_value=False):
                with mock.patch.object(launch, "try_run_command", return_value=False) as try_run_command:
                    launch.ensure_optional_rbc_security(python_path, pip_settings)
                    launch.ensure_optional_rbc_security(python_path, pip_settings)

        self.assertEqual(try_run_command.call_count, 2)

    def test_build_pip_install_command_uses_primary_and_extra_indexes(self) -> None:
        command = launch.build_pip_install_command(
            Path("/tmp/python"),
            launch.PipBootstrapSettings(
                index_urls=("https://primary.example/simple", "https://secondary.example/simple"),
                trusted_hosts=("primary.example", "secondary.example"),
            ),
            "-e",
            ".",
        )

        self.assertEqual(
            command,
            [
                "/tmp/python",
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://primary.example/simple",
                "--extra-index-url",
                "https://secondary.example/simple",
                "--trusted-host",
                "primary.example",
                "--trusted-host",
                "secondary.example",
                "-e",
                ".",
            ],
        )

    def test_load_pip_bootstrap_settings_reads_local_config_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "launch.local.json"
            config_path.write_text(
                """
                {
                  "index_urls": [
                    "https://primary.example/simple",
                    "https://secondary.example/simple"
                  ],
                  "trusted_hosts": [
                    "primary.example",
                    "secondary.example"
                  ]
                }
                """.strip()
            )
            namespace = mock.Mock(
                bootstrap_config=str(config_path),
                index_urls=[],
                trusted_hosts=[],
            )

            settings = launch.load_pip_bootstrap_settings(namespace)

        self.assertEqual(
            settings,
            launch.PipBootstrapSettings(
                index_urls=("https://primary.example/simple", "https://secondary.example/simple"),
                trusted_hosts=("primary.example", "secondary.example"),
            ),
        )

    def test_wait_for_server_ready_returns_true_when_healthcheck_succeeds(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        response = mock.MagicMock()
        response.__enter__.return_value.status = 200

        with mock.patch("launch.urllib_request.urlopen", return_value=response):
            ready = launch.wait_for_server_ready(
                "http://127.0.0.1:8765/api/health",
                process,
                timeout_seconds=0.1,
                poll_interval_seconds=0.0,
            )

        self.assertTrue(ready)

    def test_wait_for_server_ready_returns_false_if_process_exits(self) -> None:
        process = mock.Mock()
        process.poll.return_value = 1

        ready = launch.wait_for_server_ready(
            "http://127.0.0.1:8765/api/health",
            process,
            timeout_seconds=0.1,
            poll_interval_seconds=0.0,
        )

        self.assertFalse(ready)


if __name__ == "__main__":
    unittest.main()
