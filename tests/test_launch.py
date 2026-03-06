import os
from pathlib import Path
import unittest
from unittest import mock

import launch


class LaunchTests(unittest.TestCase):
    def test_optional_rbc_security_install_retries_on_each_launch_when_missing(self) -> None:
        python_path = Path("/tmp/python")

        with mock.patch.dict(os.environ, {"MEETING_NOTES_SKIP_RBC_SECURITY_INSTALL": ""}, clear=False):
            with mock.patch.object(launch, "module_available", return_value=False):
                with mock.patch.object(launch, "try_run_command", return_value=False) as try_run_command:
                    launch.ensure_optional_rbc_security(python_path)
                    launch.ensure_optional_rbc_security(python_path)

        self.assertEqual(try_run_command.call_count, 2)


if __name__ == "__main__":
    unittest.main()
