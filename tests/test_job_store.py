from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from meeting_notes_transcriber.job_store import INTERRUPTED_JOB_MESSAGE, JobStore


class JobStoreTests(unittest.TestCase):
    def test_progress_tracking_sets_started_at_and_completion_percent(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir))
            job = store.create(
                job_id="job-1",
                original_filename="meeting.mp4",
                stored_path="/tmp/meeting.mp4",
                settings={},
            )

            self.assertEqual(job.progress_percent, 0.0)
            self.assertIsNone(job.started_at)

            running_job = store.mark_running("job-1", "Preparing transcription", progress_percent=1.0)
            self.assertEqual(running_job.progress_percent, 1.0)
            self.assertIsNotNone(running_job.started_at)

            progress_job = store.update_progress("job-1", message="Transcribing audio", progress_percent=42.6)
            self.assertEqual(progress_job.progress_percent, 42.6)

            completed_job = store.mark_completed(
                "job-1",
                message="Done",
                transcript_preview="hello",
                metadata={},
                outputs={},
            )
            self.assertEqual(completed_job.progress_percent, 100.0)

    def test_reloads_running_jobs_as_failed_after_restart(self) -> None:
        with TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir)
            store = JobStore(jobs_root)
            store.create(
                job_id="job-2",
                original_filename="meeting.mp4",
                stored_path="/tmp/meeting.mp4",
                settings={},
            )
            store.mark_running("job-2", "Transcribing audio", progress_percent=55.0)

            reloaded = JobStore(jobs_root).get("job-2")

            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.status, "failed")
            self.assertEqual(reloaded.status_message, INTERRUPTED_JOB_MESSAGE)
            self.assertEqual(reloaded.error, INTERRUPTED_JOB_MESSAGE)
            self.assertEqual(reloaded.progress_percent, 55.0)


if __name__ == "__main__":
    unittest.main()
