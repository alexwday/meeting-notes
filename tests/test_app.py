from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from meeting_notes_transcriber.app import (
    JobTask,
    TranscriptionWorker,
    TranscriptionSettings,
    derive_speaker_metadata_from_transcript,
)


class AppMetadataTests(unittest.TestCase):
    def test_derive_speaker_metadata_prefers_transcript_assigned_speakers(self) -> None:
        transcript = {
            "diarization_enabled": True,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello.", "speaker": "SPEAKER_00"},
            ],
            "speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
            "num_speakers_detected": 2,
        }

        metadata = derive_speaker_metadata_from_transcript(transcript)

        self.assertEqual(metadata["speaker_labels"], ["SPEAKER_00"])
        self.assertEqual(metadata["num_speakers_detected"], 1)
        self.assertEqual(metadata["diarization_speaker_labels"], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(metadata["diarization_speaker_count"], 2)


class AppWorkerTests(unittest.TestCase):
    def test_worker_removes_uploaded_source_after_success(self) -> None:
        class FakeJobStore:
            def __init__(self) -> None:
                self.completed = threading.Event()
                self.failed = False

            def mark_running(self, *args, **kwargs) -> None:
                return None

            def update_progress(self, *args, **kwargs) -> None:
                return None

            def mark_completed(self, *args, **kwargs) -> None:
                self.completed.set()
                return None

            def mark_failed(self, *args, **kwargs) -> None:
                self.failed = True
                return None

        class FakeService:
            def transcribe(self, **kwargs):
                return (
                    {
                        "text": "Transcript text",
                        "language": "en",
                        "language_probability": 0.99,
                        "duration": 1.0,
                        "duration_after_vad": 1.0,
                        "device": "cpu",
                        "compute_type": "int8",
                        "model": "distil-large-v3",
                        "model_label": "Distil-Whisper Large v3",
                        "diarization_enabled": False,
                        "num_speakers_detected": 0,
                        "speaker_labels": [],
                        "diarization_speaker_count": 0,
                        "diarization_speaker_labels": [],
                    },
                    {},
                )

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "meeting.mp4"
            source_path.write_bytes(b"video")
            output_root = root / "job"
            output_root.mkdir()

            job_store = FakeJobStore()
            worker = TranscriptionWorker(job_store, FakeService())
            worker.start()
            try:
                worker.submit(
                    JobTask(
                        job_id="job-1",
                        source_path=source_path,
                        output_root=output_root,
                        settings=TranscriptionSettings(
                            model_key="distil-large-v3",
                            custom_model_id=None,
                            language="en",
                            task="transcribe",
                            beam_size=5,
                            temperature=0.0,
                            vad_filter=True,
                            word_timestamps=False,
                            initial_prompt=None,
                            enable_diarization=False,
                            diarization_exclusive=False,
                            num_speakers=None,
                            min_speakers=None,
                            max_speakers=None,
                        ),
                        hf_token=None,
                    )
                )

                self.assertTrue(job_store.completed.wait(timeout=2))
                self.assertFalse(job_store.failed)
                deadline = time.time() + 2
                while source_path.exists() and time.time() < deadline:
                    time.sleep(0.01)
                self.assertFalse(source_path.exists())
            finally:
                worker.stop()


if __name__ == "__main__":
    unittest.main()
