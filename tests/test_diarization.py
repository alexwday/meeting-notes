from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from meeting_notes_transcriber.diarization import (
    DiarizationSettings,
    LocalSpeakerDiarizer,
    _reset_ffmpeg_resolution_cache,
    assign_speakers_to_segments,
    validate_diarization_settings,
)


class DiarizationTests(unittest.TestCase):
    def test_assign_speakers_by_overlap(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello."},
            {"start": 1.5, "end": 3.0, "text": "Hi."},
        ]
        diarization_turns = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 3.0, "speaker": "SPEAKER_01"},
        ]

        assigned = assign_speakers_to_segments(segments, diarization_turns)

        self.assertEqual(assigned[0]["speaker"], "SPEAKER_00")
        self.assertEqual(assigned[1]["speaker"], "SPEAKER_01")

    def test_assign_speakers_uses_nearest_turn_as_fallback(self) -> None:
        segments = [{"start": 3.1, "end": 3.4, "text": "Thanks."}]
        diarization_turns = [{"start": 2.0, "end": 3.0, "speaker": "SPEAKER_02"}]

        assigned = assign_speakers_to_segments(segments, diarization_turns, fallback_gap=0.2)

        self.assertEqual(assigned[0]["speaker"], "SPEAKER_02")

    def test_validate_diarization_settings_rejects_exact_and_range(self) -> None:
        with self.assertRaises(ValueError):
            validate_diarization_settings(
                DiarizationSettings(
                    enabled=True,
                    exclusive=True,
                    num_speakers=3,
                    min_speakers=2,
                    max_speakers=4,
                )
            )

    def test_status_prefers_bundled_local_model(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundled_model_root = root / "models" / "pyannote-speaker-diarization-community-1"
            bundled_model_root.mkdir(parents=True)
            (bundled_model_root / "config.yaml").write_text("pipeline: {}")

            diarizer = LocalSpeakerDiarizer(
                root / ".data" / "pyannote",
                bundled_model_root=bundled_model_root,
                device="cpu",
            )

            status = diarizer.status()
            self.assertTrue(status["local_model_ready"])
            self.assertEqual(status["local_model_source"], "bundled")
            self.assertEqual(status["local_model_path"], str(bundled_model_root))

    def test_status_uses_cached_snapshot_when_bundled_model_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_root = root / ".data" / "pyannote"
            snapshot_root = (
                cache_root
                / "models--pyannote--speaker-diarization-community-1"
                / "snapshots"
                / "revision123"
            )
            snapshot_root.mkdir(parents=True)
            (snapshot_root / "config.yaml").write_text("pipeline: {}")
            refs_root = cache_root / "models--pyannote--speaker-diarization-community-1" / "refs"
            refs_root.mkdir(parents=True)
            (refs_root / "main").write_text("revision123")

            diarizer = LocalSpeakerDiarizer(
                cache_root,
                bundled_model_root=root / "models" / "pyannote-speaker-diarization-community-1",
                device="cpu",
            )

            status = diarizer.status()
            self.assertTrue(status["local_model_ready"])
            self.assertEqual(status["local_model_source"], "cache")
            self.assertEqual(status["local_model_path"], str(snapshot_root))

    def test_status_uses_bundled_ffmpeg_runtime_when_system_ffmpeg_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundled_model_root = root / "models" / "pyannote-speaker-diarization-community-1"
            bundled_model_root.mkdir(parents=True)
            (bundled_model_root / "config.yaml").write_text("pipeline: {}")

            bundled_ffmpeg = root / "vendor" / "ffmpeg"
            bundled_ffmpeg.parent.mkdir(parents=True)
            bundled_ffmpeg.write_text("placeholder")

            diarizer = LocalSpeakerDiarizer(
                root / ".data" / "pyannote",
                bundled_model_root=bundled_model_root,
                device="cpu",
            )

            with mock.patch("meeting_notes_transcriber.diarization.shutil.which", return_value=None):
                with mock.patch("imageio_ffmpeg.get_ffmpeg_exe", return_value=str(bundled_ffmpeg)):
                    _reset_ffmpeg_resolution_cache()
                    status = diarizer.status()

            self.assertTrue(status["ffmpeg_available"])
            self.assertEqual(status["ffmpeg_source"], "imageio-ffmpeg")
            self.assertEqual(status["ffmpeg_path"], str(bundled_ffmpeg))


if __name__ == "__main__":
    unittest.main()
