from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from meeting_notes_transcriber.transcriber import (
    TranscriptionService,
    TranscriptionSettings,
    build_plain_text,
    format_timestamp,
    render_srt,
    render_transcript_text,
    render_vtt,
    resolve_settings,
    summarize_assigned_speakers,
)


class TranscriberFormattingTests(unittest.TestCase):
    def test_timestamp_formatting(self) -> None:
        self.assertEqual(format_timestamp(65.432), "00:01:05.432")
        self.assertEqual(format_timestamp(65.432, srt=True), "00:01:05,432")

    def test_render_outputs(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello there."},
            {"start": 1.5, "end": 3.0, "text": "General Kenobi."},
        ]

        srt = render_srt(segments)
        vtt = render_vtt(segments)

        self.assertIn("00:00:00,000 --> 00:00:01,500", srt)
        self.assertIn("WEBVTT", vtt)
        self.assertIn("General Kenobi.", vtt)

    def test_english_only_model_defaults_language(self) -> None:
        resolved = resolve_settings(
            TranscriptionSettings(
                model_key="distil-large-v3",
                custom_model_id=None,
                language=None,
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
            )
        )

        self.assertEqual(resolved.language, "en")

    def test_render_transcript_text_with_speakers(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello there.", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "General Kenobi.", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.0, "text": "You are a bold one.", "speaker": "SPEAKER_01"},
        ]

        self.assertEqual(
            render_transcript_text(segments),
            "[SPEAKER_00] Hello there. General Kenobi.\n[SPEAKER_01] You are a bold one.",
        )

    def test_no_speech_message(self) -> None:
        self.assertEqual(build_plain_text([]), "")
        self.assertEqual(render_transcript_text([]), "No speech detected.")

    def test_summarize_assigned_speakers_ignores_duplicates(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello there.", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "General Kenobi.", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.0, "text": "You are a bold one.", "speaker": "SPEAKER_01"},
            {"start": 3.0, "end": 4.0, "text": "Ignored.", "speaker": None},
        ]

        self.assertEqual(summarize_assigned_speakers(segments), ["SPEAKER_00", "SPEAKER_01"])

    def test_transcription_service_prefers_bundled_whisper_model_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundled_model_root = root / "models"
            turbo_bundle = bundled_model_root / "whisper" / "turbo"
            turbo_bundle.mkdir(parents=True)
            (turbo_bundle / "model.bin").write_text("placeholder")

            service = TranscriptionService(
                root / ".data" / "models",
                bundled_models_root=bundled_model_root,
                device="cpu",
                compute_type="int8",
            )

            self.assertEqual(
                service._resolve_model_reference("turbo", "turbo"),
                str(turbo_bundle),
            )

    def test_transcription_service_clears_retryable_model_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_root = root / ".data" / "models"
            repo_cache = cache_root / "models--Systran--faster-distil-whisper-large-v3"
            lock_cache = cache_root / ".locks" / "models--Systran--faster-distil-whisper-large-v3"
            repo_cache.mkdir(parents=True)
            lock_cache.mkdir(parents=True)

            service = TranscriptionService(
                cache_root,
                bundled_models_root=root / "models",
                device="cpu",
                compute_type="int8",
            )

            retried = service._repair_model_cache_and_retry(
                RuntimeError(
                    "cannot find the appropriate snapshot folder for the specified revision on the local disk"
                ),
                "distil-large-v3",
                "distil-large-v3",
            )

            self.assertTrue(retried)
            self.assertFalse(repo_cache.exists())
            self.assertFalse(lock_cache.exists())


if __name__ == "__main__":
    unittest.main()
