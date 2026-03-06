import unittest

from meeting_notes_transcriber.app import derive_speaker_metadata_from_transcript


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


if __name__ == "__main__":
    unittest.main()
