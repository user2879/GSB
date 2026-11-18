from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dialogwb.lexicon import load_lexicon
from dialogwb.loader import load_jsonl


ROOT = Path(__file__).resolve().parents[1]


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lexicon = load_lexicon(ROOT / "samples" / "lexicon.json")
        cls.loaded = load_jsonl(ROOT / "samples" / "events.jsonl", cls.lexicon)

    def test_required_business_codes_and_continuation(self) -> None:
        codes = [failure["code"] for failure in self.loaded.failures]
        self.assertEqual(
            codes,
            [
                "E_BAD_JSON",
                "E_EMPTY_UTTERANCE",
                "E_UNKNOWN_TOKEN",
                "E_TURN_GAP",
            ],
        )
        turns = {
            session_id: [
                event["turn"]
                for event in self.loaded.events
                if event["session_id"] == session_id
            ]
            for session_id in ("s90_empty", "s91_unknown", "s92_gap")
        }
        self.assertEqual(turns["s90_empty"], [1, 3])
        self.assertEqual(turns["s91_unknown"], [2])
        self.assertEqual(turns["s92_gap"], [1, 4])

    def test_token_spans_are_original_character_offsets(self) -> None:
        tokens, unknown = self.lexicon.split("we review deadlines.")
        self.assertEqual(unknown, [])
        self.assertEqual(
            [(item.token, item.start, item.end) for item in tokens],
            [
                ("we", 0, 2),
                ("review", 3, 9),
                ("dead", 10, 14),
                ("line", 14, 18),
                ("s", 18, 19),
            ],
        )

    def test_unknown_token_message_has_offsets(self) -> None:
        failure = next(
            item for item in self.loaded.failures if item["code"] == "E_UNKNOWN_TOKEN"
        )
        self.assertEqual(failure["message"], "量子@2:4")
        self.assertEqual(failure["stage"], "split")
        self.assertEqual(failure["session_id"], "s91_unknown")

    def test_invalid_timestamp_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            records = [
                {
                    "session_id": "s",
                    "turn": 1,
                    "speaker": "a",
                    "text": "团队讨论预算。",
                    "timestamp": "2026-11-01T00:00:00Z",
                },
                {
                    "session_id": "s",
                    "turn": 2,
                    "speaker": "a",
                    "text": "用户反馈问题。",
                    "timestamp": "not-a-time",
                },
                {
                    "session_id": "s",
                    "turn": 3,
                    "speaker": "a",
                    "text": "负责人跟进测试。",
                    "timestamp": "2026-11-01T00:02:00Z",
                },
            ]
            path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in records)
                + "\n",
                encoding="utf-8",
            )
            loaded = load_jsonl(path, self.lexicon)
            self.assertEqual([event["turn"] for event in loaded.events], [1, 3])
            self.assertEqual(loaded.failures[0]["code"], "E_BAD_TIMESTAMP")


if __name__ == "__main__":
    unittest.main()

