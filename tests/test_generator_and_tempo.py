import unittest
import tempfile
from pathlib import Path

from score_pipeline.generator import generate_score, write_score_bundle
from score_pipeline.model import content_hash, read_json
from score_pipeline.tempo import TempoMap


class GeneratorTests(unittest.TestCase):
    def test_deterministic_byte_identical(self):
        first = generate_score(seed=42).to_dict()
        second = generate_score(seed=42).to_dict()
        self.assertEqual(content_hash(first), content_hash(second))
        different = generate_score(seed=43).to_dict()
        self.assertNotEqual(content_hash(first), content_hash(different))

    def test_bundle_files_deterministic(self):
        import tempfile

        with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
            report_a = write_score_bundle(generate_score(seed=7), dir_a)
            report_b = write_score_bundle(generate_score(seed=7), dir_b)
            self.assertEqual(report_a, report_b)
            for name in ("score.json", "generation_report.json"):
                self.assertEqual(
                    Path(dir_a, name).read_bytes(), Path(dir_b, name).read_bytes()
                )

    def test_report_counts_and_beat_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = write_score_bundle(generate_score(seed=9), tmp)
            score = read_json(Path(tmp) / "score.json")
            note_total = sum(len(part["notes"]) for part in score["parts"])
            self.assertEqual(report["note_event_total"], note_total)
            self.assertEqual(report["part_count"], len(score["parts"]))
            self.assertEqual(report["beat_unit_ms"], 500.0)
            self.assertGreater(report["total_beats"], 0)

    def test_tempo_piecewise_linear(self):
        points = [
            {"beat": 0.0, "bpm": 120.0},
            {"beat": 4.0, "bpm": 60.0},
        ]
        tempo = TempoMap(points)
        # 0 拍：0ms；段内平均 bpm=90 -> 4 拍 = 4*60000/90
        self.assertEqual(tempo.beat_to_ms_r(0.0), 0.0)
        self.assertAlmostEqual(tempo.beat_to_ms_r(4.0), 4 * 60000.0 / 90.0, places=6)
        self.assertAlmostEqual(
            tempo.beat_to_ms_r(8.0),
            4 * 60000.0 / 90.0 + 4 * 60000.0 / 60.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
