from __future__ import annotations

import unittest
import uuid
import tempfile
from pathlib import Path

from voice_conversion.config import PipelineConfig
from voice_conversion.convert import (downgrade_v2_to_v1, precision_loss,
                                      restore_v1_with_sidecar,
                                      upgrade_v1_to_v2)
from voice_conversion.errors import (E_FIELD_TRUNCATED, E_UNKNOWN_TIMBRE,
                                     E_V2_SCHEMA, FailureLog, PipelineError)
from voice_conversion.model import Segment
from voice_conversion.v1 import parse_v1, render_v1
from voice_conversion.v2 import parse_v2, write_v2

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")

    def test_v1_fixed_width_round_trip(self):
        failures = FailureLog()
        segments = parse_v1(SAMPLES / "exchange" / "segments.v1.txt",
                            self.config, failures)
        self.assertEqual(len(segments), 5)
        self.assertEqual(failures.to_list(), [])
        text = render_v1(segments, self.config)
        again = parse_v1_path(text, self.config)
        self.assertEqual([s.index for s in again], [0, 1, 2, 3, 4])
        self.assertEqual(again[0].intensity_permille, 825)
        self.assertEqual(again[0].timbre, "alto")

    def test_v1_truncation_reports_line_and_field(self):
        failures = FailureLog()
        parse_v1(SAMPLES / "exchange" / "segments_broken.v1.txt",
                 self.config, failures)
        rows = failures.to_list()
        self.assertTrue(any(f["code"] == E_FIELD_TRUNCATED for f in rows))
        bad = next(f for f in rows if f["code"] == E_FIELD_TRUNCATED)
        self.assertGreaterEqual(bad["line"], 1)
        self.assertTrue(bad["field"])

    def test_unknown_timbre_code(self):
        line = "     0       0      60   500   999  "
        failures = FailureLog()
        segments = parse_v1_path(line, self.config, failures)
        self.assertEqual(segments, [])
        self.assertEqual(failures.sorted()[0].code, E_UNKNOWN_TIMBRE)

    def test_v2_schema_and_unknown_label(self):
        good = write_temp_v2([{
            "segment": 0, "start_frame": 0.0, "end_frame": 10.0,
            "intensity": 0.5, "timbre": "alto", "passthrough": {},
        }])
        failures = FailureLog()
        parsed = parse_v2(good, self.config, failures)
        self.assertEqual(len(parsed), 1)
        bad = write_temp_v2([{
            "segment": 1, "start_frame": 0.0, "end_frame": 10.0,
            "intensity": 0.5, "timbre": "clarinet", "passthrough": {},
        }])
        failures = FailureLog()
        parse_v2(bad, self.config, failures)
        self.assertEqual(failures.sorted()[0].code, E_UNKNOWN_TIMBRE)
        broken = write_raw_v2([{"segment": 2}])
        failures = FailureLog()
        parse_v2(broken, self.config, failures)
        self.assertEqual(failures.sorted()[0].code, E_V2_SCHEMA)

    def test_upgrade_is_exact(self):
        failures = FailureLog()
        v1_segments = parse_v1(SAMPLES / "exchange" / "segments.v1.txt",
                               self.config, failures)
        upgraded = upgrade_v1_to_v2(v1_segments, self.config)
        for before, after in zip(v1_segments, upgraded):
            self.assertEqual(after.intensity,
                             before.intensity_permille / 1000.0)
            self.assertEqual(after.timbre, before.timbre)
            self.assertEqual(after.start_frame, float(before.start_frame))
            self.assertIn("exact", after.precision_tag)

    def test_downgrade_sidecar_restores_everything(self):
        failures = FailureLog()
        v2_segments = parse_v2(SAMPLES / "exchange" / "segments.v2.json",
                               self.config, failures)
        downgraded, sidecar = downgrade_v2_to_v1(v2_segments, self.config)
        loss = precision_loss(v2_segments, downgraded)
        self.assertGreater(loss["start_frame"]["max_abs"], 0.0)
        restored = restore_v1_with_sidecar(downgraded, sidecar)
        for before, after in zip(v2_segments, restored):
            self.assertEqual(before.start_frame, after.start_frame)
            self.assertEqual(before.end_frame, after.end_frame)
            self.assertEqual(before.intensity, after.intensity)
            self.assertEqual(before.timbre, after.timbre)
            self.assertEqual(before.passthrough, after.passthrough)

    def test_oversized_v1_value_raises_pipeline_error(self):
        seg = Segment(index=0, start_frame=0.0, end_frame=10.0,
                      intensity=0.5, timbre="alto", source_version="v1",
                      intensity_permille=500)
        seg.end_frame = 123456789.0
        with self.assertRaises(PipelineError):
            render_v1([seg], self.config)


def parse_v1_path(text_or_path, config, failures=None):
    if failures is None:
        failures = FailureLog()
    path = text_or_path
    if not isinstance(text_or_path, Path):
        tmp = Path(tempfile.gettempdir()) / f"_tmp_v1_{uuid.uuid4().hex}.txt"
        tmp.write_text(text_or_path, encoding="utf-8")
        path = tmp
    return parse_v1(path, config, failures)


def write_temp_v2(segments):
    tmp = Path(tempfile.gettempdir()) / f"_tmp_v2_{uuid.uuid4().hex}.json"
    write_v2(tmp, [
        Segment(
            index=e["segment"], start_frame=e["start_frame"],
            end_frame=e["end_frame"], intensity=e["intensity"],
            timbre=e["timbre"], source_version="v2",
            passthrough=e.get("passthrough", {}),
        ) for e in segments
    ])
    return tmp


def write_raw_v2(entries):
    tmp = Path(tempfile.gettempdir()) / f"_tmp_raw_{uuid.uuid4().hex}.json"
    import json
    tmp.write_text(json.dumps(
        {"schema": "vc-exchange-v2", "segments": entries}),
        encoding="utf-8")
    return tmp


if __name__ == "__main__":
    unittest.main()
