from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vertical_ocr.config import default_config, load_command_stream, validate_config
from vertical_ocr.errors import BusinessError
from vertical_ocr.exchange import (
    V1_WIDTH,
    convert_text,
    downgrade_v3_to_v1,
    parse_v1,
    parse_v2,
    parse_v3,
    to_v2,
    to_v3,
)
from vertical_ocr.generator import generate_workspace
from vertical_ocr.model import ExchangeRecord
from vertical_ocr.reading import build_reading_sequence, choose_authoritative_keypoint
from vertical_ocr.scheduler import BudgetScheduler
from vertical_ocr.validation import load_dataset
from vertical_ocr.vision import split_tall_segment


class ExchangeTests(unittest.TestCase):
    def test_v1_width_and_parse(self):
        line = "0001" + "0002" + "".join(str(v).zfill(5) for v in [10, 20, 110, 220]) + "987".zfill(5) + "0010"
        self.assertEqual(len(line), V1_WIDTH)
        rec = parse_v1(line + "\n")[0]
        self.assertEqual(rec.image_no, 1)
        self.assertEqual(rec.bbox, [10, 20, 110, 220])
        self.assertAlmostEqual(rec.confidence, 0.987)
        self.assertEqual(rec.reading_order, "rtl_top_to_bottom")

    def test_field_errors_include_version_and_field(self):
        with self.assertRaises(BusinessError) as ctx:
            parse_v1("short\n")
        self.assertEqual(ctx.exception.code, "E_FIELD_TRUNCATED")
        self.assertEqual(ctx.exception.version, "v1")

        with self.assertRaises(BusinessError) as ctx:
            parse_v2(json.dumps({"records": [{"image_no": 1}]}))
        self.assertEqual(ctx.exception.code, "E_FIELD_TRUNCATED")
        self.assertEqual(ctx.exception.field, "char_no")

        with self.assertRaises(BusinessError) as ctx:
            parse_v3("# vertical-ocr/v3\nimage_no=1\n")
        self.assertEqual(ctx.exception.code, "E_FIELD_TRUNCATED")
        self.assertEqual(ctx.exception.version, "v3")

    def test_downgrade_missing_fields_stay_explicit(self):
        rec = ExchangeRecord(image_no=1, char_no=2, source_version="v3")
        text, _ = downgrade_v3_to_v1([rec])
        parsed = parse_v1(text)[0]
        self.assertIn("confidence", parsed.missing_fields)
        self.assertIn("char", parsed.missing_fields)

    def test_all_three_versions_convert_each_direction(self):
        rec = ExchangeRecord(
            image_no=1, char_no=2, bbox=[11, 22, 33, 44], confidence=0.987,
            reading_order="rtl_top_to_bottom", char="天", keypoint_version="kpv-1",
            keypoint_x=22, keypoint_y=33, annotation_version="ann-1", source_version="v3",
        )
        v3 = to_v3([rec])
        v2 = to_v2([rec])
        v1 = convert_text("v3", "v1", v3)
        self.assertEqual(len(parse_v1(v1)), 1)
        self.assertEqual(len(parse_v2(convert_text("v1", "v2", v1))), 1)
        self.assertEqual(len(parse_v3(convert_text("v2", "v3", v2))), 1)
        self.assertEqual(len(parse_v1(convert_text("v2", "v1", v2))), 1)
        self.assertEqual(len(parse_v3(convert_text("v1", "v3", v1))), 1)
        self.assertEqual(convert_text("v3", "v2", v3), v2)


class ReadingTests(unittest.TestCase):
    def test_authority_version_time_lex(self):
        subs = [
            {"image": "a.png", "block_id": "b", "version_no": 2, "version_name": "v2z", "annotated_ms": 100, "annotator": "z", "x": 1, "y": 2},
            {"image": "a.png", "block_id": "b", "version_no": 2, "version_name": "v2a", "annotated_ms": 100, "annotator": "a", "x": 3, "y": 4},
            {"image": "a.png", "block_id": "b", "version_no": 5, "version_name": "v5", "annotated_ms": 1, "annotator": "low", "x": 5, "y": 6},
        ]
        winner = choose_authoritative_keypoint(subs, "a.png", "b")
        self.assertEqual(winner["version_no"], 5)

        subs[2].update({"version_no": 1, "annotated_ms": 1})
        winner = choose_authoritative_keypoint(subs, "a.png", "b")
        self.assertEqual(winner["annotator"], "a")

    def test_conflict_types(self):
        records = [
            {"character_id": "top", "bbox": [0, 0, 10, 10], "rtl_index": 0,
             "truth_column_id": "c", "truth_reading_no": 2},
            {"character_id": "bottom", "bbox": [0, 20, 10, 30], "rtl_index": 0,
             "truth_column_id": "c", "truth_reading_no": 1},
        ]
        result = build_reading_sequence("a.png", records, [])
        self.assertEqual(result["conflict_counts"]["wrong_order"], 1)

        records = [
            {"character_id": "right", "bbox": [100, 0, 110, 10], "rtl_index": 0,
             "truth_column_id": "right", "truth_reading_no": 4},
            {"character_id": "left", "bbox": [0, 0, 10, 10], "rtl_index": 1,
             "truth_column_id": "left", "truth_reading_no": 1},
        ]
        result = build_reading_sequence("a.png", records, [])
        self.assertEqual(result["conflict_counts"]["cross_column"], 1)


class VisionSchedulerTests(unittest.TestCase):
    def test_dp_equal_cost_prefers_upper_cut(self):
        cfg = validate_config({**default_config(), "char_min_height": 40, "char_max_height": 80,
                               "split_penalty": 0.2})
        projection = np.zeros(180, dtype=float)
        segments, meta = split_tall_segment(projection, 0, 180, cfg)
        self.assertTrue(meta["split"])
        self.assertEqual(meta["cuts"][0], 40)
        self.assertEqual([b - a for a, b in segments][0], 40)

    def test_budget_pending_priority(self):
        with tempfile.TemporaryDirectory() as td:
            queue = Path(td) / "queue.json"
            items = [{"work_id": f"w{i}", "image": "a.png", "rtl_index": 0, "char_index": i,
                      "point_cost": 10} for i in range(3)]
            scheduler = BudgetScheduler(25, queue, max_workers=2)
            admitted, pending = scheduler.admit(items)
            self.assertEqual([x["work_id"] for x in admitted], ["w0", "w1"])
            self.assertEqual([x["work_id"] for x in pending], ["w2"])

            scheduler2 = BudgetScheduler(100, queue, max_workers=2)
            admitted, pending = scheduler2.admit(items)
            self.assertEqual(admitted[0]["work_id"], "w2")
            self.assertEqual(pending, [])


class ValidationConfigTests(unittest.TestCase):
    def test_generated_dataset_is_valid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            generate_workspace(root)
            cfg = validate_config(default_config())
            dataset = load_dataset(root, cfg)
            self.assertEqual(dataset["failures"], [])
            self.assertEqual(len(dataset["images"]), 3)

    def test_bad_config_codes(self):
        with self.assertRaises(BusinessError) as ctx:
            validate_config({**default_config(), "sliding_window": {"width": 13, "height": 24}})
        self.assertEqual(ctx.exception.code, "E_INPUT_INVALID")
        with self.assertRaises(BusinessError) as ctx:
            validate_config({**default_config(), "template_version": "nope"})
        self.assertEqual(ctx.exception.code, "E_TEMPLATE_VERSION_UNKNOWN")

    def test_command_stream_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "commands.json"
            path.write_text(json.dumps({"commands": [
                {"op": "remove_block", "image": "a.png", "block_id": "b1"},
            ]}), encoding="utf-8")
            self.assertEqual(len(load_command_stream(path)), 1)
            path.write_text(json.dumps({"commands": [{"op": "mystery"}]}), encoding="utf-8")
            with self.assertRaises(BusinessError) as ctx:
                load_command_stream(path)
            self.assertEqual(ctx.exception.code, "E_INPUT_INVALID")


if __name__ == "__main__":
    unittest.main()
