from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from subtitle_engine.errors import EngineError
from subtitle_engine.incremental import build_incremental_plan
from subtitle_engine.layout import build_event_layout
from subtitle_engine.spec import load_spec

from tests.helpers import clone, description, event, run_case, run_cli


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _expect(self, mutate, code: str) -> None:
        value = description()
        mutate(value)
        result = run_cli(self.root, f"invalid-{code}", value)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], code)
        self.assertFalse((self.root / f"invalid-{code}" / "output" / "frames.npy").exists())

    def test_business_codes(self) -> None:
        cases = [
            (lambda d: d["events"][0].update(end_ms=0), "E_TIME"),
            (lambda d: d["events"][0].update(text="   "), "E_TEXT"),
            (lambda d: d["events"][0].update(max_chars_per_sec=0), "E_RATE"),
            (
                lambda d: d["events"][0].update(
                    anchor={"x": 1, "y": 1, "horizontal": "left", "vertical": "top"}
                ),
                "E_ANCHOR",
            ),
            (lambda d: d["style"].update(font_size=0), "E_STYLE_FONT"),
            (lambda d: d["style"].update(line_height=0), "E_STYLE_LINE_HEIGHT"),
            (lambda d: d["style"].update(max_lines=0), "E_STYLE_MAX_LINES"),
            (lambda d: d["style"].update(contrast_enhance="yes"), "E_STYLE_CONTRAST"),
            (lambda d: d["style"].update(mask_opacity=2), "E_STYLE_MASK"),
        ]
        for mutate, code in cases:
            with self.subTest(code=code):
                self._expect(mutate, code)


class LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_wrap_is_deterministic_and_rate_is_reported(self) -> None:
        text = "One two three four five six seven eight nine ten"
        value = description(events=[event("e1", 0, 500, text, rate=10)], duration_ms=500)
        path = self.root / "wrap.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        spec = load_spec(path)
        first = build_event_layout(spec.events[0], spec.style, spec.safe_area)
        second = build_event_layout(spec.events[0], spec.style, spec.safe_area)
        self.assertEqual(first.lines, second.lines)
        self.assertEqual(first.line_widths, second.line_widths)
        self.assertTrue(all(width <= first.wrap_width for width in first.line_widths))
        self.assertGreater(len(first.lines), 1)
        self.assertTrue(first.speed_limited)
        self.assertEqual(first.reading_chars, len(text))
        self.assertEqual(first.min_display_ms, math.ceil(len(text) * 1000 / 10))
        joined = " ".join(first.lines)
        for token in text.split():
            self.assertIn(token, joined)

    def test_overflow_refuses_to_truncate(self) -> None:
        value = description(
            events=[event("e1", 0, 400, "One two three four five six seven eight")],
            style={"max_lines": 1},
        )
        path = self.root / "overflow.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        spec = load_spec(path)
        with self.assertRaises(EngineError) as raised:
            build_event_layout(spec.events[0], spec.style, spec.safe_area)
        self.assertEqual(raised.exception.code, "E_LINE_OVERFLOW")

    def test_visibility_background_mask_and_critical_marker(self) -> None:
        value = description(
            events=[event("e1", 60, 140, "Critical", critical=True)],
            duration_ms=200,
            style={"contrast_enhance": True, "mask_opacity": 0.6},
        )
        result = run_case(self.root, "visibility", value)
        self.assertEqual(result.status, "complete")
        frames = np.load(result.frames_path)
        background = value["frame"]["background"]
        for index in (0, 1, 2, 7, 8, 9):
            self.assertTrue(np.all(frames[index] == background), index)
        self.assertTrue(np.any(frames[3] != background))
        subtitle = result.report["subtitles"][0]
        self.assertEqual((subtitle["bbox_start_frame"], subtitle["bbox_end_frame"]), (3, 7))
        x, y, width, height = subtitle["bbox"]
        crop = frames[3, y : y + height, x : x + width]
        self.assertGreater(int(crop.max()), int(crop.min()))
        marker = crop[2:5, 2:5]
        self.assertEqual(sorted(set(int(v) for v in marker.ravel())), [0, 255])

        clear_value = clone(value)
        clear_value["style"]["contrast_enhance"] = False
        clear_value["style"]["mask_opacity"] = 0.0
        clear_result = run_case(self.root, "clear-mask", clear_value)
        clear_frames = np.load(clear_result.frames_path)
        self.assertFalse(np.array_equal(frames, clear_frames))
        clear_bbox = clear_result.report["subtitles"][0]["bbox"]
        x, y, width, height = clear_bbox
        self.assertGreater(
            float(clear_frames[3, y : y + height, x : x + width].mean()),
            float(crop.mean()),
        )

    def test_text_only_edit_invalidates_only_visible_blocks(self) -> None:
        value = description(
            events=[event("e1", 0, 200), event("e2", 200, 400, "Delta")],
            edits=[{"op": "set_text", "event_id": "e1", "text": "Changed alpha"}],
        )
        path = self.root / "incremental.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        spec = load_spec(path)
        plan = build_incremental_plan(spec)
        self.assertEqual(plan.steps[0].affected_event_ids, ("e1",))
        self.assertEqual(plan.steps[0].dirty_blocks, (0, 1))
        self.assertEqual(plan.invalidated_blocks, (0, 1))

        time_edit = description(
            events=[event("e1", 0, 200), event("e2", 200, 400, "Delta")],
            edits=[{"op": "set_time", "event_id": "e2", "start_ms": 150, "end_ms": 250}],
        )
        time_path = self.root / "time-edit.json"
        time_path.write_text(json.dumps(time_edit), encoding="utf-8")
        time_plan = build_incremental_plan(load_spec(time_path))
        self.assertEqual(time_plan.steps[0].dirty_blocks, (1, 2, 3))

        style_edit = description(
            events=[event("e1", 0, 200), event("e2", 200, 400, "Delta")],
            edits=[{"op": "set_style", "field": "font_size", "value": 12}],
        )
        style_path = self.root / "style-edit.json"
        style_path.write_text(json.dumps(style_edit), encoding="utf-8")
        style_plan = build_incremental_plan(load_spec(style_path))
        self.assertEqual(set(style_plan.steps[0].affected_event_ids), {"e1", "e2"})
        self.assertEqual(style_plan.steps[0].dirty_blocks, (0, 1, 2, 3))


class StateAndExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_reuse_digest_repair_and_version_namespaces(self) -> None:
        value = description()
        first = run_case(self.root, "state", value)
        self.assertEqual(first.report["state"]["reused_block_count"], 0)
        self.assertEqual(first.report["state"]["recomputed_block_count"], 4)
        version_dir = next((self.root / "state" / "state").glob("*" )).parent
        version_dirs = [p for p in (self.root / "state" / "state").iterdir() if p.is_dir()]
        self.assertEqual(len(version_dirs), 1)
        old_version_dir = version_dirs[0]

        second = run_case(self.root, "state", value)
        self.assertEqual(second.report["state"]["reused_block_count"], 4)
        self.assertEqual(second.report["state"]["recomputed_block_count"], 0)

        block_path = old_version_dir / "blocks" / "000001.npz"
        corrupted = bytearray(block_path.read_bytes())
        corrupted[len(corrupted) // 2] ^= 0xFF
        block_path.write_bytes(corrupted)
        repaired = run_case(self.root, "state", value)
        self.assertIn("E_DIGEST", repaired.report["business_codes"])
        self.assertEqual(repaired.report["state"]["recomputed_block_count"], 1)
        self.assertEqual(repaired.report["state"]["reused_block_count"], 3)

        changed = clone(value)
        changed["events"][0]["text"] = "A different alpha"
        changed_result = run_case(self.root, "state", changed)
        self.assertNotEqual(changed_result.version, first.version)
        version_dirs = [p for p in (self.root / "state" / "state").iterdir() if p.is_dir()]
        self.assertEqual(len(version_dirs), 2)
        self.assertTrue(old_version_dir.exists())
        self.assertEqual(changed_result.report["state"]["reused_block_count"], 0)

    def test_manifest_frame_count_mismatch_triggers_digest_repair(self) -> None:
        value = description()
        run_case(self.root, "manifest", value)
        version_dir = next(
            p for p in (self.root / "manifest" / "state").iterdir() if p.is_dir()
        )
        manifest_path = version_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["blocks"]["2"]["frame_count"] = 999
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        repaired = run_case(self.root, "manifest", value)
        self.assertIn("E_DIGEST", repaired.report["business_codes"])
        self.assertGreaterEqual(repaired.report["state"]["recomputed_block_count"], 1)

    def test_export_commit_pairs_frames_and_report(self) -> None:
        value = description()
        result = run_case(self.root, "export", value)
        output = self.root / "export" / "output"
        commit = json.loads((output / "commit.json").read_text())
        report = json.loads((output / "report.json").read_text())
        frames = np.load(output / "frames.npy")
        self.assertEqual(commit["version"], result.version)
        self.assertEqual(commit["version"], report["version"])
        self.assertEqual(commit["frame_count"], report["frame_count"])
        self.assertEqual(commit["frame_count"], frames.shape[0])
        for name, digest_key in (("frames.npy", "frames_digest"), ("report.json", "report_digest")):
            self.assertEqual(
                hashlib.sha256((output / name).read_bytes()).hexdigest(), commit[digest_key]
            )
        self.assertEqual(report["blocks"], result.report["blocks"])
        self.assertEqual(len(report["dependency_graph"]["safe_area"]), 2)

    def test_memory_budget_serializes_and_rejects_impossible_blocks(self) -> None:
        value = description(concurrency=3)
        serial = run_case(self.root, "serial", value, concurrency=1)
        limited = run_case(
            self.root,
            "limited",
            value,
            concurrency=3,
            memory_budget_bytes=350_000,
        )
        left = np.load(serial.frames_path)
        right = np.load(limited.frames_path)
        self.assertTrue(np.array_equal(left, right))
        self.assertEqual(serial.version, limited.version)
        with self.assertRaises(EngineError) as raised:
            run_case(
                self.root,
                "too-small",
                value,
                concurrency=3,
                memory_budget_bytes=100_000,
            )
        self.assertEqual(raised.exception.code, "E_MEMORY_BUDGET")
        self.assertFalse((self.root / "too-small" / "output" / "frames.npy").exists())

    def test_block_error_is_reported_and_other_blocks_cancel(self) -> None:
        value = description(concurrency=2)

        def failing_compose(*args, **kwargs):
            frame_index = args[3]
            if frame_index >= 5:
                raise RuntimeError("synthetic render failure")
            from subtitle_engine.render import compose_frame as real_compose

            return real_compose(*args, **kwargs)

        with mock.patch("subtitle_engine.blocks.compose_frame", side_effect=failing_compose):
            with self.assertRaises(EngineError) as raised:
                run_case(self.root, "block-error", value)
        self.assertEqual(raised.exception.code, "E_INTERNAL")
        self.assertEqual(raised.exception.details["block_index"], 1)
        self.assertFalse((self.root / "block-error" / "output" / "frames.npy").exists())


if __name__ == "__main__":
    unittest.main()
