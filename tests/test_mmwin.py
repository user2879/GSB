from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from mmwin.canonical import atomic_write_json
from mmwin.config import parse_config
from mmwin.engine import canonical_business_projection, run_engine
from mmwin.events import load_events
from mmwin.errors import ConfigError
from mmwin.generate import generate_samples
from mmwin.plan import load_plan


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _tiny_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "window_length_ms": 1000,
        "window_step_ms": 500,
        "allowed_lateness_ms": 100,
        "watermark_rule": "max_seen_minus_lateness",
        "late_policy": "resettle",
        "visual_feature_dim": 4,
        "text_feature_dim": 8,
        "visual_block_rows": 2,
        "visual_block_cols": 2,
        "visual_frames_per_window": 2,
        "quantization_level": "Q8",
        "fractional_bits": 8,
        "quantization_levels": {"Q8": 8, "Q12": 12},
        "fusion_temperature": 1.0,
        "thread_pool_size": 2,
        "concurrency_limit": 4,
        "max_events_per_window": 16,
        "checkpoint_commit_interval_windows": 1,
        "batch_event_budget": 2,
        "sensor_channels": {"temperature": {"min": -20.0, "max": 120.0}},
    }
    config.update(overrides)
    return config


def _text(event_id: str, timestamp: int, value: str = "ok", *, attribute: str = "state") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "timestamp_ms": timestamp,
        "entity": {"id": "e-1", "type": "entity"},
        "attribute": attribute,
        "value": value,
        "confidence": 0.9,
        "polarity": 0.5,
    }


def _tiny_sample(
    root: Path,
    text_events: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    plan_ids: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    samples = root / "samples"
    samples.mkdir(parents=True)
    _jsonl(samples / "text_assertions.jsonl", text_events)
    (samples / "sensor_readings.jsonl").write_text("", encoding="utf-8")
    config = config or _tiny_config()
    atomic_write_json(samples / "window_config.json", config)
    ids = plan_ids or [event["event_id"] for event in text_events]
    atomic_write_json(
        samples / "injection_plan.json",
        {
            "arrivals": [
                {"arrival_seq": index, "event_id": event_id, "delay_ms": 0}
                for index, event_id in enumerate(ids)
            ]
        },
    )
    return samples, samples / "window_config.json", samples / "injection_plan.json"


class MMWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="mmwin-tests-")
        cls.base = Path(cls.temp.name) / "base"
        cls.sample_summary = generate_samples(cls.base, seed=20261124)
        # One reusable clean run keeps the independent tests fast.
        cls.base_checkpoint = Path(cls.temp.name) / "base-checkpoint"
        cls.base_report = run_engine(
            cls.base,
            cls.base / "window_config.json",
            cls.base / "injection_plan.json",
            cls.base_checkpoint,
            reset=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_generated_sample_and_validation(self) -> None:
        self.assertEqual(self.sample_summary["counts"]["image"], 16)
        self.assertEqual(self.sample_summary["counts"]["text"], 34)
        self.assertEqual(self.sample_summary["counts"]["sensor"], 40)
        config = parse_config(json.loads((self.base / "window_config.json").read_text()))
        loaded = load_events(self.base, config)
        self.assertEqual(loaded.counts, {"image": 16, "text": 34, "sensor": 40})
        self.assertEqual(loaded.failures, [])

        with tempfile.TemporaryDirectory(prefix="mmwin-invalid-") as temp:
            root = Path(temp)
            _jsonl(
                root / "text_assertions.jsonl",
                [
                    _text("dup", 1),
                    _text("dup", 2),
                    _text("bad-ts", -1),
                    {
                        "event_id": "bad-entity",
                        "timestamp_ms": 3,
                        "attribute": "state",
                        "value": "ok",
                    },
                    {
                        "event_id": "nan-value",
                        "timestamp_ms": 4,
                        "entity": {"id": "e", "type": "t"},
                        "attribute": "score",
                        "value": float("nan"),
                    },
                ],
            )
            _jsonl(
                root / "sensor_readings.jsonl",
                [
                    {"event_id": "bad-channel", "timestamp_ms": 5, "channel": "unknown", "value": 1.0},
                    {
                        "event_id": "nan-reading",
                        "timestamp_ms": 6,
                        "channel": "temperature",
                        "value": float("nan"),
                    },
                ],
            )
            invalid = load_events(root, config)
            self.assertEqual(set(invalid.events), {"dup"})
            self.assertEqual(len(invalid.failures), 6)
            self.assertTrue(all(failure.code == "E_EVENT_INVALID" for failure in invalid.failures))

        raw = json.loads((self.base / "window_config.json").read_text())
        for mutation in [
            {"window_length_ms": 1000, "window_step_ms": 300},
            {"allowed_lateness_ms": -1},
            {"thread_pool_size": 9, "concurrency_limit": 8},
            {"max_events_per_window": 0},
            {"quantization_level": "Q12", "fractional_bits": 8},
        ]:
            with self.subTest(mutation=mutation):
                candidate = json.loads(json.dumps(raw))
                candidate.update(mutation)
                with self.assertRaises(ConfigError):
                    parse_config(candidate)

    def test_replay_order_watermark_boundary_and_late_rule(self) -> None:
        plan = load_plan(self.base / "injection_plan.json")
        report = self.base_report
        self.assertEqual(
            [entry["event_id"] for entry in report["delivery_order"]],
            [arrival.event_id for arrival in plan.arrivals],
        )
        self.assertEqual(
            report["watermark"]["final_watermark_ms"],
            report["watermark"]["max_seen_timestamp_ms"]
            - report["watermark"]["allowed_lateness_ms"],
        )
        self.assertGreater(len(report["resettlements"]), 0)
        self.assertTrue(all(item["policy"] == "resettle" for item in report["resettlements"]))

        with tempfile.TemporaryDirectory(prefix="mmwin-boundary-") as temp:
            samples, config, plan_path = _tiny_sample(
                Path(temp),
                [
                    _text("e-1000", 1000),
                    _text("e-900", 900),
                    _text("e-1500", 1500),
                    _text("e-50", 50),
                ],
            )
            checkpoint = Path(temp) / "checkpoint"
            boundary = run_engine(samples, config, plan_path, checkpoint, reset=True)
            self.assertEqual(
                [item["event_id"] for item in boundary["delivery_order"]],
                ["e-1000", "e-900", "e-1500", "e-50"],
            )
            self.assertEqual(boundary["watermark"]["final_watermark_ms"], 1400)
            late_ids = [item["late_event_id"] for item in boundary["resettlements"]]
            self.assertNotIn("e-900", late_ids)  # lateness exactly 100 == allowed
            self.assertIn("e-50", late_ids)
            late = next(item for item in boundary["resettlements"] if item["late_event_id"] == "e-50")
            self.assertEqual(late["lateness_ms"], 1450)
            self.assertIsNotNone(late["old_conclusion"])
            self.assertIsNotNone(late["new_conclusion"])

    def test_empty_visual_quantization_and_overflow(self) -> None:
        q8 = self.base_report
        with tempfile.TemporaryDirectory(prefix="mmwin-q12-") as temp:
            q12 = run_engine(
                self.base,
                self.base / "window_config_q12.json",
                self.base / "injection_plan.json",
                Path(temp) / "checkpoint",
                reset=True,
            )
        self.assertEqual(q8["empty_windows"], q12["empty_windows"])
        self.assertTrue(q8["empty_windows"])
        for window_id in q8["empty_windows"]:
            window = next(item for item in q8["windows"] if item["window_id"] == window_id)
            self.assertTrue(window["empty_visual"])
            self.assertEqual(window["visual"]["vector_q"], [0] * 12)
        changed = 0
        for left, right in zip(q8["windows"], q12["windows"]):
            if left.get("visual") and right.get("visual"):
                self.assertEqual(
                    left["visual"]["selected_frame_indices"],
                    sorted(left["visual"]["selected_frame_indices"]),
                )
                if left["visual"]["vector_q"] != right["visual"]["vector_q"]:
                    changed += 1
        self.assertGreater(changed, 0)

        with tempfile.TemporaryDirectory(prefix="mmwin-drop-") as temp:
            dropped = run_engine(
                self.base,
                self.base / "window_config_drop.json",
                self.base / "injection_plan.json",
                Path(temp) / "checkpoint",
                reset=True,
            )
        self.assertGreater(len(dropped["resettlements"]), 0)
        for record in dropped["resettlements"]:
            self.assertEqual(record["policy"], "drop")
            self.assertFalse(record["applied"])
            self.assertEqual(record["old_snapshot_id"], record["new_snapshot_id"])
            self.assertEqual(record["old_conclusion"], record["new_conclusion"])

        with tempfile.TemporaryDirectory(prefix="mmwin-overflow-") as temp:
            root = Path(temp)
            samples, config, plan = _tiny_sample(
                root,
                [_text("a", 10), _text("b", 20), _text("c", 30)],
                config=_tiny_config(max_events_per_window=2),
                plan_ids=["a", "b", "c"],
            )
            report = run_engine(samples, config, plan, root / "checkpoint", reset=True)
            self.assertEqual(report["windows"][0]["status"], "overflow")
            self.assertIsNone(report["windows"][0]["confidence_q"])
            self.assertIn("E_WINDOW_OVERFLOW", {failure["code"] for failure in report["failures"]})

    def test_budget_pending_resume_and_checkpoint_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmwin-budget-") as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            first = run_engine(
                self.base,
                self.base / "window_config.json",
                self.base / "injection_plan.json",
                checkpoint,
                reset=True,
                max_batches=1,
            )
            self.assertEqual(first["meta"]["processed_arrival_count"], 7)
            self.assertEqual(len(first["pending"]), 83)
            self.assertEqual(
                {failure["code"] for failure in first["failures"]},
                {"E_EVENT_BUDGET_PENDING"},
            )
            manifest = json.loads((checkpoint / "MANIFEST.json").read_text())
            self.assertEqual(manifest["processed_arrival_count"], 7)
            self.assertEqual(len(manifest["pending_arrival_seqs"]), 83)
            self.assertEqual(
                set(manifest["stages"]),
                {"replay", "membership", "visual", "fusion", "resettle", "pending", "runtime", "cache"},
            )
            for stage in manifest["stages"].values():
                self.assertEqual(stage["version"], manifest["global_version"])
            self.assertFalse(any(path.name.startswith(".staging-") for path in checkpoint.iterdir()))

            resumed = run_engine(
                self.base,
                self.base / "window_config.json",
                self.base / "injection_plan.json",
                checkpoint,
            )
            self.assertTrue(resumed["checkpoint"]["resumed"])
            self.assertEqual(resumed["pending"], [])
            self.assertEqual(
                canonical_business_projection(resumed),
                canonical_business_projection(self.base_report),
            )
            rerun = run_engine(
                self.base,
                self.base / "window_config.json",
                self.base / "injection_plan.json",
                checkpoint,
            )
            self.assertTrue(rerun["checkpoint"]["resumed"])
            self.assertEqual(rerun["cache"]["stats"]["visual_computations"], 0)
            self.assertEqual(rerun["cache"]["stats"]["fusion_computations"], 0)
            self.assertEqual(
                canonical_business_projection(rerun),
                canonical_business_projection(self.base_report),
            )

    def test_incremental_edit_sequence_matches_blank_full_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmwin-edits-") as temp:
            root = Path(temp)
            samples = root / "samples"
            shutil.copytree(self.base, samples)
            checkpoint = root / "incremental"
            full_root = root / "full"
            full_root.mkdir()

            def incremental() -> dict[str, Any]:
                return run_engine(
                    samples,
                    samples / "window_config.json",
                    samples / "injection_plan.json",
                    checkpoint,
                )

            def assert_equal(label: str, report: dict[str, Any]) -> None:
                full = run_engine(
                    samples,
                    samples / "window_config.json",
                    samples / "injection_plan.json",
                    full_root / label,
                    reset=True,
                )
                self.assertEqual(
                    canonical_business_projection(report),
                    canonical_business_projection(full),
                    label,
                )

            first = run_engine(
                samples,
                samples / "window_config.json",
                samples / "injection_plan.json",
                checkpoint,
                reset=True,
            )
            self.assertEqual(len(first["windows"]), len(self.base_report["windows"]))

            plan = json.loads((samples / "injection_plan.json").read_text())
            ids = [row["event_id"] for row in plan["arrivals"]][::-1]
            plan["arrivals"] = [
                {"arrival_seq": index, "event_id": event_id, "delay_ms": (index * 37) % 1400}
                for index, event_id in enumerate(ids)
            ]
            atomic_write_json(samples / "injection_plan.json", plan)
            report = incremental()
            self.assertFalse(report["checkpoint"]["resumed"])
            assert_equal("plan", report)

            text_rows = [
                json.loads(line)
                for line in (samples / "text_assertions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            text_rows[5]["value"] = "edited-value"
            text_rows[5]["polarity"] = -0.75
            _jsonl(samples / "text_assertions.jsonl", text_rows)
            report = incremental()
            assert_equal("replace", report)

            text_rows.pop(7)
            _jsonl(samples / "text_assertions.jsonl", text_rows)
            report = incremental()
            assert_equal("delete", report)
            self.assertIn(
                "E_EVENT_INVALID",
                {failure["code"] for failure in report["failures"]},
            )

            config = json.loads((samples / "window_config.json").read_text())
            config["window_length_ms"] = 1600
            config["window_step_ms"] = 400
            atomic_write_json(samples / "window_config.json", config)
            report = incremental()
            assert_equal("windows", report)

            config["allowed_lateness_ms"] = 900
            atomic_write_json(samples / "window_config.json", config)
            report = incremental()
            assert_equal("lateness", report)

            config["quantization_level"] = "Q12"
            config["fractional_bits"] = 12
            atomic_write_json(samples / "window_config.json", config)
            report = incremental()
            assert_equal("quantization", report)
            self.assertEqual(report["meta"]["quantization_level"], "Q12")

    def test_exact_cache_invalidation_and_thread_consistency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmwin-cache-") as temp:
            root = Path(temp)
            samples = root / "samples"
            shutil.copytree(self.base, samples)
            checkpoint = root / "checkpoint"
            before = run_engine(
                samples,
                samples / "window_config.json",
                samples / "injection_plan.json",
                checkpoint,
                reset=True,
            )
            text_path = samples / "text_assertions.jsonl"
            rows = [json.loads(line) for line in text_path.read_text().splitlines() if line.strip()]
            target = rows[10]["event_id"]
            rows[10]["value"] = "cache-invalidation-value"
            rows[10]["polarity"] = 0.95
            _jsonl(text_path, rows)
            after = run_engine(
                samples,
                samples / "window_config.json",
                samples / "injection_plan.json",
                checkpoint,
            )
            refs = {
                window["window_id"]: {ref["event_id"] for ref in window["event_refs"]}
                for window in before["windows"]
            }
            for manifest_entry in after["cache"]["window_manifest"]:
                if target in refs.get(manifest_entry["window_id"], set()):
                    continue
                status = manifest_entry["cache_status"]
                self.assertEqual(status["membership"], "reused")
                self.assertEqual(status["visual"], "reused")
                self.assertEqual(status["fusion"], "reused")
                self.assertEqual(status["resettle"], "reused")
            self.assertEqual(after["cache"]["stats"]["visual_computations"], 0)
            self.assertGreater(after["cache"]["stats"]["fusion_computations"], 0)

            # Deleting a text event must not shift semantic resettlement keys for
            # windows that do not reference it.
            deleted_id = rows[5]["event_id"]
            rows.pop(5)
            _jsonl(text_path, rows)
            deleted = run_engine(
                samples,
                samples / "window_config.json",
                samples / "injection_plan.json",
                checkpoint,
            )
            original_refs = {
                window["window_id"]: {ref["event_id"] for ref in window["event_refs"]}
                for window in before["windows"]
            }
            for manifest_entry in deleted["cache"]["window_manifest"]:
                if deleted_id in original_refs.get(manifest_entry["window_id"], set()):
                    continue
                status = manifest_entry["cache_status"]
                self.assertEqual(
                    status,
                    {"membership": "reused", "visual": "reused", "fusion": "reused", "resettle": "reused"},
                )

            serial_config = json.loads((samples / "window_config.json").read_text())
            serial_config["thread_pool_size"] = 1
            parallel_config = json.loads((samples / "window_config.json").read_text())
            parallel_config["thread_pool_size"] = 6
            atomic_write_json(root / "serial.json", serial_config)
            atomic_write_json(root / "parallel.json", parallel_config)
            serial = run_engine(
                samples, root / "serial.json", samples / "injection_plan.json", root / "serial", reset=True
            )
            parallel = run_engine(
                samples,
                root / "parallel.json",
                samples / "injection_plan.json",
                root / "parallel",
                reset=True,
            )
            for field in [
                "windows",
                "resettlements",
                "empty_windows",
                "failures",
                "delivery_order",
                "watermark",
                "pending",
                "invalidation_manifest",
            ]:
                self.assertEqual(serial[field], parallel[field], field)
            keys = [entry["stable_sort_key"] for entry in parallel["budget"]["scheduler_task_order"]]
            self.assertEqual(keys, sorted(keys))

    def test_checkpoint_mismatch_and_duplicate_arrival(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmwin-checkpoint-") as temp:
            root = Path(temp)
            samples = root / "samples"
            shutil.copytree(self.base, samples)
            checkpoint = root / "checkpoint"
            run_engine(
                samples,
                samples / "window_config.json",
                samples / "injection_plan.json",
                checkpoint,
                reset=True,
            )
            config = json.loads((samples / "window_config.json").read_text())
            config["allowed_lateness_ms"] = 777
            atomic_write_json(samples / "window_config.json", config)
            changed = run_engine(
                samples,
                samples / "window_config.json",
                samples / "injection_plan.json",
                checkpoint,
            )
            self.assertIn(
                "E_CHECKPOINT_VERSION_MISMATCH",
                {item["code"] for item in changed["diagnostics"]},
            )
            self.assertFalse(changed["checkpoint"]["resumed"])
            full = run_engine(
                samples,
                samples / "window_config.json",
                samples / "injection_plan.json",
                root / "full",
                reset=True,
            )
            self.assertEqual(
                canonical_business_projection(changed),
                canonical_business_projection(full),
            )

            duplicate = run_engine(
                self.base,
                self.base / "window_config.json",
                self.base / "injection_plan_duplicate.json",
                root / "duplicate",
                reset=True,
            )
            self.assertEqual(
                duplicate["delivery_order"][-1]["status"],
                "duplicate",
            )
            self.assertIn("E_EVENT_INVALID", {item["code"] for item in duplicate["failures"]})
            plan = load_plan(self.base / "injection_plan_duplicate.json")
            self.assertEqual(
                [entry["event_id"] for entry in duplicate["delivery_order"]],
                [arrival.event_id for arrival in plan.arrivals],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
