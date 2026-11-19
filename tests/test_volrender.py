from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import volrender.pipeline as pipeline_module
from volrender.errors import PipelineError
from volrender.geometry import BlockPlan, build_camera_rays, make_block_plans
from volrender.pipeline import render_task, run_task
from volrender.samples import generate_samples, verify_samples
from volrender.scheduler import execute_blocks
from volrender.spec import load_task
from volrender.volume import apply_edits
from volrender.spec import Edit


def _transfer(gray: float = 0.8, opacity: float = 0.12) -> dict[str, object]:
    return {
        "points": [
            {"scalar": 0.0, "opacity": 0.0, "gray": 0.0},
            {"scalar": 0.4, "opacity": opacity * 0.4, "gray": gray * 0.5},
            {"scalar": 1.0, "opacity": opacity, "gray": gray},
        ]
    }


def _payload(volume_path: Path, *, edits=None, transfer=None, camera=None, **overrides):
    payload = {
        "schema_version": 1,
        "volume": {
            "path": str(volume_path),
            "dims": [16, 16, 16],
            "spacing_mm": [1.0, 1.0, 1.0],
            "scalar_domain": {"min": 0.0, "max": 1.0},
        },
        "transfer_function": transfer or _transfer(),
        "camera": camera
        or {
            "position": [28.0, 24.0, 30.0],
            "target": [8.0, 8.0, 8.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 50.0,
        },
        "output": {"width": 24, "height": 24},
        "blocks": {"width": 8, "height": 8},
        "concurrency": 2,
        "memory_budget_bytes": 4 * 1024 * 1024,
        "sampling_step_mm": 0.8,
        "background_gray": 0.0,
        "edits": edits or [],
        "state_dir": "state",
        "outputs": {
            "gray": "out/gray.npy",
            "depth": "out/depth.npy",
            "report": "out/report.json",
        },
    }
    payload.update(overrides)
    return payload


def _write_task(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _sphere_volume() -> np.ndarray:
    z, y, x = np.indices((16, 16, 16), dtype=np.float64)
    distance = (x - 8.0) ** 2 + (y - 8.0) ** 2 + (z - 8.0) ** 2
    volume = np.clip(1.0 - distance / 48.0, 0.0, 1.0)
    volume[distance > 49.0] = 0.0
    return volume


class VolumeRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.volume_path = self.root / "volume.npy"
        np.save(self.volume_path, _sphere_volume(), allow_pickle=False)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_validation_errors_do_not_create_outputs_or_state(self) -> None:
        cases = [
            ("dims", {"volume": {**_payload(self.volume_path)["volume"], "dims": [0, 16, 16]}}, "E_DIMENSIONS"),
            ("spacing", {"volume": {**_payload(self.volume_path)["volume"], "spacing_mm": [1.0, 0.0, 1.0]}}, "E_SPACING"),
            (
                "transfer",
                {"transfer_function": {"points": [
                    {"scalar": 0.5, "opacity": 0.1, "gray": 0.4},
                    {"scalar": 0.4, "opacity": 0.2, "gray": 0.5},
                ]}},
                "E_TRANSFER",
            ),
            ("camera", {"camera": {
                "position": [8.0, 8.0, 20.0],
                "target": [8.0, 8.0, 8.0],
                "up": [0.0, 0.0, 1.0],
                "fov_y_deg": 50.0,
            }}, "E_CAMERA"),
            ("blocks", {"blocks": {"width": 7, "height": 8}}, "E_BLOCK"),
            ("concurrency", {"concurrency": 0}, "E_CONCURRENCY"),
            ("memory", {"memory_budget_bytes": 0}, "E_MEMORY"),
            ("edit", {"edits": [{"center": [15, 15, 15], "size": [4, 4, 4], "value": 0.5}]}, "E_EDIT"),
        ]
        for index, (name, override, expected_code) in enumerate(cases):
            with self.subTest(name=name):
                task_dir = self.root / f"invalid_{index}"
                payload = _payload(self.volume_path)
                payload.update(override)
                task_path = _write_task(task_dir / "task.json", payload)
                with self.assertRaises(PipelineError) as caught:
                    render_task(task_path)
                self.assertEqual(caught.exception.code, expected_code)
                self.assertFalse((task_dir / "state").exists())
                self.assertFalse((task_dir / "out").exists())

    def test_edits_are_ordered_deterministic_and_bounded(self) -> None:
        base = np.arange(16**3, dtype=np.float64).reshape(16, 16, 16) / 16**3
        edits = (
            Edit(center=(8, 8, 8), size=(7, 7, 7), value=0.2),
            Edit(center=(8, 8, 8), size=(3, 3, 3), value=0.9),
        )
        first = apply_edits(base, edits)
        second = apply_edits(base, edits)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(base, np.arange(16**3, dtype=np.float64).reshape(16, 16, 16) / 16**3))
        self.assertEqual(first.shape, base.shape)
        self.assertEqual(float(first[8, 8, 8]), 0.9)
        self.assertEqual(float(first[5, 8, 8]), 0.2)

    def test_transfer_change_reuses_only_all_zero_blocks(self) -> None:
        np.save(self.volume_path, np.zeros((16, 16, 16)), allow_pickle=False)
        task_path = _write_task(
            self.root / "task.json", _payload(self.volume_path, camera={
                "position": [30.0, 26.0, 34.0],
                "target": [8.0, 8.0, 8.0],
                "up": [0.0, 1.0, 0.0],
                "fov_y_deg": 65.0,
            })
        )
        first = render_task(task_path)
        self.assertTrue(np.all(first.gray == 0.0))
        self.assertTrue(np.all(first.depth == first.report["render"]["background_depth_mm"]))

        changed_payload = _payload(self.volume_path, transfer=_transfer(gray=0.1, opacity=0.3))
        changed_payload["camera"] = {
            "position": [30.0, 26.0, 34.0],
            "target": [8.0, 8.0, 8.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 65.0,
        }
        _write_task(task_path, changed_payload)
        second = render_task(task_path)
        self.assertEqual(second.report["changes"]["transfer_function_changed"], True)
        self.assertEqual(second.report["summary"]["rerendered_blocks"], 0)
        self.assertEqual(second.report["summary"]["reused_blocks"], 9)
        self.assertTrue(all(block["invalid_categories"] == [] for block in second.report["blocks"]))

    def test_transfer_extra_fields_participate_in_change_detection(self) -> None:
        first_transfer = _transfer()
        for point in first_transfer["points"]:
            point["bandwidth"] = 0.25
        task_path = _write_task(
            self.root / "task.json",
            _payload(self.volume_path, transfer=first_transfer),
        )
        render_task(task_path)

        second_transfer = _transfer()
        for point in second_transfer["points"]:
            point["bandwidth"] = 0.5
        _write_task(
            task_path,
            _payload(self.volume_path, transfer=second_transfer),
        )
        second = render_task(task_path)
        self.assertTrue(second.report["changes"]["transfer_function_changed"])
        self.assertGreater(second.report["summary"]["rerendered_blocks"], 0)
        self.assertIn("transfer_function", {
            category
            for block in second.report["blocks"]
            for category in block["invalid_categories"]
        })

    def test_serial_and_parallel_renders_are_bitwise_equal(self) -> None:
        serial_payload = _payload(self.volume_path, concurrency=1)
        serial_task = _write_task(self.root / "serial" / "task.json", serial_payload)
        spec_for_budget = load_task(serial_task)
        rays = build_camera_rays(
            spec_for_budget.camera,
            spec_for_budget.width,
            spec_for_budget.height,
            spec_for_budget.volume,
        )
        plans = make_block_plans(spec_for_budget, rays)
        budget = max(plan.sample_buffer_bytes for plan in plans) * 2
        parallel_payload = _payload(
            self.volume_path, concurrency=4, memory_budget_bytes=budget
        )
        parallel_task = _write_task(self.root / "parallel" / "task.json", parallel_payload)
        serial = render_task(serial_task)
        parallel = render_task(parallel_task)
        self.assertTrue(np.array_equal(serial.gray, parallel.gray))
        self.assertTrue(np.array_equal(serial.depth, parallel.depth))
        self.assertEqual(serial.gray.shape, serial.depth.shape)
        self.assertLessEqual(
            parallel.report["render"]["peak_concurrent_sample_bytes"],
            parallel.report["render"]["memory_budget_bytes"],
        )

    def test_scheduler_cancels_remaining_blocks_after_first_error(self) -> None:
        plans = [
            BlockPlan(
                index=index,
                row=0,
                col=index,
                x0=index,
                y0=0,
                x1=index + 1,
                y1=1,
                pixel_count=1,
                voxel_bounds_min=None,
                voxel_bounds_max=None,
                ray_hit_count=0,
                max_steps=1,
                sample_buffer_bytes=64,
            )
            for index in range(5)
        ]

        def worker(plan: BlockPlan) -> int:
            if plan.index == 1:
                raise RuntimeError("injected")
            return plan.index

        execution = execute_blocks(
            plans,
            worker,
            concurrency=1,
            memory_budget_bytes=256,
            interrupt_event=threading.Event(),
        )
        self.assertEqual(sorted(execution.results), [0])
        self.assertEqual(execution.first_error()[0], 1)
        self.assertEqual(execution.not_started_indices, [2, 3, 4])

    def test_cache_archive_and_incremental_full_equivalence(self) -> None:
        task_path = _write_task(self.root / "task.json", _payload(self.volume_path))
        initial = render_task(task_path)
        repeated = render_task(task_path)
        self.assertEqual(repeated.report["summary"]["rerendered_blocks"], 0)
        self.assertEqual(repeated.report["summary"]["reused_blocks"], 9)

        edited_payload = _payload(
            self.volume_path,
            edits=[{"center": [8, 8, 8], "size": [5, 5, 5], "value": 0.95}],
        )
        _write_task(task_path, edited_payload)
        incremental = render_task(task_path)
        self.assertGreater(incremental.report["summary"]["reused_blocks"], 0)
        self.assertLess(incremental.report["summary"]["rerendered_blocks"], 9)
        self.assertTrue(any((self.root / "state" / "archive").rglob("*.npz")))

        full = render_task(
            task_path,
            state_dir_override=self.root / "full_state",
            output_dir_override=self.root / "full_out",
            force_full=True,
        )
        self.assertTrue(np.array_equal(incremental.gray, full.gray))
        self.assertTrue(np.array_equal(incremental.depth, full.depth))

        reused_again = render_task(task_path)
        self.assertEqual(reused_again.report["summary"]["rerendered_blocks"], 0)
        self.assertTrue(np.array_equal(reused_again.gray, full.gray))
        self.assertTrue(np.array_equal(reused_again.depth, full.depth))

    def test_digest_mismatch_is_rerendered_and_reported(self) -> None:
        task_path = _write_task(self.root / "task.json", _payload(self.volume_path))
        first = render_task(task_path)
        block_file = self.root / "state" / "blocks" / "block_r0000_c0000.npz"
        with np.load(block_file, allow_pickle=False) as data:
            gray = np.asarray(data["gray"]).copy()
            depth = np.asarray(data["depth"]).copy()
            meta = np.asarray(data["meta"]).copy()
        gray.flat[0] = 1.0 - gray.flat[0]
        with block_file.open("wb") as handle:
            np.savez(handle, gray=gray, depth=depth, meta=meta)

        recovered = render_task(task_path)
        self.assertIn("E_DIGEST", recovered.report["summary"]["business_code_distribution"])
        self.assertEqual(recovered.report["blocks"][0]["cache"]["digest_retry"], True)
        self.assertTrue(np.array_equal(recovered.gray, first.gray))
        self.assertTrue(np.array_equal(recovered.depth, first.depth))

    def test_interrupt_only_at_block_boundary_and_restart_recovers(self) -> None:
        concurrency_one = _payload(self.volume_path, concurrency=1)
        task_path = _write_task(self.root / "task.json", concurrency_one)
        task = load_task(task_path)
        interrupt = threading.Event()
        original = pipeline_module.render_block
        calls = {"count": 0}

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            calls["count"] += 1
            if calls["count"] == 3:
                interrupt.set()
            return result

        with mock.patch.object(pipeline_module, "render_block", side_effect=wrapped):
            with self.assertRaises(PipelineError) as caught:
                run_task(task, interrupt_event=interrupt)
        self.assertEqual(caught.exception.code, "E_INTERRUPTED")
        self.assertTrue((self.root / "state" / "INTERRUPTED.json").exists())
        self.assertFalse((self.root / "out" / "gray.npy").exists())
        self.assertFalse((self.root / "out" / "report.json").exists())

        restarted = run_task(task)
        self.assertEqual(restarted.report["summary"]["persisted_recovery_blocks"], 3)
        self.assertEqual(restarted.report["summary"]["missing_before_render"], 6)
        self.assertEqual(restarted.report["summary"]["rerendered_blocks"], 6)
        self.assertFalse((self.root / "state" / "INTERRUPTED.json").exists())

        full = run_task(
            load_task(
                task_path,
                state_dir_override=self.root / "full_state",
                output_dir_override=self.root / "full_out",
            ),
            force_full=True,
        )
        self.assertTrue(np.array_equal(restarted.gray, full.gray))
        self.assertTrue(np.array_equal(restarted.depth, full.depth))

    def test_sample_suite_incremental_results_match_full_renders(self) -> None:
        samples = self.root / "samples"
        generate_samples(samples)
        result = verify_samples(samples)
        self.assertTrue(result["ok"], result)
        for item in result["tasks"]:
            self.assertTrue(item["incremental_vs_full_gray_equal"])
            self.assertTrue(item["incremental_vs_full_depth_equal"])
            self.assertTrue(item["strict_shapes_equal"])


if __name__ == "__main__":
    unittest.main()
