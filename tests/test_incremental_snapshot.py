from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from panorama_workbench.commands import apply_commands
from panorama_workbench.config import load_config
from panorama_workbench.engine import Workbench
from panorama_workbench.samples import generate_demo
from panorama_workbench.validation import discover_image_specs


class IncrementalSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.demo = generate_demo(cls.root / "demo", image_count=3)
        cls.config = load_config(cls.demo.config)
        cls.commands = json.loads(cls.demo.commands.read_text(encoding="utf-8"))["commands"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_iou_edit_reuses_every_tile(self) -> None:
        state_dir = self.root / "iou-state"
        Workbench(state_dir).run(self.demo.samples, self.config)
        commands = [
            {"op": "set_merge_iou", "value": 0.2},
            {"op": "set_min_instance_area", "value": 31},
        ]
        incremental = Workbench(state_dir).run(
            self.demo.samples, self.config, commands=commands
        )
        self.assertEqual(incremental.report["cache"]["tile_misses"], 0)
        self.assertGreater(incremental.report["cache"]["tile_hits"], 0)
        for image in incremental.report["images"].values():
            self.assertEqual(image["cache"]["tile_misses"], 0)
            self.assertEqual(image["cache"]["mask_recomputes"], 0)

        final_samples = self.root / "iou-final"
        _materialize(
            final_samples,
            discover_image_specs(self.demo.samples),
            self.config,
            [],
            self.root,
        )
        final_config = apply_commands(
            discover_image_specs(self.demo.samples), self.config, commands
        ).config
        full = Workbench(self.root / "iou-full-state").run(final_samples, final_config)
        self.assertEqual(_core(incremental.report), _core(full.report))
        self.assertEqual(incremental.report["failure_list"], full.report["failure_list"])

    def test_incremental_edit_sequence_matches_full_rebuild(self) -> None:
        state_dir = self.root / "incremental-state"
        first = Workbench(state_dir).run(self.demo.samples, self.config)
        self.assertFalse(first.pending)
        incremental = Workbench(state_dir).run(
            self.demo.samples,
            self.config,
            commands=self.commands,
            command_base=self.root / "demo",
        )
        applied = apply_commands(
            discover_image_specs(self.demo.samples),
            self.config,
            self.commands,
            command_base=self.root / "demo",
        )
        self.assertFalse(applied.failures)
        final_samples = self.root / "final-samples"
        _materialize(final_samples, applied.specs, applied.config, [], self.root)
        full = Workbench(self.root / "full-state").run(final_samples, applied.config)
        self.assertEqual(_core(incremental.report), _core(full.report))
        self.assertEqual(incremental.report["failure_list"], full.report["failure_list"])
        self.assertEqual(
            incremental.report["config"]["hash"], full.report["config"]["hash"]
        )
        self.assertEqual(incremental.report["pending_queue"], [])

    def test_graceful_stop_resume_and_corrupt_state_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            one = root / "samples"
            one.mkdir()
            for index in (1, 2, 3):
                shutil.copy2(
                    self.demo.samples / f"scene_{index:02d}.png",
                    one / f"scene_{index:02d}.png",
                )
                shutil.copy2(
                    self.demo.samples / f"scene_{index:02d}.json",
                    one / f"scene_{index:02d}.json",
                )
            config = load_config(self.demo.config).with_updates(max_workers=1)
            state_dir = root / "state"
            stopped = Workbench(state_dir).run(one, config, stop_after_images=1)
            self.assertEqual(stopped.exit_code, 130)
            self.assertTrue(stopped.stopped)
            self.assertTrue(stopped.pending)
            manifest = json.loads((state_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["payload"]["clean"])
            resumed = Workbench(state_dir).run(one, config)
            self.assertEqual(resumed.recovery_code, "R_RESUME_CHECKPOINT")
            self.assertEqual(resumed.exit_code, 0)
            self.assertFalse(resumed.pending)
            self.assertEqual(len(resumed.report["state"]["completed_images"]), 3)

            state_path = state_dir / "state.json"
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            payload["payload"]["config_hash"] = "corrupt"
            state_path.write_text(json.dumps(payload), encoding="utf-8")
            rebuilt = Workbench(state_dir).run(one, config)
            self.assertEqual(rebuilt.recovery_code, "R_FULL_REBUILD")
            self.assertEqual(rebuilt.exit_code, 0)
            self.assertEqual(len(rebuilt.report["state"]["completed_images"]), 3)

    def test_replacement_invalidates_only_changed_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = root / "samples"
            samples.mkdir()
            shutil.copy2(self.demo.samples / "scene_01.png", samples / "scene_01.png")
            shutil.copy2(self.demo.samples / "scene_01.json", samples / "scene_01.json")
            state_dir = root / "state"
            Workbench(state_dir).run(samples, self.config)
            original_size = (samples / "scene_01.png").stat().st_size
            rgb = Image.open(samples / "scene_01.png").convert("RGB")
            array = np.array(rgb)
            array[0:2, 0:2, :] = 255 - array[0:2, 0:2, :]
            Image.fromarray(array, mode="RGB").save(samples / "scene_01.png")
            changed = Workbench(state_dir).run(samples, self.config)
            image_report = changed.report["images"]["scene_01.png"]
            self.assertGreater(image_report["cache"]["tile_hits"], 0)
            self.assertGreater(image_report["cache"]["tile_misses"], 0)
            self.assertGreater(original_size, 0)

    def test_category_noop_reuses_masks_and_stage_versions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = root / "samples"
            samples.mkdir()
            shutil.copy2(self.demo.samples / "scene_01.png", samples / "scene_01.png")
            shutil.copy2(self.demo.samples / "scene_01.json", samples / "scene_01.json")
            state_dir = root / "state"
            first = Workbench(state_dir).run(samples, self.config)
            manifest_path = state_dir / "manifest.json"
            first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["payload"]
            stable = Workbench(state_dir).run(samples, self.config)
            stable_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["payload"]
            self.assertEqual(stable.recovery_code, "R_STABLE_RERUN")
            self.assertEqual(
                first_manifest["stage_versions"], stable_manifest["stage_versions"]
            )
            self.assertEqual(stable.report["cache"]["tile_hits"], 49)

            category_edit = [
                {
                    "op": "add_category",
                    "category": {"code": 9, "name": "noop", "weight": 0.001},
                }
            ]
            edited = Workbench(state_dir).run(
                samples, self.config, commands=category_edit
            )
            image = edited.report["images"]["scene_01.png"]
            self.assertEqual(image["cache"]["tile_hits"], 49)
            self.assertEqual(image["cache"]["mask_hits"], 49)
            self.assertEqual(image["cache"]["mask_recomputes"], 0)
            self.assertEqual(image["invalidated_instance_ids"], [])
            self.assertEqual(first.report["images"]["scene_01.png"]["instances"], image["instances"])

    def test_cache_budget_evicts_deterministically_and_remains_usable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            samples = root / "samples"
            samples.mkdir()
            shutil.copy2(self.demo.samples / "scene_01.png", samples / "scene_01.png")
            shutil.copy2(self.demo.samples / "scene_01.json", samples / "scene_01.json")
            state_dir = root / "state"
            config = replace(self.config, cache_budget=10, max_workers=1)
            first = Workbench(state_dir).run(samples, config)
            self.assertGreater(first.report["cache"]["evicted_tile_count"], 0)
            state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
            self.assertLessEqual(len(state["payload"]["tile_cache"]), 10)
            stable = Workbench(state_dir).run(samples, config)
            self.assertEqual(stable.exit_code, 0)
            merged = Workbench(state_dir).run(
                samples,
                config,
                commands=[{"op": "set_merge_iou", "value": 0.2}],
            )
            self.assertEqual(merged.exit_code, 0)
            self.assertEqual(
                first.report["images"]["scene_01.png"]["instances"],
                stable.report["images"]["scene_01.png"]["instances"],
            )


def _materialize(target: Path, specs, config, commands, base: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        destination = target / spec.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.image_path, destination)
        truth = destination.with_suffix(".json")
        shutil.copy2(spec.truth_path, truth)


def _core(report: dict) -> dict:
    result = {}
    for name, image in report["images"].items():
        result[name] = {
            "dropped_instances": image["dropped_instances"],
            "evaluation": image["evaluation"],
            "instance_raster": image["instance_raster"],
            "instances": image["instances"],
            "semantic_raster": image["semantic_raster"],
            "truncated_instances": image["truncated_instances"],
        }
    return result


if __name__ == "__main__":
    unittest.main()
