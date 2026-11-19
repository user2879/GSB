import dataclasses
import json
import unittest
from pathlib import Path

from tests._helpers import load_test_config, make_workspace
from workbench.config import STAGE_NAMES
from workbench.errors import E_SAMPLE_INVALID
from workbench.pipeline import run_pipeline
from workbench.report import _comparable_blob
from workbench.sample_gen import sample_paths
from workbench.utils import atomic_write_json, digest_obj, png_bytes


def pixel_map(report):
    return {
        sample["sample"]: sample["final_pixel_hash"]
        for sample in report["samples"]
        if sample["status"] == "ok"
    }


def stage_summary_map(report):
    return {
        sample["sample"]: sample.get("stage_summaries")
        for sample in report["samples"]
    }


class IncrementalRebuildTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)
        self.config = load_test_config(self.workspace)

    def _clone_with_snapshot(self, snapshot_suffix: str):
        root = Path(self.workspace.name)
        snap = str(root / snapshot_suffix)
        raw = {**self.config.raw, "snapshots_dir": snap}
        return dataclasses.replace(self.config, snapshots_dir=snap, raw=raw)

    def test_full_edit_sequence_matches_blank_rebuild(self):
        incremental = self._clone_with_snapshot("snap-inc")
        # 1. initial small run
        run_pipeline(incremental, run_mode="parallel")
        # 2. widen to medium then large
        medium = dataclasses.replace(
            incremental, active_tier="medium", raw={**incremental.raw, "active_tier": "medium"}
        )
        run_pipeline(medium)
        large = dataclasses.replace(
            incremental, active_tier="large", raw={**incremental.raw, "active_tier": "large"}
        )
        run_pipeline(large)
        # 3. modify one instruction
        tier_large = large.tier()
        json_path = sample_paths(large, "sample_0004", tier_large)["json"]
        instruction = json.loads(json_path.read_text())
        instruction["ops"][0]["attribute_value"] = 222
        atomic_write_json(json_path, instruction)
        run_pipeline(large)
        # 4. replace one image
        from workbench.sample_gen import render_image

        png_path = sample_paths(large, "sample_0007", tier_large)["png"]
        image = render_image(large, tier_large, 7)
        image[2, 2] = (1, 2, 3)
        png_path.write_bytes(png_bytes(image))
        run_pipeline(large)
        # 5. switch quantization level
        quant_config = dataclasses.replace(
            large, quant_override=16, raw={**large.raw, "quant_override": 16}
        )
        incr_final = run_pipeline(quant_config)
        # 6. narrow back to small
        small_quant = dataclasses.replace(
            quant_config, active_tier="small", raw={**quant_config.raw, "active_tier": "small"}
        )
        incr_small = run_pipeline(small_quant)

        # Blank-workspace full rebuild with identical final inputs.
        rebuild = self._clone_with_snapshot("snap-rebuild")
        rebuild_large = dataclasses.replace(
            rebuild, active_tier="large", quant_override=16,
            raw={**rebuild.raw, "active_tier": "large", "quant_override": 16},
        )
        full_large = run_pipeline(rebuild_large, run_mode="parallel", reset=True)
        rebuild_small = dataclasses.replace(
            rebuild, active_tier="small", quant_override=16,
            raw={**rebuild.raw, "active_tier": "small", "quant_override": 16},
        )
        full_small = run_pipeline(rebuild_small, reset=True)

        self.assertEqual(pixel_map(incr_final), pixel_map(full_large))
        self.assertEqual(pixel_map(incr_small), pixel_map(full_small))
        self.assertEqual(
            digest_obj(stage_summary_map(incr_final)),
            digest_obj(stage_summary_map(full_large)),
        )
        # decisions are content driven and must replay identically
        self.assertEqual(
            incr_final["global_decision"]["selected"],
            full_large["global_decision"]["selected"],
        )

    def test_only_affected_chain_recomputes_on_instruction_edit(self):
        run_pipeline(self.config)
        tier = self.config.tier()
        json_path = sample_paths(self.config, "sample_0002", tier)["json"]
        instruction = json.loads(json_path.read_text())
        instruction["ops"][1]["style_strength"] = 0.4242
        atomic_write_json(json_path, instruction)
        report = run_pipeline(self.config)
        recomputed = {s["sample"]: s["recomputed"] for s in report["samples"] if s["recomputed"]}
        self.assertEqual(list(recomputed), ["sample_0002"])
        self.assertEqual(set(recomputed["sample_0002"]), set(STAGE_NAMES))
        untouched = [s for s in report["samples"] if s["sample"] != "sample_0002"]
        self.assertTrue(all(not s["recomputed"] for s in untouched))

    def test_quant_switch_only_invalidates_style_and_composite(self):
        run_pipeline(self.config)
        quant_config = dataclasses.replace(
            self.config, quant_override=32, raw={**self.config.raw, "quant_override": 32}
        )
        report = run_pipeline(quant_config)
        self.assertEqual(report["recompute_counts"]["style"], self.config.tier().sample_count)
        self.assertEqual(report["recompute_counts"]["composite"], self.config.tier().sample_count)
        for stage in ("parse", "mask", "inpaint", "recolor"):
            self.assertEqual(report["recompute_counts"][stage], 0)


class ConcurrencyEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)
        self.config = load_test_config(self.workspace)

    def test_parallel_and_serial_field_identical_with_isolated_stores(self):
        root = Path(self.workspace.name)
        parallel_cfg = dataclasses.replace(
            self.config, snapshots_dir=str(root / "snap-p"),
            raw={**self.config.raw, "snapshots_dir": str(root / "snap-p")},
        )
        serial_cfg = dataclasses.replace(
            self.config, snapshots_dir=str(root / "snap-s"),
            raw={**self.config.raw, "snapshots_dir": str(root / "snap-s")},
        )
        parallel = run_pipeline(parallel_cfg, run_mode="parallel", reset=True)
        serial = run_pipeline(serial_cfg, run_mode="serial", reset=True)
        self.assertEqual(
            digest_obj(_comparable_blob(parallel)),
            digest_obj(_comparable_blob(serial)),
        )
        self.assertEqual(
            [s["sample"] for s in parallel["samples"]],
            [s["sample"] for s in serial["samples"]],
        )
        self.assertEqual(parallel["completed_samples"], serial["completed_samples"])
        self.assertEqual(parallel["failures"], serial["failures"])
        self.assertEqual(parallel["stage_cost_units"], serial["stage_cost_units"])


class FailureIsolationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()
        self.addCleanup(self.workspace.cleanup)
        self.config = load_test_config(self.workspace)

    def test_bad_sample_recorded_but_run_continues(self):
        first = run_pipeline(self.config)
        self.assertEqual(len(first["failures"]), 0)
        paths = sample_paths(self.config, "sample_0001", self.config.tier())
        paths["png"].write_bytes(b"broken")
        report = run_pipeline(self.config)
        self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(report["failures"][0]["code"], E_SAMPLE_INVALID)
        self.assertEqual(report["failures"][0]["sample"], "sample_0001")
        # failure list is ordered by sample name; remaining samples still succeed
        self.assertEqual(len(report["completed_samples"]), self.config.tier().sample_count - 1)
        self.assertEqual(
            [s["sample"] for s in report["samples"]],
            sorted(s["sample"] for s in report["samples"]),
        )
        # restoring the sample lets the next run recover it
        from workbench.sample_gen import generate_tier

        generate_tier(self.config, self.config.tier(), skip_existing=False)
        recovered = run_pipeline(self.config)
        self.assertEqual(len(recovered["failures"]), 0)
        self.assertEqual(len(recovered["completed_samples"]), self.config.tier().sample_count)


if __name__ == "__main__":
    unittest.main()
