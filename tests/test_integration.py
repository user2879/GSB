import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_workbench.config import load_prompts, load_yaml, validate_config
from render_workbench.edits import EditPlanner
from render_workbench.pipeline import Pipeline
from render_workbench.png_codec import encode_png
from render_workbench.hashing import fingerprint
from render_workbench.config import config_fingerprint_payload


def write_minimal_project(tmp, steps=3, concurrency=2):
    vocab = ["a", "b", "c", "d"]
    config = {
        "vocab": vocab,
        "embedding_dim": 5,
        "latent_channels": 3,
        "seed": 10,
        "seed_min": 1,
        "seed_max": 1000,
        "denoise_steps": steps,
        "guidance_scale": 1.0,
        "img2img_replace_steps": 1,
        "patch_tolerance": 0.01,
        "concurrency": concurrency,
        "snapshot_interval": 1,
        "tiers": [
            {"id": "low", "width": 5, "height": 4,
             "noise": [0.1 * i for i in range(1, steps + 1)],
             "guidance": [0.2 * i for i in range(1, steps + 1)]},
        ],
        "noise_tables": {
            "low": [0.1 * i for i in range(1, steps + 1)]},
        "guidance_tables": {
            "low": [0.2 * i for i in range(1, steps + 1)]},
    }
    cfg_path = os.path.join(tmp, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as h:
        yaml.safe_dump(config, h)

    pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    img_path = os.path.join(tmp, "init.png")
    with open(img_path, "wb") as h:
        h.write(encode_png(pixels))

    prompts = [
        {"id": "p1", "tokens": ["a", "b"]},
        {"id": "p2", "tokens": ["c", "d"], "init_image": "init.png"},
    ]
    prompt_path = os.path.join(tmp, "prompts.jsonl")
    with open(prompt_path, "w", encoding="utf-8") as h:
        for record in prompts:
            h.write(json.dumps(record) + "\n")
    return cfg_path, prompt_path


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workdir = os.path.join(self.tmp, "work")
        self.cfg_path, self.prompt_path = write_minimal_project(self.tmp)
        cfg_raw = load_yaml(self.cfg_path)
        self.cfg, self.cfg_errors = validate_config(cfg_raw)
        self.prompts, self.prompt_errors = load_prompts(
            self.prompt_path, self.cfg, base_dir=self.tmp)
        self.assertEqual(self.cfg_errors, [])
        self.assertEqual(self.prompt_errors, [])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, cfg=None, prompts=None):
        cfg = cfg or self.cfg
        prompts = prompts or self.prompts
        pipeline = Pipeline(cfg, self.workdir)
        pipeline.store.recover()
        pipeline.execute(prompts)
        return pipeline

    def test_full_render_populates_all_nodes(self):
        pipeline = self._run()
        nodes = pipeline.store.manifest["nodes"]
        kinds = {entry["kind"] for entry in nodes.values()}
        self.assertEqual(
            kinds, {"embedding", "init_latent", "img_encode",
                    "denoise", "frame"})
        for node_id, entry in nodes.items():
            self.assertTrue(pipeline.store.verify_node(node_id, entry))

    def test_second_run_reuses_everything(self):
        first = self._run()
        first_count = first.computed_count
        second = self._run()
        self.assertEqual(second.computed_count, 0)
        self.assertEqual(len(second.reuse_records), len(first.store.manifest["nodes"]))
        self.assertEqual(first_count, len(first.store.manifest["nodes"]))

    def test_prompt_edit_only_invalidates_own_chain(self):
        self._run()
        planner = EditPlanner(self.cfg, self.prompts, self.tmp)
        errors = planner.apply_batch({
            "id": "edit", "edits": [
                {"op": "replace_prompt_tokens", "prompt_id": "p1",
                 "tokens": ["a", "c"]}]})
        self.assertEqual(errors, [])
        pipeline = Pipeline(planner.cfg, self.workdir)
        pipeline.execute(planner.prompt_list())
        recomputed = {r["node_id"] for r in pipeline.recompute_records}
        self.assertIn("embedding::p1", recomputed)
        self.assertNotIn("embedding::p2", recomputed)
        self.assertTrue(all("p2" not in node_id for node_id in recomputed))

    def test_seed_change_keeps_embeddings(self):
        self._run()
        planner = EditPlanner(self.cfg, self.prompts, self.tmp)
        planner.apply_batch({"id": "seed", "edits": [
            {"op": "set_seed", "seed": 99}]})
        pipeline = Pipeline(planner.cfg, self.workdir)
        pipeline.execute(planner.prompt_list())
        recomputed = {r["node_id"] for r in pipeline.recompute_records}
        self.assertNotIn("embedding::p1", recomputed)
        self.assertIn("init_latent::p1::low", recomputed)

    def test_failed_batch_rolls_back_state(self):
        self._run()
        import copy
        before = copy.deepcopy(self.cfg)
        planner = EditPlanner(self.cfg, self.prompts, self.tmp)
        errors = planner.apply_batch({"id": "bad", "edits": [
            {"op": "set_seed", "seed": 999999}]})
        self.assertTrue(errors)
        self.assertEqual(planner.cfg["seed"], before["seed"])
        pipeline = Pipeline(planner.cfg, self.workdir)
        pipeline.execute(planner.prompt_list())
        self.assertEqual(pipeline.computed_count, 0)

    def test_patch_boundary_and_noop(self):
        self._run()
        planner = EditPlanner(self.cfg, self.prompts, self.tmp)
        errors = planner.apply_batch({"id": "patch", "edits": [
            {"op": "add_patch", "prompt_id": "p1", "patch": {
                "step": 1, "x": 0, "y": 0, "width": 2, "height": 2,
                "values": [0.9] * (2 * 2 * 3)}}]})
        self.assertEqual(errors, [])
        pipeline = Pipeline(planner.cfg, self.workdir)
        pipeline.execute(planner.prompt_list())
        recomputed = {r["node_id"] for r in pipeline.recompute_records}
        self.assertNotIn("embedding::p1", recomputed)
        self.assertNotIn("init_latent::p1::low", recomputed)
        self.assertTrue(any(n.startswith("patch::p1") for n in recomputed))
        self.assertTrue(any(n.startswith("frame::txt2img::p1")
                            for n in recomputed))

        # A patch carrying the existing latent values (plus a sub-tolerance
        # nudge) changes no content and must be flagged as a no-op.
        source_id = "denoise::txt2img::p1::low::s0002"
        current = pipeline.store.read_npy(
            source_id, "latent.bin", (4, 5, 3), np.float64)
        existing = current[1, 1, :].tolist()
        nudged = [value + 0.001 for value in existing]  # within 0.01 tolerance
        planner2 = EditPlanner(planner.cfg, planner.prompt_list(), self.tmp)
        errors = planner2.apply_batch({"id": "noop", "edits": [
            {"op": "add_patch", "prompt_id": "p1", "patch": {
                "step": 2, "x": 1, "y": 1, "width": 1, "height": 1,
                "values": nudged}}]})
        self.assertEqual(errors, [])
        pipeline2 = Pipeline(planner2.cfg, self.workdir)
        pipeline2.execute(planner2.prompt_list())
        noop_entry = next(
            entry for nid, entry in pipeline2.store.manifest["nodes"].items()
            if nid.startswith("patch::p1") and entry.get("step_index") == 2)
        self.assertTrue(noop_entry["patch_noop"])
        # A no-op patch node still exists (the edit was applied), but its
        # output equals the source latent, so no downstream fingerprint moves.
        downstream_before = pipeline.store.manifest["nodes"][
            "frame::txt2img::p1::low::s0002"]["content_fp"]
        downstream_after = pipeline2.store.manifest["nodes"][
            "frame::txt2img::p1::low::s0002"]["content_fp"]
        self.assertEqual(downstream_before, downstream_after)

    def test_concurrency_matches_serial(self):
        serial_dir = os.path.join(self.tmp, "serial")
        parallel_dir = os.path.join(self.tmp, "parallel")
        cfg_serial = dict(self.cfg, concurrency=1)
        cfg_parallel = dict(self.cfg, concurrency=4)
        serial = Pipeline(cfg_serial, serial_dir)
        serial.execute(self.prompts)
        parallel = Pipeline(cfg_parallel, parallel_dir)
        parallel.execute(self.prompts)
        self.assertEqual(
            set(serial.store.manifest["nodes"]),
            set(parallel.store.manifest["nodes"]))
        for node_id in serial.store.manifest["nodes"]:
            self.assertEqual(
                serial.store.manifest["nodes"][node_id]["content_fp"],
                parallel.store.manifest["nodes"][node_id]["content_fp"])

    def test_recovery_drops_partial_nodes(self):
        pipeline = self._run()
        node_count = len(pipeline.store.manifest["nodes"])
        # Corrupt one artifact file to simulate a torn write.
        any_node = next(iter(pipeline.store.manifest["nodes"]))
        target = pipeline.store.node_file(any_node,
            next(iter(pipeline.store.manifest["nodes"][any_node]["file_fps"])))
        with open(target, "r+b") as h:
            h.write(b"\x00\x00")
        recovered = Pipeline(self.cfg, self.workdir)
        dropped = recovered.store.recover()
        self.assertIn(any_node, dropped)
        recovered.execute(self.prompts)
        self.assertEqual(len(recovered.store.manifest["nodes"]), node_count)


if __name__ == "__main__":
    unittest.main()
