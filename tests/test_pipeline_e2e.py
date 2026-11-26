import hashlib
import json
import os
import tempfile
import unittest

from fixedpoint_render.cli import main
from fixedpoint_render.errors import ErrorBundle
from fixedpoint_render.pipeline import Renderer
from fixedpoint_render.samplegen import default_config, generate_workspace
from fixedpoint_render.state import RenderState
from fixedpoint_render.workspace import load_workspace


def _fresh(workdir, frames=4, max_workers=1, **overrides):
    cfg = default_config()
    data = cfg.to_dict()
    data.update(max_workers=max_workers)
    data.update(overrides)
    from fixedpoint_render.config import RenderConfig
    config = RenderConfig(**data)
    generate_workspace(workdir, config, frames)
    return config


def _state(workdir):
    loaded = load_workspace(workdir)
    return RenderState(loaded["config"], len(loaded["frame_indices"]),
                       loaded["prompts"], loaded["images"], loaded["motions"])


def _png_hashes(workdir):
    out = {}
    for name in sorted(os.listdir(os.path.join(workdir, "output"))):
        if name.endswith(".png"):
            with open(os.path.join(workdir, "output", name), "rb") as fh:
                out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _report(workdir):
    with open(os.path.join(workdir, "diagnostic_report.json")) as fh:
        return json.load(fh)


class ConcurrencyParityTests(unittest.TestCase):
    def test_serial_vs_parallel_pixel_identical(self):
        serial_dir = tempfile.mkdtemp(prefix="fpr-ser-")
        parallel_dir = tempfile.mkdtemp(prefix="fpr-par-")
        _fresh(serial_dir, frames=5, max_workers=1)
        _fresh(parallel_dir, frames=5, max_workers=8)
        Renderer(serial_dir).run(_state(serial_dir))
        Renderer(parallel_dir).run(_state(parallel_dir))
        self.assertEqual(_png_hashes(serial_dir), _png_hashes(parallel_dir))
        r1, r2 = _report(serial_dir), _report(parallel_dir)
        self.assertEqual(r1["budget"], r2["budget"])
        self.assertEqual(
            r1["per_frame_error_attribution"],
            r2["per_frame_error_attribution"])


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="fpr-inc-")
        _fresh(self.workdir, frames=4)
        self.renderer = Renderer(self.workdir)
        self.base_state = _state(self.workdir)
        self.renderer.run(self.base_state)
        self.base_hashes = _png_hashes(self.workdir)

    def _edit(self, command):
        state = RenderState.load(os.path.join(self.workdir, "state.json"))
        state.apply(command)
        return self.renderer.run(state)

    def test_unchanged_rerun_reuses_everything(self):
        outcome = self.renderer.run(
            RenderState.load(os.path.join(self.workdir, "state.json")))
        self.assertEqual(outcome.recomputed, 0)
        self.assertGreater(outcome.reused, 0)
        self.assertEqual(_png_hashes(self.workdir), self.base_hashes)

    def test_single_motion_edit_only_invalidates_downstream(self):
        motion = json.loads(open(os.path.join(
            self.workdir, "motion/frame_01.json")).read())
        motion["tiles"][0]["vx"] = 99
        outcome = self._edit({"verb": "set_motion", "frame": 1,
                              "motion": motion})
        report = outcome.report
        # Only frame 1's motion encoder is rebuilt; its text/image encoders and
        # every other frame's encoders are reused.
        self.assertGreater(report["reuse"]["reused_total"], 0)
        recomputed_stages = report["reuse"]["recomputed_by_stage"]
        recomputed_encodes = [
            e for e in report["reuse"]["recomputed_by_stage"].items()]
        self.assertEqual(recomputed_stages.get("encode", 0), 1)
        self.assertEqual(recomputed_stages.get("fuse", 0), 1)
        # Frame 2 latent chain is recomputed purely from the frame dependency.
        self.assertEqual(recomputed_stages.get("denoise", 0), 2)
        # Frame 0 output must remain byte-identical.
        hashes = _png_hashes(self.workdir)
        self.assertEqual(hashes["frame_00.png"], self.base_hashes["frame_00.png"])
        self.assertNotEqual(hashes["frame_01.png"],
                            self.base_hashes["frame_01.png"])

    def test_weight_change_versions_config_and_keeps_determinism(self):
        before = _report(self.workdir)["version"]
        self._edit({"verb": "set_weight", "condition": "motion",
                    "value": "0.40"})
        after = _report(self.workdir)["version"]
        self.assertEqual(after, before + 1)
        # Re-run yields full reuse at the new version.
        outcome = self.renderer.run(
            RenderState.load(os.path.join(self.workdir, "state.json")))
        self.assertEqual(outcome.recomputed, 0)

    def test_full_command_sequence_matches_clean_render(self):
        commands = [
            {"verb": "set_prompt", "frame": 0, "prompt": "neon fog"},
            {"verb": "set_motion", "frame": 2,
             "motion": {"tiles": [{"tile": t, "vx": 7, "vy": -7}
                                  for t in range(default_config().tile_rows)],
                        "frame_cover": 80}},
            {"verb": "set_weight", "condition": "text", "value": "0.40"},
            {"verb": "set_frame_budget", "frame": 1, "budget": 3},
            {"verb": "set_qmode", "qmode": "nearest"},
        ]
        state = RenderState.load(os.path.join(self.workdir, "state.json"))
        for command in commands:
            state.apply(command)
        incremental = self.renderer.run(state)
        incremental_hashes = _png_hashes(self.workdir)
        incremental_report = incremental.report

        # Clean full render from the same final inputs.
        clean = tempfile.mkdtemp(prefix="fpr-clean-")
        _fresh(clean, frames=4)
        clean_state = _state(clean)
        for command in commands:
            clean_state.apply(command)
        clean_outcome = Renderer(clean).run(clean_state)
        self.assertEqual(_png_hashes(clean), incremental_hashes)
        self.assertEqual(clean_outcome.report["budget"],
                         incremental_report["budget"])
        self.assertEqual(
            clean_outcome.report["per_frame_error_attribution"],
            incremental_report["per_frame_error_attribution"])
        self.assertEqual(clean_outcome.report["degraded_frames"],
                         incremental_report["degraded_frames"])

    def test_image_delete_and_replace(self):
        self._edit({"verb": "delete_image", "frame": 2})
        deleted_hashes = _png_hashes(self.workdir)
        self.assertNotEqual(deleted_hashes["frame_02.png"],
                            self.base_hashes["frame_02.png"])
        # Replace with a deterministic new image payload.
        from PIL import Image
        img = Image.new("RGB", (default_config().width,
                                default_config().height), (200, 50, 50))
        self._edit({"verb": "set_image", "frame": 2,
                    "image": {"mode": img.mode, "size": list(img.size),
                              "bytes": list(img.tobytes())}})
        replaced = _png_hashes(self.workdir)
        self.assertNotEqual(replaced["frame_02.png"],
                            deleted_hashes["frame_02.png"])


class SnapshotRecoveryTests(unittest.TestCase):
    def test_forced_terminination_then_resume(self):
        workdir = tempfile.mkdtemp(prefix="fpr-crash-")
        _fresh(workdir, frames=3)
        renderer = Renderer(workdir)
        state = _state(workdir)
        from fixedpoint_render.snapshot import STAGES
        with self.assertRaises(RuntimeError):
            renderer.run(state, force_crash_after=STAGES[3])
        # Temp leftovers must not be treated as committed state.
        self.assertTrue(os.path.isdir(os.path.join(workdir, "snapshots", "tmp")))
        recovered = renderer.snapshots.recover()
        self.assertIsNotNone(recovered)
        version, manifest, _ = recovered
        self.assertIn(STAGES[3], manifest["committed_stages"])
        self.assertNotIn(STAGES[4], manifest["committed_stages"])
        # Resume completes and yields the same pixels as a clean run.
        renderer.run(state, resume=True)
        clean = tempfile.mkdtemp(prefix="fpr-recover-clean-")
        _fresh(clean, frames=3)
        Renderer(clean).run(_state(clean))
        self.assertEqual(_png_hashes(clean), _png_hashes(workdir))

    def test_corrupt_manifest_detected(self):
        workdir = tempfile.mkdtemp(prefix="fpr-corrupt-")
        _fresh(workdir, frames=2)
        Renderer(workdir).run(_state(workdir))
        with open(os.path.join(workdir, "snapshots", "manifest.json"),
                  "w") as fh:
            fh.write("{not-json")
        with self.assertRaises(ErrorBundle):
            Renderer(workdir).snapshots.recover()


if __name__ == "__main__":
    unittest.main()
