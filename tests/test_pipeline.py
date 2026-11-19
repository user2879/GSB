import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pcrecon import config
from pcrecon.application import Application
from pcrecon.binary_format import read_array, write_array
from pcrecon.errors import (
    BudgetExceededError,
    MeshMismatchError,
    NeighborRadiusError,
    PointCloudFormatError,
    SnapshotStaleError,
    UndoEmptyError,
)
from pcrecon.framing import FrameBudgeter
from pcrecon.generator import generate_sample
from pcrecon.hashing import digest_array, digest_file
from pcrecon.mesh import ReconstructionParams, full_rebuild
from pcrecon.normals import reestimate_local
from pcrecon.service import ReconstructionService, build_initial_snapshot
from pcrecon.spatial import VoxelGrid

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class GeneratorTests(unittest.TestCase):
    def test_byte_identical_regeneration(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ra = generate_sample(a)
            rb = generate_sample(b)
            self.assertEqual(ra["files"], rb["files"])
            for name in ra["files"]:
                self.assertEqual(digest_file(os.path.join(a, name)), digest_file(os.path.join(b, name)))

    def test_sample_shape_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            generate_sample(d)
            pts = read_array(os.path.join(d, config.POINTS_FILE), 3)
            nrm = read_array(os.path.join(d, config.NORMALS_FILE), 4)
            self.assertEqual(pts.shape, (120_000, 3))
            self.assertEqual(nrm.shape, (120_000, 4))
            self.assertTrue(np.isfinite(pts).all())
            self.assertTrue(np.all((nrm[:, 3] >= 0.0) & (nrm[:, 3] <= 1.0)))
            with open(os.path.join(d, config.MANIFEST_FILE)) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["stats"]["point_total"], 120_000)
            self.assertEqual(sum(manifest["stats"]["surface_counts"].values()), 120_000)


class BinaryFormatTests(unittest.TestCase):
    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            arr = np.random.default_rng(1).random((100, 3))
            path = os.path.join(d, "x.pcrc")
            write_array(path, arr)
            self.assertTrue(np.array_equal(read_array(path, 3), arr.astype(np.float64)))
            with open(path, "rb") as fh:
                raw = bytearray(fh.read())
            raw[20] ^= 0xFF
            with open(path, "wb") as fh:
                fh.write(raw)
            with self.assertRaises(PointCloudFormatError):
                read_array(path, 3)


class SpatialAndNormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as d:
            generate_sample(d)
            cls.points = read_array(os.path.join(d, config.POINTS_FILE), 3)
            cls.normals = read_array(os.path.join(d, config.NORMALS_FILE), 4)

    def test_invalid_radius_rejected(self):
        grid = VoxelGrid(self.points[:1000], 0.5)
        with self.assertRaises(NeighborRadiusError):
            grid.radius_search(self.points[:2], 0.0)

    def test_local_reestimate_preserves_outside_normals(self):
        pts = self.points[:8000].copy()
        nrm = self.normals[:8000].copy()
        grid = VoxelGrid(pts, 0.45)
        region = grid.points_within_region(np.array([1.0, 1.0, 0.0]), np.array([4.0, 4.0, 2.0]))
        targets = np.unique(np.concatenate(grid.radius_search(pts[region], 0.45)))
        updated, counts, audit = reestimate_local(pts, nrm, targets, 0.45)
        outside = np.setdiff1d(np.arange(pts.shape[0]), targets)
        self.assertTrue(np.array_equal(updated[outside], nrm[outside]))
        self.assertEqual(audit["radius"], 0.45)
        self.assertEqual(audit["reestimated_point_count"], targets.size)
        self.assertTrue((counts > 0).all())
        self.assertIn("neighbour_count_max", audit)


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as d:
            generate_sample(d)
            cls.points = read_array(os.path.join(d, config.POINTS_FILE), 3)
            cls.normals = read_array(os.path.join(d, config.NORMALS_FILE), 4)
            with open(os.path.join(d, config.EDITS_FILE)) as fh:
                cls.edits = json.load(fh)["edits"]

    def test_full_rebuild_deterministic(self):
        params = ReconstructionParams()
        a = full_rebuild(self.points, params)
        b = full_rebuild(self.points, params)
        self.assertTrue(np.array_equal(a["merged"]["vertices"], b["merged"]["vertices"]))
        self.assertTrue(np.array_equal(a["merged"]["faces"], b["merged"]["faces"]))

    def test_incremental_matches_full_and_closure_matches_rebuilt(self):
        service = ReconstructionService(build_initial_snapshot(self.points, self.normals))
        for edit in self.edits:
            result = service.apply_edit(edit, base_version=service.snapshot.version)
            self.assertEqual(set(result.invalidated), set(result.rebuilt) | set())
            if edit["type"] == "cull_region":
                self.assertEqual(set(result.invalidated), set(result.rebuilt))
                self.assertTrue(result.reused)
            self.assertEqual(result.consistency["status"], "match")
            # Independent full rebuild on the edited point set.
            independent = full_rebuild(service.snapshot.points, service.snapshot.params)
            self.assertTrue(
                np.allclose(
                    service.snapshot.merged["vertices"],
                    independent["merged"]["vertices"],
                    atol=1e-7,
                )
            )

    def test_cached_tiles_byte_identical(self):
        service = ReconstructionService(build_initial_snapshot(self.points, self.normals))
        before = {k: digest_array(v["vertices"]) for k, v in service.snapshot.tiles.items()}
        edit = self.edits[0]
        result = service.apply_edit(edit, base_version=0)
        for tile_id in result.reused:
            self.assertEqual(before[tile_id], digest_array(service.snapshot.tiles[tile_id]["vertices"]))

    def test_stale_snapshot_rejected(self):
        service = ReconstructionService(build_initial_snapshot(self.points, self.normals))
        service.apply_edit(self.edits[0], base_version=0)
        with self.assertRaises(SnapshotStaleError):
            service.apply_edit(self.edits[1], base_version=0)

    def test_mesh_mismatch_locates_first_vertex(self):
        from pcrecon.consistency import compare_meshes

        snap = build_initial_snapshot(self.points, self.normals)
        full = full_rebuild(self.points, snap.params)
        broken = {
            "vertices": snap.merged["vertices"].copy(),
            "faces": snap.merged["faces"].copy(),
        }
        broken["vertices"][10] += 5.0
        with self.assertRaises(MeshMismatchError) as ctx:
            compare_meshes(broken, full["merged"], op_id=7)
        self.assertEqual(ctx.exception.context["first_divergent_vertex"], 10)
        self.assertEqual(ctx.exception.context["op_id"], 7)


class FramingTests(unittest.TestCase):
    def test_degradation_ladder_and_budget(self):
        budgeter = FrameBudgeter(33.3)
        record = budgeter.plan_frame("box_select", 120_000, 600, 0)
        self.assertLessEqual(record.chosen_cost_ms, 33.3)
        self.assertTrue(record.degradation_reasons)
        report = budgeter.report()
        self.assertEqual(report["frame_count"], 1)

    def test_budget_exceeded_error(self):
        budgeter = FrameBudgeter(0.1)
        with self.assertRaises(BudgetExceededError):
            budgeter.plan_frame("idle", 120_000, 600, 0)


class ApplicationTests(unittest.TestCase):
    def test_end_to_end_reports_reproducible_and_undo(self):
        with tempfile.TemporaryDirectory() as d:
            generate_sample(os.path.join(d, "data"))
            app = Application(d)
            summary = app.run_edit_sequence()
            restored = app.undo_last()
            head = app.service.snapshot
            # Undo restores points/normals/mesh hashes of the previous version.
            prior = app._version_stack[-1]
            self.assertEqual(head.points_hash, prior.points_hash)
            self.assertEqual(head.normals_hash, prior.normals_hash)
            self.assertEqual(head.mesh_hash, prior.mesh_hash)
            app.write_clean_exit_report()
            # Second run with identical inputs yields byte-identical reports.
            app2 = Application(d)
            summary2 = app2.run_edit_sequence()
            app2.write_clean_exit_report()
            self.assertEqual(
                digest_file(summary["incremental_report"]),
                digest_file(summary2["incremental_report"]),
            )
            self.assertEqual(
                digest_file(summary["frame_report"]),
                digest_file(summary2["frame_report"]),
            )

    def test_undo_empty(self):
        with tempfile.TemporaryDirectory() as d:
            generate_sample(os.path.join(d, "data"))
            app = Application(d)
            with self.assertRaises(UndoEmptyError):
                app.undo_last()

    def test_crash_recovery_reaps_orphans_without_exit_report(self):
        with tempfile.TemporaryDirectory() as d:
            generate_sample(os.path.join(d, "data"))
            app = Application(d)
            app.run_edit_sequence()
            orphan = os.path.join(d, "ledger", "cache", "tmp-crash")
            os.makedirs(orphan)
            app.ledger.remove_exit_report()
            recovered = app.ledger.recover()
            self.assertIn("tmp-crash", recovered["recovered_orphans"])
            self.assertFalse(os.path.exists(orphan))
            self.assertFalse(os.path.exists(os.path.join(d, "ledger", "exit_report.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class GLContextTests(unittest.TestCase):
    def test_context_failure_raises_code_not_crash(self):
        from unittest import mock

        from pcrecon import render_gl
        from pcrecon.errors import GLContextError

        with mock.patch.object(render_gl, "verify_offscreen_context", side_effect=GLContextError("simulated FBO failure")):
            with self.assertRaises(GLContextError) as ctx:
                render_gl.verify_offscreen_context()
        self.assertEqual(ctx.exception.code, "E_GL_CONTEXT")

    def test_offscreen_context_live(self):
        from pcrecon.render_gl import verify_offscreen_context

        info = verify_offscreen_context()
        self.assertEqual(info["status"], "ok")
        self.assertGreater(info["fbo_id"], 0)


class PatchDependencyTests(unittest.TestCase):
    def test_indivisible_grid_step_raises_dependency_code(self):
        from pcrecon.errors import PatchDependencyError
        from pcrecon.mesh import ReconstructionParams, full_rebuild

        pts = np.zeros((12, 3), dtype=np.float64)
        bad = ReconstructionParams(voxel_size=1.0, support_radius=0.85, mesh_grid_step=0.3)
        with self.assertRaises(PatchDependencyError) as ctx:
            full_rebuild(pts, bad)
        self.assertEqual(ctx.exception.code, "E_PATCH_DEPENDENCY")
