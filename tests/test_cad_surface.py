from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cad_surface.engine import CadApplication
from cad_surface.errors import DomainError, ErrorCode
from cad_surface.geometry import PatchGeometry
from cad_surface.model import NURBSModel, generate_model
from cad_surface.persistence import PersistentStore
from cad_surface.state import build_initial_state, derive_edited_state


class BaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def assert_error(self, code: ErrorCode, callable_) -> DomainError:
        with self.assertRaises(DomainError) as context:
            callable_()
        self.assertEqual(context.exception.code, code, context.exception.as_dict())
        return context.exception


class ModelTests(BaseTest):
    def test_deterministic_generation_roundtrip_and_validation(self) -> None:
        first = generate_model(12345)
        second = generate_model(12345)
        self.assertEqual(first.to_bytes(), second.to_bytes())
        self.assertEqual(first.content_hash(), second.content_hash())
        roundtrip = NURBSModel.from_dict(copy.deepcopy(first.to_dict()))
        self.assertEqual(roundtrip.to_bytes(), first.to_bytes())

        for patch in first.patches:
            self.assertEqual(
                len(patch.knots_u),
                patch.control_shape[0] + patch.degree_u + 1,
            )
            self.assertEqual(
                len(patch.knots_v),
                patch.control_shape[1] + patch.degree_v + 1,
            )
            self.assertTrue(np.all(patch.knots_u[1:] >= patch.knots_u[:-1]))
            self.assertTrue(np.all(patch.knots_v[1:] >= patch.knots_v[:-1]))

        malformed = copy.deepcopy(first.to_dict())
        malformed["patches"][0]["degree_u"] = 0
        self.assert_error(ErrorCode.SCHEMA_VERSION, lambda: NURBSModel.from_dict(malformed))

        bad_knots = copy.deepcopy(first.to_dict())
        bad_knots["patches"][0]["knots_u"][2] = 0.8
        bad_knots["patches"][0]["knots_u"][3] = 0.0
        self.assert_error(ErrorCode.KNOT_MONOTONIC, lambda: NURBSModel.from_dict(bad_knots))

        bad_length = copy.deepcopy(first.to_dict())
        bad_length["patches"][0]["knots_v"].pop()
        self.assert_error(ErrorCode.SCHEMA_VERSION, lambda: NURBSModel.from_dict(bad_length))

    def test_unified_evaluation_pipeline_and_byte_repeatability(self) -> None:
        model = generate_model(77)
        patch = model.patch("p00")
        for u, v in ((0.0, 0.0), (0.17, 0.63), (1.0, 1.0)):
            first = patch.evaluate(u, v)
            second = patch.evaluate(u, v)
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertEqual(first.content_hash(), second.content_hash())
            self.assertEqual(first.degree_u, patch.degree_u)
            self.assertEqual(first.degree_v, patch.degree_v)
            self.assertTrue(first.support_u)
            self.assertTrue(first.support_v)
            self.assertIn("tensor_product_rational_bspline", first.convention)
            norm = float(np.linalg.norm(first.normal))
            self.assertAlmostEqual(norm, 1.0, places=14)

        parameters = np.linspace(0.0, 1.0, 9, dtype="<f8")
        grid = patch.evaluate_grid(parameters, parameters)
        for i, j in ((0, 0), (2, 5), (8, 8), (4, 7)):
            sample = patch.evaluate(float(parameters[i]), float(parameters[j]))
            self.assertEqual(tuple(grid.points[i, j]), sample.point)
            self.assertEqual(tuple(grid.normals[i, j]), sample.normal)
            self.assertEqual(int(grid.span_u[i, j]), sample.span_u)
            self.assertEqual(int(grid.span_v[i, j]), sample.span_v)

        geometry = PatchGeometry.build(patch, version=0, u_values=parameters, v_values=parameters)
        self.assertEqual(geometry.points.tobytes(), grid.points.tobytes())
        self.assertEqual(geometry.normals.tobytes(), grid.normals.tobytes())
        u_segments, v_segments = geometry.iso_segments()
        flat_points = geometry.points.reshape(geometry.vertex_count, 3)
        for a, b in u_segments[:10] + v_segments[:10]:
            self.assertTrue(np.array_equal(flat_points[a], geometry.points[a // geometry.width, a % geometry.width]))
            self.assertTrue(np.array_equal(flat_points[b], geometry.points[b // geometry.width, b % geometry.width]))
        for triangle in geometry.triangles[:20]:
            self.assertTrue(np.all(triangle >= 0))
            self.assertTrue(np.all(triangle < geometry.vertex_count))
        self.assertEqual(geometry.evaluation_record["degree_u"], patch.degree_u)
        self.assertEqual(geometry.evaluation_record["sample_count"], [9, 9])

        state = build_initial_state(model, grid_resolution=7)
        first_display = {mode: buffer.content_hash for mode, buffer in state.render_all().items()}
        second_display = {mode: buffer.content_hash for mode, buffer in state.render_all().items()}
        self.assertEqual(first_display, second_display)
        self.assertEqual(state.geometries["p00"].vertex_count, 49)


class ClosureTests(BaseTest):
    def test_local_edit_closure_is_exact(self) -> None:
        model = generate_model(2026)
        initial = build_initial_state(model, grid_resolution=10)
        previous = initial.geometries["p00"]
        point = model.patch("p00").control_points[4, 4].copy()
        edited, _, closure = derive_edited_state(
            initial,
            new_version=1,
            operation_version=1,
            patch_key="p00",
            index_u=4,
            index_v=4,
            new_value=[float(point[0]), float(point[1]), float(point[2] + 0.2)],
        )
        geometry = edited.geometries["p00"]
        self.assertGreater(len(closure.vertices), 0)
        self.assertLess(len(closure.vertices), geometry.vertex_count)
        self.assertEqual(geometry.recompute_trace["actual_recomputed"]["patch_keys"], ["p00"])
        self.assertEqual(tuple(geometry.recompute_trace["actual_recomputed"]["vertices"]), closure.vertices)
        self.assertEqual(tuple(geometry.recompute_trace["actual_recomputed"]["iso_samples"]), closure.iso_samples)
        self.assertEqual(tuple(geometry.recompute_trace["actual_recomputed"]["triangles"]), closure.triangles)

        mask = np.ones(geometry.vertex_count, dtype=bool)
        mask[np.asarray(closure.vertices, dtype=np.int64)] = False
        for old, new in (
            (previous.points, geometry.points),
            (previous.du, geometry.du),
            (previous.dv, geometry.dv),
            (previous.normals, geometry.normals),
        ):
            self.assertEqual(
                old.reshape(geometry.vertex_count, 3)[mask].tobytes(),
                new.reshape(geometry.vertex_count, 3)[mask].tobytes(),
            )
        for key in ("p01", "p02"):
            self.assertEqual(initial.geometries[key].geometry_hash, edited.geometries[key].geometry_hash)
            self.assertEqual(initial.geometries[key].normals_hash, edited.geometries[key].normals_hash)
            self.assertEqual(initial.geometries[key].points.tobytes(), edited.geometries[key].points.tobytes())

        tampered = closure.__class__(
            patch_keys=closure.patch_keys,
            vertices=closure.vertices[1:],
            iso_samples=closure.iso_samples[1:],
            triangles=closure.triangles,
        )
        self.assert_error(
            ErrorCode.CLOSURE_MISMATCH,
            lambda: geometry.verify_closure(previous, tampered),
        )

    def test_geometry_stale_for_out_of_domain_and_wrong_patch(self) -> None:
        model = generate_model(8)
        patch = model.patch("p00")
        self.assert_error(ErrorCode.GEOM_STALE, lambda: patch.evaluate(-0.1, 0.5))


class HistoryTests(BaseTest):
    def test_undo_redo_branch_depth_and_reconstruction(self) -> None:
        app = CadApplication.create(
            self.base / "workspace",
            seed=99,
            max_undo_depth=2,
            grid_resolution=8,
        )
        self.addCleanup(lambda: app.close() if not app._closed else None)
        initial_hash = app.current.state_hash

        p = app.current.model.patch("p00").control_points[4, 4].copy()
        state1 = app.edit_control_point("p00", 4, 4, [p[0], p[1], p[2] + 0.11])
        p = app.current.model.patch("p00").control_points[4, 3].copy()
        state2 = app.edit_control_point("p00", 4, 3, [p[0], p[1], p[2] - 0.07])
        p = app.current.model.patch("p01").control_points[3, 3].copy()
        state3 = app.edit_control_point("p01", 3, 3, [p[0], p[1], p[2] + 0.09])
        self.assertEqual(state3.version, 3)
        self.assertEqual(state3.parent_version, 2)
        operation_one = app.history.operations[1]
        operation_two = app.history.operations[2]
        self.assertEqual(operation_one.patch_key, "p00")
        self.assertEqual((operation_one.index_u, operation_one.index_v), (4, 4))
        self.assertEqual(operation_one.parent_version, 0)
        self.assertNotEqual(operation_one.before, operation_one.after)
        self.assertEqual(operation_two.parent_version, 1)
        self.assertEqual((operation_two.index_u, operation_two.index_v), (4, 3))

        replayed_p00 = app.reconstruct_patch_at_version(3, "p00")
        self.assertTrue(replayed_p00.to_dict()["match"])
        self.assertFalse(replayed_p00.full_model_rebuild)
        self.assertEqual(replayed_p00.patches_rebuilt, 1)
        self.assertLess(replayed_p00.vertex_evaluations, replayed_p00.total_patch_vertices * 3)
        replayed_p01 = app.reconstruct_patch_at_version(3, "p01")
        self.assertTrue(replayed_p01.to_dict()["match"])

        self.assertEqual(app.undo().state_hash, state2.state_hash)
        self.assertEqual(app.current.display_hashes, state2.display_hashes)
        self.assertEqual(app.redo().state_hash, state3.state_hash)
        self.assertEqual(app.current.display_hashes, state3.display_hashes)

        self.assertEqual(app.undo().version, 2)
        self.assertEqual(app.undo().version, 1)
        self.assert_error(ErrorCode.UNDO_EMPTY, app.undo)

        p = app.current.model.patch("p02").control_points[2, 2].copy()
        branch = app.edit_control_point("p02", 2, 2, [p[0], p[1], p[2] + 0.15])
        self.assertEqual(branch.version, 4)
        self.assertEqual(branch.parent_version, 1)
        self.assertEqual(app.history.redo_stack, [])
        self.assertNotEqual(branch.state_hash, initial_hash)
        self.assertEqual(state3.version, 3)

        self.assertEqual(app.switch_to(1).version, 1)
        self.assertEqual(app.current.state_hash, state1.state_hash)


class PersistenceTests(BaseTest):
    def test_controlled_close_and_reopen_restores_exact_hashes(self) -> None:
        root = self.base / "workspace"
        app = CadApplication.create(root, seed=11, max_undo_depth=8, grid_resolution=8)
        point = app.current.model.patch("p00").control_points[4, 4].copy()
        app.edit_control_point("p00", 4, 4, [point[0], point[1], point[2] + 0.14])
        expected = app.state_summary()
        expected_buffers = {mode: buffer.content_hash for mode, buffer in app.current.render_all().items()}
        close_report = app.close()
        self.assertTrue(close_report["closed"])

        reopened = CadApplication.open(root, max_undo_depth=8)
        self.assertIsNotNone(reopened.recovery)
        self.assertEqual(reopened.recovery.status, "CONTROLLED_EXIT")
        self.assertEqual(reopened.current.state_hash, expected["state_hash"])
        self.assertEqual(reopened.current.display_hashes, expected_buffers)
        self.assertEqual(reopened.undo().version, 0)
        self.assertEqual(reopened.redo().state_hash, expected["state_hash"])
        reopened.close()

    def test_forced_termination_recovers_last_commit_and_cleans_orphan(self) -> None:
        root_a = self.base / "state_a"
        root_b = self.base / "state_b"
        app_a = CadApplication.create(root_a, seed=22, max_undo_depth=8, grid_resolution=8)
        app_b = CadApplication.create(root_b, seed=22, max_undo_depth=8, grid_resolution=8)
        point = app_a.current.model.patch("p00").control_points[4, 4].copy()
        value = [point[0], point[1], point[2] + 0.12]
        app_a.edit_control_point("p00", 4, 4, value)
        app_b.edit_control_point("p00", 4, 4, value)
        expected = app_a.state_summary()

        stage = app_a.stage_crash_probe(
            "p01",
            3,
            3,
            [float(app_a.current.model.patch("p01").control_points[3, 3, 0]), 0.0, 1.0],
        )
        self.assertTrue(Path(stage["staging_path"]).exists())

        recovered = CadApplication.open(root_a, max_undo_depth=8)
        self.assertIsNotNone(recovered.recovery)
        self.assertEqual(recovered.recovery.status, "FORCED_TERMINATION")
        self.assertGreaterEqual(len(recovered.recovery.orphan_paths), 1)
        self.assertFalse(Path(stage["staging_path"]).exists())
        self.assertEqual(recovered.current.state_hash, expected["state_hash"])
        self.assertEqual(recovered.current.display_hashes, expected["display_hashes"])

        continuation = [
            float(app_b.current.model.patch("p02").control_points[2, 2, 0]),
            0.0,
            1.0,
        ]
        recovered.edit_control_point("p02", 2, 2, continuation)
        app_b.edit_control_point("p02", 2, 2, continuation)
        self.assertEqual(recovered.current.state_hash, app_b.current.state_hash)
        self.assertEqual(recovered.current.display_hashes, app_b.current.display_hashes)
        self.assertEqual(recovered.current.version, app_b.current.version)
        recovered.close()
        app_b.close()

    def test_cache_hash_snapshot_stale_and_missing_manifest_errors(self) -> None:
        root = self.base / "workspace"
        app = CadApplication.create(root, seed=33, max_undo_depth=4, grid_resolution=8)
        point = app.current.model.patch("p00").control_points[4, 4].copy()
        app.edit_control_point("p00", 4, 4, [point[0], point[1], point[2] + 0.05])
        app.close()

        cache_copy = self.base / "cache_copy"
        shutil.copytree(root, cache_copy)
        manifest = json.loads((cache_copy / "MANIFEST.json").read_text(encoding="utf-8"))
        version = int(manifest["current_version"])
        cache_file = cache_copy / "commits" / f"v{version:06d}" / "patches" / "p00.json"
        cache_file.write_bytes(cache_file.read_bytes() + b" ")
        self.assert_error(
            ErrorCode.CACHE_HASH,
            lambda: CadApplication.open(cache_copy, max_undo_depth=4),
        )

        stale_copy = self.base / "stale_copy"
        shutil.copytree(root, stale_copy)
        manifest_path = stale_copy / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["current_state_hash"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        self.assert_error(
            ErrorCode.SNAPSHOT_STALE,
            lambda: CadApplication.open(stale_copy, max_undo_depth=4),
        )

        missing_copy = self.base / "missing_copy"
        shutil.copytree(root, missing_copy)
        (missing_copy / "MANIFEST.json").unlink()
        self.assert_error(
            ErrorCode.SNAPSHOT_STALE,
            lambda: CadApplication.open(missing_copy, max_undo_depth=4),
        )

        store = PersistentStore(root)
        store.open()
        manifest = store.manifest()
        self.assert_error(
            ErrorCode.SNAPSHOT_STALE,
            lambda: store.load_state(int(manifest["current_version"]), expected_state_hash="0" * 64),
        )

    def test_stale_manifest_revision_cannot_overwrite_newer_state(self) -> None:
        root = self.base / "revision"
        app = CadApplication.create(root, seed=44, max_undo_depth=4, grid_resolution=8)
        app.close()
        store_one = PersistentStore(root)
        store_one.open()
        store_two = PersistentStore(root)
        store_two.open()
        manifest = store_one.manifest()
        version = int(manifest["current_version"])
        state_hash = str(manifest["current_state_hash"])

        store_one.save_session(
            current_version=version,
            current_state_hash=state_hash,
            undo_stack=[],
            redo_stack=[],
            closed=True,
        )
        self.assert_error(
            ErrorCode.SNAPSHOT_STALE,
            lambda: store_two.save_session(
                current_version=version,
                current_state_hash=state_hash,
                undo_stack=[],
                redo_stack=[],
                closed=True,
            ),
        )
        loaded = store_one.load_state(version)
        self.assertEqual(loaded.state.state_hash, state_hash)

        versions = store_one.commit_versions()
        self.assertEqual(list(versions), sorted(versions))
        for version in versions:
            loaded = store_one.load_state(version)
            self.assertEqual(loaded.state.version, version)
            self.assertEqual(set(loaded.state.geometries), {"p00", "p01", "p02"})


if __name__ == "__main__":
    unittest.main()
