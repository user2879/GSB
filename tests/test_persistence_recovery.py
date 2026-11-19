"""Business rules 5-6: persistence, version recovery and block invalidation."""

import json
import os
import unittest

import numpy as np

from pointcloud_engine.engine import render_scene
from pointcloud_engine.io_utils import read_json
from pointcloud_engine.pointcloud import load_point_cloud, save_point_cloud
from pointcloud_engine.render import prepare_points, tile_origin
from pointcloud_engine.scene import load_scene
from pointcloud_engine.state import TileStore, compute_input_digest
from tests._harness import Workspace


class PersistenceRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = Workspace()
        self.scene_path = self.ws.write()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def _render(self):
        return render_scene(self.scene_path, install_signals=False)

    def test_tiles_persisted_and_reused_on_second_run(self):
        first = self._render()
        self.assertEqual(first["status"], "completed")
        state_files = [n for n in os.listdir(self.ws.path("state")) if n.endswith(".npz")]
        self.assertEqual(len(state_files), first["report"]["tile_count"])

        second = self._render()
        self.assertEqual(second["status"], "completed")
        self.assertEqual(sorted(second["report"]["tiles_reused"]),
                         list(range(second["report"]["tile_count"])))
        self.assertEqual(second["report"]["tiles_rendered"], [])

        g1 = np.load(self.ws.path("output", "gray.npy"))
        # Outputs identical and not replaced on same-version rerun.
        self.assertFalse(second["report"]["output"]["write"]["replaced"])

    def test_version_change_archives_old_tiles_and_rerenders_all(self):
        self._render()
        old_tiles = sorted(n for n in os.listdir(self.ws.path("state")) if n.endswith(".npz"))

        with open(self.scene_path, encoding="utf-8") as handle:
            scene = json.load(handle)
        scene["camera"]["vfov_deg"] = 75.0  # changes canonical version
        with open(self.scene_path, "w", encoding="utf-8") as handle:
            json.dump(scene, handle)

        result = self._render()
        self.assertEqual(result["status"], "completed")
        # All tiles freshly rendered, old ones archived, nothing silently reused.
        self.assertEqual(sorted(result["report"]["tiles_rendered"]),
                         list(range(result["report"]["tile_count"])))
        self.assertTrue(len(result["report"]["archived_files"]) >= len(old_tiles))
        archive_root = self.ws.path("state", "archive")
        self.assertTrue(os.path.isdir(archive_root))
        archived = []
        for root, _, files in os.walk(archive_root):
            archived.extend(files)
        self.assertTrue(any(name.endswith(".npz") for name in archived))

    def test_local_point_edit_invalidates_only_affected_tiles(self):
        first = self._render()
        scene = load_scene(self.scene_path)
        prepared = prepare_points(scene)

        # Pick a visible point and compute exactly which tiles its footprint touches.
        point_global_index = int(prepared.index[0])
        compact = 0
        cx, cy = prepared.center_x[compact], prepared.center_y[compact]
        r = prepared.radius_px[compact]
        x0, x1 = cx - r, cx + r
        y0, y1 = cy - r, cy + r
        expected_tiles = set()
        for tile_index in range(scene.tile_count):
            tx0, ty0 = tile_origin(scene, tile_index)
            tx1, ty1 = tx0 + scene.tile_width, ty0 + scene.tile_height
            if x1 >= tx0 and x0 < tx1 and y1 >= ty0 and y0 < ty1:
                expected_tiles.add(tile_index)
        self.assertTrue(expected_tiles)

        # Edit only that point's gray value in the cloud file.
        cloud = load_point_cloud(self.ws.path("cloud.pts3"))
        cloud.grays[point_global_index] = float(np.clip(cloud.grays[point_global_index] + 0.3, 0, 1))
        save_point_cloud(self.ws.path("cloud.pts3"), cloud)

        result = self._render()
        self.assertEqual(result["status"], "completed")
        invalidated = set(result["report"]["tiles_invalidated"])
        rerendered = set(result["report"]["tiles_rendered"])
        self.assertEqual(invalidated, expected_tiles)
        self.assertEqual(rerendered, expected_tiles)
        # Every other tile is still reusable.
        reused = set(result["report"]["tiles_reused"])
        self.assertEqual(reused, set(range(scene.tile_count)) - expected_tiles)

    def test_tampered_tile_detected_and_repaired_with_e_digest_path(self):
        self._render()
        scene = load_scene(self.scene_path)
        version_id = scene.version_id(scene.cloud)
        store = TileStore(scene, version_id)

        # Corrupt a persisted tile's depth buffer but keep the file readable.
        target_tile = 2
        record = store.load_tile(target_tile)
        self.assertIsNotNone(record)
        corrupted = record.depth.copy()
        corrupted[0, 0] = 999.0
        path = store.path_for(target_tile)
        np.savez(
            path,
            gray=record.gray,
            depth=corrupted,
            version_id=np.asarray(record.version_id),
            input_digest=np.asarray(record.input_digest),
            block_digest=np.asarray(record.block_digest),
            declared_hits=np.asarray(record.declared_hits, dtype=np.int64),
            tile_index=np.asarray(target_tile, dtype=np.int64),
            shape=np.asarray(record.gray.shape, dtype=np.int64),
        )

        result = self._render()
        # Recovery detects the stale digest, archives and re-renders the tile.
        self.assertEqual(result["status"], "completed")
        self.assertIn(target_tile, result["report"]["tiles_invalidated"])
        self.assertIn(target_tile, result["report"]["tiles_rerendered"])
        fixed = store.load_tile(target_tile)
        expected = compute_input_digest(scene, prepare_points(scene), target_tile)
        ok, _ = store.verify_record(fixed, expected)
        self.assertTrue(ok)

    def test_declared_pixel_count_mismatch_detected(self):
        self._render()
        scene = load_scene(self.scene_path)
        store = TileStore(scene, scene.version_id(scene.cloud))
        target_tile = 1
        record = store.load_tile(target_tile)
        path = store.path_for(target_tile)
        np.savez(
            path,
            gray=record.gray,
            depth=record.depth,
            version_id=np.asarray(record.version_id),
            input_digest=np.asarray(record.input_digest),
            block_digest=np.asarray(record.block_digest),
            declared_hits=np.asarray(record.declared_hits + 7, dtype=np.int64),
            tile_index=np.asarray(target_tile, dtype=np.int64),
            shape=np.asarray(record.gray.shape, dtype=np.int64),
        )
        expected = compute_input_digest(scene, prepare_points(scene), target_tile)
        tampered = store.load_tile(target_tile)
        ok, reason = store.verify_record(tampered, expected)
        self.assertFalse(ok)
        self.assertIn("pixel count", reason)

    def test_npz_temp_files_cleaned_after_hard_kill(self):
        self._render()
        decoy = self.ws.path("state", "tile_00003.npz.tmp.npz")
        with open(decoy, "wb") as handle:
            handle.write(b"\x93NUMPY partial")
        self.assertTrue(os.path.exists(decoy))
        result = self._render()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(os.path.exists(decoy))

    def test_atomic_temp_files_never_confused_as_complete(self):
        self._render()
        # A leftover temp file from a hard kill must be cleaned, not loaded.
        decoy = self.ws.path("state", ".tmp_deadbeef.tmp")
        with open(decoy, "wb") as handle:
            handle.write(b"partial")
        self.assertTrue(os.path.exists(decoy))
        result = self._render()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(os.path.exists(decoy))


if __name__ == "__main__":
    unittest.main()
