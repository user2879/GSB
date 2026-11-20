from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pointcloud_studio.engine import StudioEngine
from pointcloud_studio.errors import ErrorCode, StudioError
from pointcloud_studio.mesh import ReconstructionParams, decode_patch_cache
from pointcloud_studio.persistence import AtomicStateStore
from pointcloud_studio.sample import load_sample_bundle, write_sample_bundle


class PersistenceAndRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="pcstudio-persistence-tests-")
        cls.root = Path(cls.temporary.name)
        write_sample_bundle(cls.root / "sample")
        points, normals, confidence, _ = load_sample_bundle(cls.root / "sample")
        cls.engine = StudioEngine(points, normals, confidence, params=ReconstructionParams())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_commit_clean_reopen_and_cache_hash(self) -> None:
        root = self.root / "workspace"
        store = AtomicStateStore(root)
        store.open()
        commit = store.commit(self.engine)
        self.assertEqual(commit["version"], 1)
        store.close(self.engine)
        reopened = AtomicStateStore(root)
        report = reopened.open()
        self.assertTrue(report.clean_previous_exit)
        loaded = reopened.load_current()
        self.assertEqual(loaded.version, self.engine.version)
        self.assertEqual(loaded.engine.mesh.content_sha256, self.engine.mesh.content_sha256)
        self.assertTrue(reopened.independent_verify_current()["ok"])
        first_patch = next(iter(self.engine.patches.values()))
        payload = bytearray(first_patch.cache_payload())
        payload[40] ^= 1
        with self.assertRaises(StudioError) as caught:
            decode_patch_cache(bytes(payload), expected_content_hash=first_patch.content_sha256, path="tampered")
        self.assertEqual(caught.exception.code, ErrorCode.E_CACHE_HASH)

    def test_forced_termination_recovers_committed_state_and_orphans(self) -> None:
        source = self.root / "crash_source"
        store = AtomicStateStore(source)
        store.open()
        store.commit(self.engine)
        store.close(self.engine)
        crash_workspace = self.root / "crash_recovery"
        if crash_workspace.exists():
            shutil.rmtree(crash_workspace)
        crash_workspace.mkdir()
        current = AtomicStateStore(source)._read_current()
        (crash_workspace / "states").mkdir()
        shutil.copy2(source / "CURRENT", crash_workspace / "CURRENT")
        shutil.copytree(source / "states" / current["state"], crash_workspace / "states" / current["state"])
        shutil.copy2(source / "closed.json", crash_workspace / "closed.json")
        script = Path(__file__).resolve().parents[1] / "scripts" / "crash_child.py"
        child = subprocess.run(
            [sys.executable, str(script), str(crash_workspace)],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(child.returncode, 23)
        recovered_store = AtomicStateStore(crash_workspace)
        report = recovered_store.open()
        self.assertFalse(report.clean_previous_exit)
        self.assertGreaterEqual(len(report.recovered_orphans), 1)
        recovered = recovered_store.load_current()
        self.assertEqual(recovered.version, self.engine.version)
        self.assertTrue(recovered_store.independent_verify_current()["ok"])
        self.assertFalse((crash_workspace / "closed.json").exists())


if __name__ == "__main__":
    unittest.main()
