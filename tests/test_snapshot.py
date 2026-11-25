from __future__ import annotations

import tempfile
import unittest
import shutil
from pathlib import Path

from evidence_diag.engine import (
    DiagnosticEngine,
    config_fingerprint,
    evidence_library_fingerprint,
    run_sequence,
)
from evidence_diag.snapshot import SnapshotStore, VersionConflict
from evidence_diag.util import sha256_json

from .helpers import sample_objects


def prefix_hash(commands: list[dict], cursor: int) -> str:
    return sha256_json(commands[:cursor])


class SnapshotTests(unittest.TestCase):
    def test_atomic_snapshot_resume_and_version_conflict(self) -> None:
        config, session, commands = sample_objects()
        expected = run_sequence(config, session, commands, mode="incremental")
        with tempfile.TemporaryDirectory() as temp:
            store = SnapshotStore(Path(temp) / "snapshots")
            engine = DiagnosticEngine(config, session)
            base_config_hash = config_fingerprint(config)
            base_evidence_fp = evidence_library_fingerprint(session)
            parent = 0
            computed = engine.compute()
            manifest = store.commit(
                state=engine.export_state(),
                result=computed.result,
                execution=computed.execution,
                expected_parent_version=parent,
                base_config_hash=base_config_hash,
                base_evidence_fingerprint=base_evidence_fp,
                command_prefix_hash=prefix_hash(commands, 0),
            )
            parent = manifest["global_version"]
            for command in commands[:6]:
                engine.apply_command(command)
                computed = engine.compute()
                manifest = store.commit(
                    state=engine.export_state(),
                    result=computed.result,
                    execution=computed.execution,
                    expected_parent_version=parent,
                    base_config_hash=base_config_hash,
                    base_evidence_fingerprint=base_evidence_fp,
                    command_prefix_hash=prefix_hash(
                        commands, engine.command_cursor
                    ),
                )
                parent = manifest["global_version"]
            bundle = store.load_current()
            self.assertIsNotNone(bundle)
            assert bundle is not None
            self.assertEqual(bundle.result, expected["trace"][6]["result"])
            decision = store.recover(
                base_config_hash=base_config_hash,
                base_evidence_fingerprint=base_evidence_fp,
                command_prefix_hash=prefix_hash(commands, 6),
            )
            self.assertEqual(decision["decision"], "resume")
            bad_decision = store.recover(
                base_config_hash="0" * 64,
                base_evidence_fingerprint=base_evidence_fp,
                command_prefix_hash=prefix_hash(commands, 6),
            )
            self.assertEqual(bad_decision["decision"], "full_recompute")

            orphan = store.root / ".tmp-simulated-crash"
            orphan.mkdir()
            (orphan / "partial.json").write_text("{", encoding="utf-8")
            self.assertEqual(store.load_current().manifest["global_version"], parent)
            self.assertEqual(store.cleanup_temp(), 1)

            published_orphan = store.versions_dir / "v99999999"
            shutil.copytree(bundle.directory, published_orphan)
            self.assertEqual(
                store.load_latest_coherent().manifest["global_version"], parent
            )

            resumed_engine = bundle.engine()
            for command in commands[6:]:
                resumed_engine.apply_command(command)
                resumed_engine.compute()
            self.assertEqual(
                resumed_engine.last_result,
                expected["final_result"],
            )
            with self.assertRaises(VersionConflict):
                store.commit(
                    state=resumed_engine.export_state(),
                    result=resumed_engine.last_result,
                    execution={},
                    expected_parent_version=0,
                    base_config_hash=base_config_hash,
                    base_evidence_fingerprint=base_evidence_fp,
                    command_prefix_hash=prefix_hash(commands, 17),
                )
            store.current_path.write_text("{corrupt", encoding="utf-8")
            no_commit = store.recover(
                base_config_hash=base_config_hash,
                base_evidence_fingerprint=base_evidence_fp,
                command_prefix_hash=prefix_hash(commands, 6),
            )
            self.assertEqual(no_commit["decision"], "full_recompute")
            self.assertEqual(no_commit["reason"], "no_coherent_snapshot")


if __name__ == "__main__":
    unittest.main()
