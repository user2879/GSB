from __future__ import annotations

from pathlib import Path

from snapdiag.errors import E_FORMAT, E_INTEGRITY, E_VERSION
from snapdiag.injection import InjectionManager
from snapdiag.store import Session
from snapdiag.workflow import execute_steps


def _session(path: Path) -> Session:
    session = Session.initialize(path, block_size=512, block_count=16, threshold=0.25)
    execute_steps(
        session,
        [
            {"action": "snapshot", "snapshot_id": "s0", "parent_snapshot_id": None},
            {"action": "write", "block": 1, "fill_hex": "aa"},
            {"action": "snapshot", "snapshot_id": "s1", "parent_snapshot_id": "s0"},
            {"action": "write", "block": 2, "fill_hex": "bb"},
        ],
    )
    return session


def test_block_corruption_reports_and_repairs_every_block(tmp_path: Path) -> None:
    session = _session(tmp_path / "block")
    manager = InjectionManager(session)
    manager.corrupt_block_data(target="s0", block_index=1)
    diagnosis = manager.diagnose("block-corruption", target="s0")
    assert diagnosis["observed_code"] == E_INTEGRITY
    repair = manager.repair_and_verify(target="s0")
    assert repair["ok"]
    assert repair["block_verification"]["checked_blocks"] == session.block_count
    assert repair["block_verification"]["issue_count"] == 0


def test_truncation_and_version_injection_round_trip(tmp_path: Path) -> None:
    session = _session(tmp_path / "manifest")
    manager = InjectionManager(session)
    manager.truncate_manifest()
    assert manager.diagnose("manifest-truncation")["observed_code"] == E_FORMAT
    manager.restore_manifest()
    manager.inject_version_conflict()
    assert manager.diagnose("version-conflict")["observed_code"] == E_VERSION
    manager.restore_manifest()
    consistency = session.manifest_consistency()
    assert consistency["consistent"]
    assert consistency["file_source_format"] == "m3"
