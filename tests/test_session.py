from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from snapdiag.bitmap import get_bit, popcount
from snapdiag.errors import DiagnosticError, E_CHAIN, E_EXISTS, E_FORMAT, E_INTEGRITY, E_STATE, E_VERSION
from snapdiag.injection import InjectionManager
from snapdiag.manifest import build_manifest_model, encode_manifest_text
from snapdiag.store import Session, SimulatedInterruption
from snapdiag.util import sha256_file
from snapdiag.workflow import execute_steps


def _write_fill(session: Session, block: int, value: int) -> None:
    session.write_block(block, bytes([value]) * session.block_size)


def _create_path_session(path: Path) -> Session:
    session = Session.initialize(path, block_size=512, block_count=16, threshold=0.5, reset=True)
    execute_steps(
        session,
        [
            {"action": "snapshot", "snapshot_id": "s0", "parent_snapshot_id": None},
            {"action": "write", "block": 0, "fill_hex": "11"},
            {"action": "write", "block": 5, "fill_hex": "12"},
            {"action": "snapshot", "snapshot_id": "s1", "parent_snapshot_id": "s0"},
            {"action": "write", "block": 1, "fill_hex": "21"},
            {"action": "write", "block": 2, "fill_hex": "22"},
            {"action": "write", "block": 3, "fill_hex": "23"},
            {"action": "write", "block": 4, "fill_hex": "24"},
            {"action": "write", "block": 6, "fill_hex": "25"},
            {"action": "write", "block": 7, "fill_hex": "26"},
        ],
    )
    return session


def test_duplicate_snapshot_and_broken_parent(session: Session) -> None:
    session.create_snapshot("s0")
    with pytest.raises(DiagnosticError) as caught:
        session.create_snapshot("s0")
    assert caught.value.code == E_EXISTS
    with pytest.raises(DiagnosticError) as caught:
        session.create_snapshot("s1", parent_snapshot_id="missing")
    assert caught.value.code == E_CHAIN
    assert caught.value.snapshot_id == "s1"
    assert caught.value.details["missing_parent"] == "missing"


def test_cow_bitmap_and_reference_are_committed_together(session: Session) -> None:
    session.create_snapshot("s0")
    session.create_snapshot("s1", parent_snapshot_id="s0")
    old_digest = sha256_file(session.geometry().image_path)
    _write_fill(session, 3, 0x33)
    row = session.snapshot("s0")
    assert get_bit(bytes(row["bitmap"]), 3)
    assert row["snapshot_version"] == 2
    assert session.snapshot_blocks("s0")[3]["digest"] == session.snapshot_blocks("s0")[3]["ref"]
    assert session.snapshot_blocks("s1")[3]["ref"] == session.snapshot_blocks("s0")[3]["ref"]
    reopened = Session.open(session.root)
    assert popcount(bytes(reopened.snapshot("s0")["bitmap"])) == 1
    assert sha256_file(reopened.geometry().image_path) != old_digest


def test_manifest_replacement_is_whole_and_lists_stable_diff(tmp_path: Path) -> None:
    session = Session.initialize(tmp_path / "s", block_size=512, block_count=12, threshold=0.25)
    execute_steps(
        session,
        [
            {"action": "snapshot", "snapshot_id": "m0", "parent_snapshot_id": None},
            {"action": "write", "block": 1, "fill_hex": "31"},
            {"action": "snapshot", "snapshot_id": "m1", "parent_snapshot_id": "m0"},
            {"action": "write", "block": 2, "fill_hex": "41"},
            {"action": "snapshot", "snapshot_id": "m2", "parent_snapshot_id": "m1"},
        ],
    )
    model = session.build_manifest_model()
    full_path = tmp_path / "full.m3.json"
    full_path.write_text(encode_manifest_text(model, "m3"), encoding="utf-8")
    subset = build_manifest_model(
        block_size=model.block_size,
        block_count=model.block_count,
        snapshots=[item for item in model.snapshots if item.snapshot_id != "m2"],
        source_format="m3",
    )
    subset_path = tmp_path / "subset.m3.json"
    subset_path.write_text(encode_manifest_text(subset, "m3"), encoding="utf-8")
    report = session.import_manifest(subset_path, reason="test-remove")
    assert report["removed"] == ["m2"]
    assert [item["snapshot_id"] for item in session.snapshot_rows()] == ["m0", "m1"]
    restore = session.import_manifest(full_path, reason="test-restore")
    assert restore["added"] == ["m2"]
    replaced_model = build_manifest_model(
        block_size=model.block_size,
        block_count=model.block_count,
        snapshots=[replace(item, snapshot_version=item.snapshot_version + 5) for item in model.snapshots],
        source_format="m3",
    )
    replaced_path = tmp_path / "replaced.m3.json"
    replaced_path.write_text(encode_manifest_text(replaced_model, "m3"), encoding="utf-8")
    replaced = session.import_manifest(replaced_path, reason="test-replace")
    assert replaced["replaced"] == ["m0", "m1", "m2"]
    backups = session.list_manifest_backups()
    assert len(backups) == 3
    assert all(item["readonly"] for item in backups)


def test_threshold_boundary_uses_full_path(tmp_path: Path) -> None:
    session = _create_path_session(tmp_path / "boundary")
    boundary = session.plan_rollback("s0", path="auto")
    assert boundary.ratio == "1/2"
    assert boundary.threshold == "1/2"
    assert boundary.boundary_hit is True
    assert boundary.selected_path == "full"
    low = session.plan_rollback("s1", path="auto")
    assert low.ratio == "3/8"
    assert low.selected_path == "bitmap"


def test_bitmap_and_full_rollback_are_bitwise_identical(tmp_path: Path) -> None:
    bitmap_session = _create_path_session(tmp_path / "bitmap")
    full_session = _create_path_session(tmp_path / "full")
    bitmap_result = bitmap_session.rollback("s0", path="bitmap")
    full_result = full_session.rollback("s0", path="full")
    assert bitmap_session.geometry().read_all() == full_session.geometry().read_all()
    assert bitmap_result.result_sha256 == full_result.result_sha256
    assert bitmap_result.restored_blocks == full_result.restored_blocks
    assert bitmap_result.verification["ok"] and full_result.verification["ok"]
    assert bitmap_result.verification["checked_blocks"] == 16


def test_integrity_failure_lists_first_block_and_expected_actual(tmp_path: Path) -> None:
    session = _create_path_session(tmp_path / "integrity")
    manager = InjectionManager(session)
    injection = manager.corrupt_block_data(target="s0", block_index=0)
    with pytest.raises(DiagnosticError) as caught:
        session.rollback("s0", path="bitmap")
    error = caught.value
    assert error.code == E_INTEGRITY
    assert error.snapshot_id == "s0"
    assert error.details["block_index"] == 0
    assert error.details["expected_digest"] == injection["expected_digest"]
    assert error.details["actual_digest"] != error.details["expected_digest"]


def test_interrupted_rollback_requires_retry(tmp_path: Path) -> None:
    session = _create_path_session(tmp_path / "interrupt")
    with pytest.raises(SimulatedInterruption):
        session.rollback("s0", path="bitmap", failpoint="after_prepare")
    recovered = session.recover_session()
    assert recovered[0]["status"] == "incomplete_unknown"
    assert recovered[0]["requires_rerun"] is True
    with pytest.raises(DiagnosticError) as caught:
        session.rollback("s0", path="bitmap")
    assert caught.value.code == E_STATE
    result = session.rollback("s0", path="bitmap", retry=True)
    assert result.verification["ok"]


def test_missing_recovery_marker_is_unknown_until_retry(tmp_path: Path) -> None:
    session = _create_path_session(tmp_path / "missing-marker")
    with pytest.raises(SimulatedInterruption):
        session.rollback("s0", path="bitmap", failpoint="after_prepare")
    operation = session.operation_rows()[0]
    marker = session.tmp_dir / f"{operation['operation_id']}.json"
    marker.unlink()
    recovered = session.recover_session()
    assert recovered[0]["status"] == "unknown"
    assert recovered[0]["outcome"] == "temporary_marker_missing_or_inconsistent"
    assert recovered[0]["requires_rerun"] is True
    result = session.rollback("s0", path="bitmap", retry=True)
    assert result.verification["ok"]


def test_audit_conclusions_are_versioned_and_stale_detected(tmp_path: Path) -> None:
    session = _create_path_session(tmp_path / "audit")
    first = session.audit()
    assert first["status"] == "pass"
    _write_fill(session, 15, 0x7F)
    with pytest.raises(DiagnosticError) as caught:
        session.latest_audit(strict=True)
    assert caught.value.code == E_VERSION
    second = session.audit()
    assert second["status"] == "pass"
    history = session.audit_history()
    assert len(history) == 2
    assert history[0]["audit_version"] == second["audit_version"]
    assert history[1]["audit_version"] == first["audit_version"]


def test_manifest_truncation_and_version_conflict_codes(tmp_path: Path) -> None:
    session = _create_path_session(tmp_path / "anomaly")
    manager = InjectionManager(session)
    manager.truncate_manifest()
    with pytest.raises(DiagnosticError) as caught:
        session.read_manifest()
    assert caught.value.code == E_FORMAT
    assert caught.value.field == "json"
    manager.restore_manifest()
    manager.inject_version_conflict()
    with pytest.raises(DiagnosticError) as caught:
        session.read_manifest()
    assert caught.value.code == E_VERSION
    assert caught.value.field == "format_version"
    manager.restore_manifest()
    assert session.manifest_consistency()["consistent"]
