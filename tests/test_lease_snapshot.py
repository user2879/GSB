from __future__ import annotations

import time
from pathlib import Path

import pytest

from conftest import EXAMPLES
from ins_diag.config import LeaseDeclaration, load_config
from ins_diag.errors import LeaseExpiredError
from ins_diag.lease import LeaseManager, MonotonicClock
from ins_diag.pipeline import run_pipeline
from ins_diag.snapshot import SnapshotStore


class FakeClock:
    def __init__(self, start: int = 1000) -> None:
        self.value = start

    def now_ms(self) -> int:
        return self.value

    def advance(self, milliseconds: int) -> None:
        self.value += milliseconds


class FastAdvancingClock:
    def __init__(self) -> None:
        self.value = 0

    def now_ms(self) -> int:
        self.value += 1000
        return self.value


def test_lease_renewal_and_expiry_invalidate_artifacts(tmp_path: Path) -> None:
    clock = FakeClock()
    manager = LeaseManager(
        LeaseDeclaration("test-worker", 10, 0),
        tmp_path,
        clock=clock,
    )
    issued = manager.issue(0)
    assert issued.generation == 1
    assert issued.expires_ms == 1010
    renewed = manager.renew(issued.lease_id)
    assert renewed.generation == 2
    assert renewed.supersedes == issued.lease_id
    assert manager.artifact_is_valid(renewed.lease_id, renewed.generation)
    clock.advance(10)
    expired = manager.expire_due()
    assert any(item["lease_id"] == renewed.lease_id for item in expired)
    assert not manager.artifact_is_valid(renewed.lease_id, renewed.generation)
    with pytest.raises(LeaseExpiredError):
        manager.renew(renewed.lease_id)


def test_lease_renewal_after_expiry_uses_real_monotonic_clock(tmp_path: Path) -> None:
    manager = LeaseManager(
        LeaseDeclaration("real-clock", 5, 0),
        tmp_path,
        clock=MonotonicClock(),
    )
    issued = manager.issue(0)
    time.sleep(0.012)
    expired = manager.expire_due()
    assert any(item["lease_id"] == issued.lease_id for item in expired)
    assert not manager.artifact_is_valid(issued.lease_id, issued.generation)


def test_commit_cannot_revive_an_expired_lease(tmp_path: Path) -> None:
    clock = FakeClock()
    manager = LeaseManager(LeaseDeclaration("commit-clock", 5, 0), tmp_path, clock=clock)
    issued = manager.issue(0)
    clock.advance(5)
    with pytest.raises(LeaseExpiredError):
        manager.commit([issued.lease_id])
    assert manager.get(issued.lease_id).state == "expired"


def test_pipeline_invalidates_clusters_density_and_snapshot_on_lease_expiry(
    tmp_path: Path,
) -> None:
    report = run_pipeline(
        EXAMPLES / "config.yaml",
        state_dir=tmp_path / "state",
        clock=FastAdvancingClock(),
    )
    assert report["status"] == "failed"
    assert "E_LEASE_EXPIRED" in report["error_codes"]
    assert report["clusters"]["invalidated"] is True
    assert report["density"]["invalidated"] is True
    assert report["snapshot"]["commit"]["committed"] is False
    assert all(window["cluster_id"] is None for window in report["windows"])
    assert all(window["density"] is None for window in report["windows"])


def test_snapshot_checksum_and_four_way_restore_validation(tmp_path: Path) -> None:
    state = tmp_path / "state"
    first = run_pipeline(EXAMPLES / "config.yaml", state_dir=state)
    commit = first["snapshot"]["commit"]
    assert commit["committed"] is True
    store = SnapshotStore(state)
    snapshot = store.read()
    assert snapshot is not None
    assert store.checksum_valid(snapshot)
    assert len(snapshot["payload"]["windows"]) == 43
    assert snapshot["payload"]["windows"][0]["standardized_fixed_units"] is not None
    assert snapshot["payload"]["windows"][0]["density"] is not None

    validation = store.evaluate_restore(
        expected_global_version=commit["version"],
        expected_config_hash=first["config"]["hash"],
        expected_data_fingerprint=first["data"]["fingerprint"],
        expected_lease_generation=commit["lease_generation"],
        expected_lease_map=commit["lease_map"],
    )
    assert validation.restoreable is True
    assert validation.errors == []

    stale_cases = [
        (
            commit["version"] + 1,
            first["config"]["hash"],
            first["data"]["fingerprint"],
            commit["lease_generation"],
            "snapshot.global_version",
        ),
        (
            commit["version"],
            "different-config",
            first["data"]["fingerprint"],
            commit["lease_generation"],
            "snapshot.config_hash",
        ),
        (
            commit["version"],
            first["config"]["hash"],
            "different-data",
            commit["lease_generation"],
            "snapshot.data_fingerprint",
        ),
        (
            commit["version"],
            first["config"]["hash"],
            first["data"]["fingerprint"],
            commit["lease_generation"] + 1,
            "snapshot.lease_generation",
        ),
    ]
    for version, config_hash, data_fingerprint, lease_generation, expected_path in stale_cases:
        stale = store.evaluate_restore(
            expected_global_version=version,
            expected_config_hash=config_hash,
            expected_data_fingerprint=data_fingerprint,
            expected_lease_generation=lease_generation,
        )
        assert stale.restoreable is False
        assert any(
            error.code == "E_SNAPSHOT_STALE" and error.path == expected_path
            for error in stale.errors
        )


def test_snapshot_commit_audits_old_snapshot_and_advances_version(tmp_path: Path) -> None:
    state = tmp_path / "state"
    first = run_pipeline(EXAMPLES / "config.yaml", state_dir=state)
    second = run_pipeline(EXAMPLES / "config.yaml", state_dir=state)
    assert first["snapshot"]["commit"]["version"] == 1
    assert second["snapshot"]["commit"]["version"] == 2
    supersession = second["snapshot"]["supersession_validation"]
    assert supersession["snapshot_present"] is True
    stale_paths = {error["path"] for error in supersession["errors"]}
    assert "snapshot.lease_generation" in stale_paths
    assert second["snapshot"]["used_old_snapshot_for_analysis"] is False
