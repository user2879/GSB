from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qosfdiag.engine import CascadeEngine
from qosfdiag.error_propagation import propagate_fixed_errors
from qosfdiag.errors import ReplayError
from qosfdiag.generator import DeterministicUdpGenerator
from qosfdiag.ledger import DecisionLedger
from qosfdiag.policy import load_policy
from qosfdiag.replay import verify_snapshot_replay
from qosfdiag.samples import (
    single_stage_generator,
    single_stage_policy,
    three_stage_generator,
    three_stage_policy,
)
from qosfdiag.snapshot import (
    REPLACED_FIELDS,
    SnapshotDocument,
    SnapshotManager,
    snapshot_content_digest,
)
from qosfdiag.transport import InMemoryTransport


def _save_sample_snapshot(directory: Path, *, single: bool = False):
    policy_dict = single_stage_policy() if single else three_stage_policy()
    generator_dict = single_stage_generator() if single else three_stage_generator()
    policy = load_policy(policy_dict)
    generator = DeterministicUdpGenerator(generator_dict)
    ledger_path = directory / ("single.sqlite" if single else "three.sqlite")
    manager = SnapshotManager(directory / "snapshots")
    with DecisionLedger(ledger_path) as ledger:
        with CascadeEngine(
            policy,
            generator,
            scale=3,
            transport=InMemoryTransport([s.shaper_id for s in policy.shapers]),
        ) as engine:
            run = engine.run(ledger=ledger)
        errors = propagate_fixed_errors(
            policy,
            run.summary,
            run.events,
            scale=3,
            generator_digest=generator.sequence_digest,
        )
        document = manager.save(
            filename="single.json" if single else "three.json",
            policy=policy,
            scale=3,
            generator=generator,
            ledger=ledger,
            errors=errors,
        )
    return manager, document


def test_snapshot_match_and_replay(tmp_path: Path) -> None:
    manager, document = _save_sample_snapshot(tmp_path)
    document.verify()
    listed = manager.list_snapshots()
    assert [item["filename"] for item in listed] == ["three.json"]
    current = document.descriptor()
    matched = manager.open(document.path, current=current)
    assert matched.applied is False
    assert matched.replaced_fields == ()
    result = verify_snapshot_replay(
        document,
        current_policy_version="three-stage-v1",
        current_scale=3,
        current_generator_digest=document.payload["packet_sequence_digest"],
    )
    assert result.ok is True
    assert result.events_compared == 184


def test_whole_replacement_and_read_only_backup(tmp_path: Path) -> None:
    manager, single_document = _save_sample_snapshot(tmp_path, single=True)
    _, three_document = _save_sample_snapshot(tmp_path)
    current = single_document.descriptor()
    current["snapshot_path"] = str(single_document.path)
    result = manager.open(three_document.path, current=current)
    assert result.applied is True
    assert result.version_matches is False
    assert result.content_matches is False
    assert result.replaced_fields == REPLACED_FIELDS
    assert result.backup_path is not None and result.backup_path.exists()
    assert result.backup_path.stat().st_mode & 0o222 == 0
    backup = SnapshotDocument.load(result.backup_path)
    assert backup.content_digest == single_document.content_digest
    assert set(result.replacement) == {
        "policy_set",
        "shaper_set",
        "cascade_graph",
        "scale",
        "ledger",
        "error_conclusions",
    }


def test_snapshot_materializes_replacement_ledger(tmp_path: Path) -> None:
    _, single_document = _save_sample_snapshot(tmp_path, single=True)
    _, three_document = _save_sample_snapshot(tmp_path)
    target = tmp_path / "replaced.sqlite"
    single_document.restore_ledger(target, overwrite=True)
    with DecisionLedger(target) as ledger:
        ledger.initialize(
            policy=single_document.policy,
            scale=single_document.scale,
            generator=DeterministicUdpGenerator(single_document.generator_config),
        )
        assert ledger.progress().cursor_seq == single_document.ledger["cursor_seq"]
        assert len(ledger.decisions()) == len(single_document.ledger["decisions"])

    three_document.restore_ledger(target, overwrite=True)
    with DecisionLedger(target) as ledger:
        ledger.initialize(
            policy=three_document.policy,
            scale=three_document.scale,
            generator=DeterministicUdpGenerator(three_document.generator_config),
        )
        assert ledger.progress().cursor_seq == three_document.ledger["cursor_seq"]
        assert ledger.progress().content_digest == three_document.ledger["content_digest"]
        assert len(ledger.decisions()) == len(three_document.ledger["decisions"])
        assert len(ledger.decisions()) == 184


def test_replay_localizes_first_tampered_field(tmp_path: Path) -> None:
    _, document = _save_sample_snapshot(tmp_path)
    payload = json.loads(json.dumps(document.payload))
    event = payload["ledger"]["decisions"][0]
    packet_id = event["packet_id"]
    shaper_id = event["stage_id"]
    event["tokens_after"] += 1
    payload["content_digest"] = snapshot_content_digest(payload)
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(payload), encoding="utf-8")
    tampered = SnapshotDocument.load(tampered_path)
    with pytest.raises(ReplayError) as caught:
        verify_snapshot_replay(tampered)
    error = caught.value
    assert error.code == "E_REPLAY"
    assert error.details["classification"] == "LEDGER_DEFECT"
    assert error.mismatch is not None
    assert error.mismatch.packet_id == packet_id
    assert error.mismatch.shaper_id == shaper_id
    assert error.mismatch.field == "tokens_after"


def test_version_mismatch_is_unreproducible_not_defect(tmp_path: Path) -> None:
    _, document = _save_sample_snapshot(tmp_path)
    with pytest.raises(ReplayError) as caught:
        verify_snapshot_replay(document, current_policy_version="different")
    error = caught.value
    assert error.reason == "policy_version_mismatch"
    assert error.details["classification"] == "UNREPRODUCIBLE"
    assert error.mismatch is None
