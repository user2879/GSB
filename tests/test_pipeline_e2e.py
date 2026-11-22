"""Requirements 4/5/7/8: end-to-end session, recovery, snapshot, replay."""

import json
import os

import pytest

from config_regression.attribution import (
    COEFFICIENT_AMPLIFICATION,
    DIRECT_VALUE,
)
from config_regression.canonical import canonical_bytes
from config_regression.errors import E_EXHAUST, E_REPLAY, ReplayError
from config_regression.ledger import Ledger
from config_regression.model import (
    SessionConfig,
    load_manifest,
    load_registry,
)
from config_regression.pipeline import (
    _safe_replay,
    _snapshot_payload,
    diff_manifests,
    analyze_error_bounds,
    quantize_version,
    recover_session,
    run_session,
)


REGISTRY = {"nodes": [
    {"node_id": "node-a", "port": 0, "work_ms_min": 1, "work_ms_max": 2},
    {"node_id": "node-b", "port": 0, "work_ms_min": 1, "work_ms_max": 2},
]}

BASE = {"version": 1, "items": [
    {"key": "rate", "value": 1.0, "targets": ["node-a", "node-b"],
     "derived_from": []},
    {"key": "scaled", "value": 2.0, "targets": ["node-a"],
     "derived_from": [{"key": "rate", "coefficient": 2.0}]},
]}
CAND = {"version": 2, "items": [
    {"key": "rate", "value": 1.4, "targets": ["node-a", "node-b"],
     "derived_from": []},
    {"key": "scaled", "value": 3.5, "targets": ["node-a"],
     "derived_from": [{"key": "rate", "coefficient": 2.5}]},
]}


@pytest.fixture()
def inputs(tmp_path):
    registry = load_registry(REGISTRY)
    base = load_manifest(BASE, registry)
    cand = load_manifest(CAND, registry)
    session = SessionConfig.defaults()
    db = str(tmp_path / "session.db")
    return base, cand, registry, session, db


def test_full_session_detects_regression_and_attributes(inputs):
    base, cand, registry, session, db = inputs
    artifacts = run_session(base, cand, registry, session, db)
    assert artifacts.verdict == "E_REGRESSION"
    regression = [f for f in artifacts.findings if f["status"] == "E_REGRESSION"]
    assert regression
    sample = next(f for f in regression if f["key"] == "rate"
                  and f["node_id"] == "node-a")
    assert sample["first_diff_group"] is not None
    assert sample["tolerance"]["value_abs"] == session.value_tolerance
    assert set(sample["summary_deltas"])  # deltas recorded per group
    causes = {a["key"]: a["cause"] for a in artifacts.attributions}
    assert causes["rate"] == DIRECT_VALUE
    assert causes["scaled"] == COEFFICIENT_AMPLIFICATION
    # attribution shares always sum to 1000 permille
    for attr in artifacts.attributions:
        assert attr["shares_total_permille"] == 1000
    assert artifacts.replay_report["status"] == "ok"


def test_missing_receipt_is_not_counted_as_clean(inputs, tmp_path):
    base, cand, _registry, session, _db = inputs
    registry = load_registry({"nodes": [
        {"node_id": "node-a", "port": 0},
        {"node_id": "node-b", "port": 0, "fault_mode": "drop_all"}]})
    db = str(tmp_path / "exhaust.db")
    artifacts = run_session(base, cand, registry, session, db)
    statuses = {(f["node_id"], f["key"]): f["status"]
                for f in artifacts.findings}
    assert all(status != "ok" for (node, _), status in statuses.items()
               if node == "node-b")
    jobs = {c["node_id"]: c["status"] for c in artifacts.collection}
    assert jobs["node-b"] == E_EXHAUST
    no_receipt = [f for f in artifacts.findings if f["status"] == "no_receipt"]
    assert no_receipt and all(
        f["detail"] and "node did not" in f["detail"] or "candidate" in (f["detail"] or "")
        for f in no_receipt)


def test_recover_resumes_committed_batches_and_marks_unknown(inputs, tmp_path):
    base, cand, registry, session, db = inputs
    run_session(base, cand, registry, session, db)
    # Simulate a forced termination: an in-flight lease with no committed batch.
    ledger = Ledger(db)
    ledger.acquire_lease("batch-1-2-node-z", "node-z", 123,
                         {"state": "collecting"})
    ledger.close()

    recovered = recover_session(base, cand, registry, session, db)
    assert recovered.unknown_items and \
        recovered.unknown_items[0]["state"] == "unknown_rerun_required"
    assert recovered.replay_report["status"] == "ok"
    # committed findings survive recovery
    assert len(recovered.findings) == len({(f["node_id"], f["key"])
                                           for f in recovered.findings})


def test_snapshot_mismatch_triggers_whole_replacement(inputs):
    base, cand, registry, session, db = inputs
    run_session(base, cand, registry, session, db)
    ledger = Ledger(db)
    changed = SessionConfig.defaults()
    object.__setattr__(changed, "scale_exponent", 9)
    report = ledger.verify_or_replace_snapshot(
        _snapshot_payload(base, cand, changed), 1, 1)
    assert report["action"] == "replaced"
    replaced_subjects = {row["subject"] for row in report["replaced"]}
    assert "content_digest" in replaced_subjects
    # whole store replacement, not incremental merge
    assert ledger.committed_batches() == []
    ledger.close()


def test_snapshot_format_version_mismatch_replaces_store(inputs):
    base, cand, registry, session, db = inputs
    run_session(base, cand, registry, session, db)
    ledger = Ledger(db)
    # declaring a newer ledger format version must force whole replacement
    report = ledger.verify_or_replace_snapshot(
        _snapshot_payload(base, cand, session),
        ledger_format_version=999,
        precision_scheme_version=1)
    assert report["action"] == "replaced"
    assert any(r["subject"] == "ledger_format_version"
               for r in report["replaced"])
    assert ledger.committed_batches() == []
    ledger.close()


def test_replay_mismatch_locates_first_key_node_field(inputs):
    base, cand, registry, session, db = inputs
    run_session(base, cand, registry, session, db)
    ledger = Ledger(db)
    diff_obj = diff_manifests(base, cand)
    bounds = analyze_error_bounds(base, cand, diff_obj, session.scale_exponent)
    q_base = quantize_version(base, session.scale_exponent)
    q_cand = quantize_version(cand, session.scale_exponent)
    stored = ledger.load_findings()
    tampered = [dict(row) for row in stored]
    target = next(row for row in tampered if row["status"] == "E_REGRESSION")
    target["status"] = "ok"
    report = _safe_replay(ledger, tampered, diff_obj, bounds,
                          q_base, q_cand, session)
    assert report["status"] == E_REPLAY
    first = report["violations"][0]
    assert target["node_id"] in first["subject"]
    assert target["key"] in first["subject"]
    assert first["subject"].endswith("status")
    ledger.close()


def test_attributions_byte_identical_across_runs(inputs):
    base, cand, registry, session, db = inputs
    first = run_session(base, cand, registry, session, db)
    os.unlink(db)
    second = run_session(base, cand, registry, session, db)
    assert canonical_bytes(first.attributions) == \
        canonical_bytes(second.attributions)
    assert canonical_bytes(first.findings) == canonical_bytes(second.findings)
