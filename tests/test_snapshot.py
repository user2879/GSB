from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from inference_attribution.engine import execute_plan
from inference_attribution.errors import DiagnosticError
from inference_attribution.jsonio import canonical_json, sha256_canonical
from inference_attribution.ledger import Ledger
from inference_attribution.planner import build_plan
from inference_attribution.snapshot import (
    create_snapshot,
    load_snapshot,
    restore_snapshot,
)
from inference_attribution.spec import (
    load_merge_rules,
    load_requests,
    load_workers,
    parse_workers,
)
from helpers import SAMPLES, dynamic_workers


def _world(tmp_path: Path):
    workers_path = tmp_path / "workers.json"
    dynamic_workers(workers_path)
    workers = load_workers(workers_path)
    requests = load_requests(SAMPLES / "requests.json", workers)
    rules = load_merge_rules(SAMPLES / "merge_rules.json")
    plan = build_plan(workers, requests)
    return workers, requests, rules, plan


def _run_full(tmp_path: Path):
    workers, requests, rules, plan = _world(tmp_path)
    path = tmp_path / "full.sqlite"
    with Ledger(path) as ledger:
        outcome = execute_plan(
            plan=plan,
            request_set=requests,
            worker_set=workers,
            rules=rules,
            ledger=ledger,
        )
        assert outcome.progress_cursor == 3
        rows = ledger.rows()
    return workers, requests, rules, plan, path, rows


def test_snapshot_whole_replacement_and_replay_after_restore(tmp_path: Path):
    workers, requests, rules, plan, full_path, full_rows = _run_full(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    with Ledger(full_path) as ledger:
        create_snapshot(workers, ledger, snapshot_path)
    snapshot = load_snapshot(snapshot_path)
    assert snapshot["cursor"] == 3
    assert snapshot["ledger_digest"]

    updated_path = tmp_path / "workers-updated.json"
    dynamic_workers(
        updated_path,
        ports=(
            workers.by_id()["worker-alpha"].port,
            workers.by_id()["worker-beta"].port,
        ),
        worker_set_version=4,
        alpha_param_version=3,
    )
    updated_workers = load_workers(updated_path)
    restored_path = tmp_path / "restored.sqlite"
    outcome = restore_snapshot(snapshot_path, restored_path, updated_workers)
    assert outcome.whole_replacement is True
    assert "worker_set_version" in outcome.replacement_reasons
    assert "param_versions" in outcome.replacement_reasons
    with Ledger(restored_path) as restored:
        assert canonical_json(restored.rows()) == canonical_json(full_rows)
        assert restored.recovery_report().requires_rerun is False

    partial_snapshot = tmp_path / "partial-snapshot.json"
    partial_path = tmp_path / "partial.sqlite"
    with Ledger(partial_path) as partial:
        partial.apply_plan(plan)
        first = full_rows[0]
        batch = plan.by_id()[first["batch_id"]]
        partial.complete_batch(
            batch,
            receipt_digest=first["receipt_digest"],
            normalized_receipts=first["normalized_receipts"],
            invalidation_trace=first["invalidation_trace"],
        )
        create_snapshot(workers, partial, partial_snapshot)
    replay_target = tmp_path / "replay-target.sqlite"
    replay_outcome = restore_snapshot(partial_snapshot, replay_target, workers)
    assert len(replay_outcome.recovery.completed_batch_ids) == 1
    assert len(replay_outcome.recovery.unknown_batch_ids) == 2
    assert replay_outcome.recovery.requires_rerun is True
    with Ledger(replay_target) as replay_ledger:
        resumed = execute_plan(
            plan=plan,
            request_set=requests,
            worker_set=workers,
            rules=rules,
            ledger=replay_ledger,
        )
        assert resumed.progress_cursor == 3
        replayed_rows = replay_ledger.rows()
    from inference_attribution.invalidation import assert_replay_consistent

    assert_replay_consistent(plan, full_rows, replayed_rows)


def test_snapshot_digest_tampering_reports_e_snapshot(tmp_path: Path):
    workers, _, _, _, full_path, _ = _run_full(tmp_path)
    snapshot_path = tmp_path / "snapshot.json"
    with Ledger(full_path) as ledger:
        create_snapshot(workers, ledger, snapshot_path)
    original = json.loads(snapshot_path.read_text(encoding="utf-8"))

    tampered = copy.deepcopy(original)
    tampered["ledger_digest"] = "0" * 64
    body = {key: value for key, value in tampered.items() if key != "content_digest"}
    tampered["content_digest"] = sha256_canonical(body)
    tampered_path = tmp_path / "tampered-ledger.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DiagnosticError) as caught:
        load_snapshot(tampered_path)
    assert caught.value.code == "E_SNAPSHOT"
    assert caught.value.field == "ledger_digest"

    content_path = tmp_path / "tampered-content.json"
    content_bad = copy.deepcopy(original)
    content_bad["content_digest"] = "0" * 64
    content_path.write_text(json.dumps(content_bad), encoding="utf-8")
    with pytest.raises(DiagnosticError) as caught:
        load_snapshot(content_path)
    assert caught.value.code == "E_SNAPSHOT"
    assert caught.value.field == "content_digest"


def test_corrupt_worker_declaration_reports_e_spec(tmp_path: Path):
    document = json.loads((SAMPLES / "workers.json").read_text(encoding="utf-8"))
    document["workers"][0]["host"] = "0.0.0.0"
    corrupt = tmp_path / "corrupt-workers.json"
    corrupt.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DiagnosticError) as caught:
        load_workers(corrupt)
    assert caught.value.code == "E_SPEC"
    assert caught.value.field == "host"
