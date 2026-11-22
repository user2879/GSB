from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from pageagg.catalog import load_catalog
from pageagg.engine import build_execution
from pageagg.errors import EReplay, ERoute
from pageagg.ledger import Ledger
from pageagg.replay import replay_database, validate_database
from pageagg.sample import generate_sample
from pageagg.service import aggregate_database, evolve_database, recover_database

from .helpers import catalog_object, rewrite_catalog_json, write_custom_dataset


def test_aggregate_ledger_replay_and_validate(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    db = tmp_path / "ledger.db"
    response = aggregate_database(sample["catalog_path"], db, 4)
    assert response["snapshot_action"]["action"] == "created"
    assert response["execution"]["result_digest"] == build_execution(
        load_catalog(sample["catalog_path"]), 4
    ).result_digest
    replay = replay_database(db)
    assert replay["complete"] is True
    assert replay["checked_batches"] == 4
    validation = validate_database(db)
    assert validation["ok"] is True
    assert all(check["ok"] for check in validation["checks"])


def test_crash_recovery_only_trusts_committed_batches(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    db = tmp_path / "crash.db"
    command = [
        sys.executable,
        "-m",
        "pageagg",
        "aggregate",
        "--catalog",
        sample["catalog_path"],
        "--db",
        str(db),
        "--budget",
        "4",
        "--crash-during-batch",
        "2",
    ]
    process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True)
    assert process.returncode == 137
    recovered = recover_database(db)
    assert recovered["forced_exit_report"] is None
    assert recovered["forced_exit_report_available"] is False
    assert recovered["snapshot"]["cursor"] == 3
    assert len(recovered["unknown_attempts"]) == 1
    resumed = aggregate_database(sample["catalog_path"], db, 4)
    assert resumed["snapshot_action"]["action"] == "resumed"
    assert resumed["snapshot_action"]["verified"]["resume_cursor"] == 3
    assert validate_database(db)["ok"] is True


def test_evolve_budget_replaces_but_preserves_group_bytes(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    db = tmp_path / "evolve.db"
    first = aggregate_database(sample["catalog_path"], db, 4)
    old_groups = first["result"]["groups"]
    evolved = evolve_database(db, sample["catalog_path"], 2)
    propagation = evolved["propagation"]
    assert evolved["snapshot_action"]["plan_version"] == 2
    assert evolved["snapshot_action"]["previous_plan_version"] == 1
    assert propagation["reasons"] == ["slot_budget_changed", "selected_route_changed"]
    assert propagation["value_affected_pages"] == []
    assert propagation["route_affected_pages"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert propagation["added_groups"] == []
    assert propagation["removed_groups"] == []
    assert propagation["changed_groups"] == []
    assert propagation["unaffected_groups_byte_identical"] is True
    assert all(item["old_digest"] == item["new_digest"] for item in propagation["unaffected_group_digests"])


def test_evolve_reports_added_removed_changed_groups(tmp_path) -> None:
    base = tmp_path / "base"
    first_catalog = write_custom_dataset(
        base,
        groups_by_page={1: [1, 1, 2, 2], 2: [3, 3, 4, 4]},
    )
    db = tmp_path / "groups.db"
    aggregate_database(first_catalog, db, 1)
    changed = tmp_path / "changed"
    second_catalog = write_custom_dataset(
        changed,
        groups_by_page={1: [1, 1, 1, 5], 2: [3, 3, 4, 4]},
        values_by_page={1: [10, 20, 30, 40], 2: [0, 1, 2, 3]},
    )
    evolved = evolve_database(db, second_catalog, 1)
    propagation = evolved["propagation"]
    assert propagation["added_groups"] == [5]
    assert propagation["removed_groups"] == [2]
    assert propagation["changed_groups"] == [1]
    assert propagation["unaffected_groups"] == [3, 4]
    assert propagation["unaffected_groups_byte_identical"] is True
    assert all(item["old_digest"] == item["new_digest"] for item in propagation["unaffected_group_digests"])


def test_replay_localizes_first_page_outcome_mismatch(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    db = tmp_path / "tamper.db"
    aggregate_database(sample["catalog_path"], db, 4)
    connection = sqlite3.connect(db)
    row = connection.execute(
        "SELECT snapshot_id, payload_json FROM page_outcomes WHERE access_seq=2"
    ).fetchone()
    snapshot_id, payload_json = row
    payload = json.loads(payload_json)
    payload["outcome"] = "TAMPERED"
    with Ledger(db, verify=False) as ledger:
        ledger.connection.execute("BEGIN")
        ledger.connection.execute(
            "UPDATE page_outcomes SET payload_json=? WHERE snapshot_id=? AND access_seq=2",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), snapshot_id),
        )
        ledger.refresh_content_digest()
        ledger.connection.execute("COMMIT")
    with pytest.raises(EReplay) as caught:
        replay_database(db)
    assert caught.value.code == "E_REPLAY"
    assert caught.value.page_id == 2
    assert caught.value.event_seq == 2
    assert caught.value.field == "outcome"


def test_validate_route_conflict_is_e_route(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    db = tmp_path / "route.db"
    aggregate_database(sample["catalog_path"], db, 4)
    connection = sqlite3.connect(db)
    event = connection.execute(
        "SELECT snapshot_id, eviction_events.payload_json, event_seq FROM eviction_events "
        "ORDER BY event_seq LIMIT 1"
    ).fetchone()
    snapshot_id, payload_json, event_seq = event
    payload = json.loads(payload_json)
    payload["victim_page_id"] = 999
    with Ledger(db, verify=False) as ledger:
        ledger.connection.execute("BEGIN")
        ledger.connection.execute(
            "UPDATE eviction_events SET payload_json=? WHERE snapshot_id=? AND event_seq=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), snapshot_id, event_seq),
        )
        ledger.refresh_content_digest()
        ledger.connection.execute("COMMIT")
    with pytest.raises(ERoute) as caught:
        validate_database(db)
    assert caught.value.code == "E_ROUTE"
    assert caught.value.event_seq == event_seq
    assert caught.value.page_id == 999
