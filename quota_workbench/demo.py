"""Deterministic three-scenario demonstration and anomaly self-checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .canonical import canonical_json, digest_value, make_writable, write_canonical
from .errors import WorkbenchError
from .logging import append_structured_log
from .snapshot import load_snapshot, restore_snapshot, save_snapshot
from .session import WorkbenchSession
from .spec import load_injection_catalog, mutate_declaration_file


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "demo_output"


def reset_output() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for child in OUT.iterdir():
        try:
            make_writable(child)
        except OSError:
            pass
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _event_summary(result: Any) -> dict[str, Any]:
    record = result.record
    return {
        "event_id": record["event_id"],
        "terminal": record["terminal"],
        "iterations": [
            {
                "iteration": item["iteration"],
                "terminal_reason": item["terminal_reason"],
                "matrix_digest": item["matrix_digest"],
                "tenant_actual_units": {
                    row["tenant_id"]: row["actual_units"]
                    for row in item["matrix"]["tenants"]
                },
                "tenant_entitlement_units": {
                    row["tenant_id"]: row["entitlement_units"]
                    for row in item["matrix"]["tenants"]
                },
                "tenant_excess_units": {
                    row["tenant_id"]: row["excess_units"]
                    for row in item["matrix"]["tenants"]
                },
                "gap": item["gap"],
            }
            for item in record["iterations"]
        ],
        "preemptions": record["preemptions"],
        "final_matrix": record["final_matrix"],
        "workload_state": record["workload_state"],
        "gap": record["gap"],
    }


def scenario_normal() -> dict[str, Any]:
    db_path = OUT / "normal.sqlite3"
    session = WorkbenchSession.open(SAMPLES / "spec.json", SAMPLES / "normal.ndjson", db_path)
    try:
        results = session.run_all()
        assert len(results) == 4
        assert all(not result.unsatisfied for result in results)
        assert session.verify_replay()["status"] == "REPLAY_OK"
        preemption_count = sum(len(result.record["preemptions"]) for result in results)
        reclaimed = sum(result.record["terminal"]["reclaimed_units"] for result in results)
        assert preemption_count == 4
        assert reclaimed == 25
        return {
            "scenario": "normal_preemption",
            "status": "PASS",
            "cursor": session.cursor,
            "preemption_count": preemption_count,
            "reclaimed_units": reclaimed,
            "replay": session.verify_replay(),
            "events": [_event_summary(result) for result in results],
            "final_matrix": session.current_matrix(),
            "ledger_head_digest": session.ledger.head_digest,
        }
    finally:
        session.close()


def scenario_unsat() -> dict[str, Any]:
    db_path = OUT / "unsat.sqlite3"
    session = WorkbenchSession.open(SAMPLES / "spec.json", SAMPLES / "unsat.ndjson", db_path)
    try:
        result = session.run_next()
        assert result is not None and result.unsatisfied
        assert result.record["terminal"]["code"] == "E_UNSATISFIED"
        assert result.record["gap"]["remaining_gap_units"] == 12
        assert result.record["gap"]["affected_tenant_ids"] == ["tenant-gamma"]
        replay = session.verify_replay()
        return {
            "scenario": "unsatisfiable_gap",
            "status": "PASS_EXPECTED_UNSATISFIED",
            "business_code": "E_UNSATISFIED",
            "cursor": session.cursor,
            "result": _event_summary(result),
            "replay": replay,
            "ledger_head_digest": session.ledger.head_digest,
        }
    finally:
        session.close()


def scenario_snapshot_replay() -> dict[str, Any]:
    source_db = OUT / "snapshot-source.sqlite3"
    restored_db = OUT / "snapshot-restored.sqlite3"
    reference_db = OUT / "snapshot-reference.sqlite3"
    snapshot_path = OUT / "session.cursor-2.snapshot.json"
    source = WorkbenchSession.open(SAMPLES / "spec.json", SAMPLES / "replay.ndjson", source_db)
    restored_session = None
    reference = None
    try:
        for _ in range(2):
            assert source.run_next() is not None
        snapshot = save_snapshot(source, snapshot_path)
        source_head = source.ledger.head_digest
        source_replay = source.verify_replay()
        restored = restore_snapshot(
            snapshot_path,
            SAMPLES / "replay.ndjson",
            restored_db,
            current_declaration_path=SAMPLES / "spec_context_changed.json",
        )
        restored_session = restored.session
        assert restored_session.cursor == 2
        assert restored_session.ledger.head_digest == source_head
        restored_replay_before = restored_session.verify_replay()
        remaining = []
        while not restored_session.completed:
            result = restored_session.run_next()
            assert result is not None
            remaining.append(_event_summary(result))
        restored_replay_after = restored_session.verify_replay()

        reference = WorkbenchSession.open(SAMPLES / "spec.json", SAMPLES / "replay.ndjson", reference_db)
        reference_results = reference.run_all()
        assert reference.verify_replay()["status"] == "REPLAY_OK"
        assert restored_session.ledger.head_digest == reference.ledger.head_digest
        assert restored_session.state_digest() == reference.state_digest()
        restored_records = restored_session.ledger.committed_records()
        reference_records = reference.ledger.committed_records()
        assert [item["record_digest"] for item in restored_records] == [
            item["record_digest"] for item in reference_records
        ]
        assert restored_session.current_matrix() == reference.current_matrix()
        return {
            "scenario": "snapshot_save_restore_replay",
            "status": "PASS",
            "snapshot": {
                "path": str(snapshot.path),
                "cursor": snapshot.ledger_cursor,
                "content_digest": snapshot.content_digest,
                "head_digest": snapshot.ledger_head_digest,
                "readonly": True,
            },
            "whole_replacement_report": restored.change_report,
            "source_replay": source_replay,
            "restored_replay_before_remaining": restored_replay_before,
            "remaining_event_results": remaining,
            "restored_replay_after_remaining": restored_replay_after,
            "reference_event_count": len(reference_results),
            "restored_event_count": restored_session.cursor,
            "head_digest_equal": True,
            "state_digest_equal": True,
            "record_digests_equal": True,
            "restored_final_matrix": restored_session.current_matrix(),
        }
    finally:
        source.close()
        if restored_session is not None:
            restored_session.close()
        if reference is not None:
            reference.close()


def _capture_expected_error(callable_obj: Any, expected_code: str) -> WorkbenchError:
    try:
        callable_obj()
    except WorkbenchError as exc:
        assert exc.code == expected_code, (exc.code, expected_code, exc.to_dict())
        return exc
    raise AssertionError(f"expected {expected_code}, no error was raised")


def anomaly_checks(snapshot_path: Path) -> dict[str, Any]:
    catalog = load_injection_catalog(SAMPLES / "injections.json")
    log_path = OUT / "anomalies.ndjson"
    checks: list[dict[str, Any]] = []
    quota = _capture_expected_error(
        lambda: mutate_declaration_file(SAMPLES / "spec.json", catalog["quota-overflow"]["mutation"]),
        "E_QUOTA",
    )
    append_structured_log(
        log_path,
        {"phase": "INJECTION", "injection_id": "quota-overflow", "error": quota.to_dict()},
    )
    checks.append({"injection_id": "quota-overflow", "code": quota.code, "field": quota.field})

    reserve = _capture_expected_error(
        lambda: mutate_declaration_file(SAMPLES / "spec.json", catalog["reserve-overflow"]["mutation"]),
        "E_RESERVE",
    )
    append_structured_log(
        log_path,
        {"phase": "INJECTION", "injection_id": "reserve-overflow", "error": reserve.to_dict()},
    )
    checks.append({"injection_id": "reserve-overflow", "code": reserve.code, "field": reserve.field})

    valid_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    valid_data["declaration"]["cluster_total_units"] += 1
    bad_snapshot_path = OUT / "snapshot-digest-bad.json"
    write_canonical(bad_snapshot_path, valid_data)
    snapshot_error = _capture_expected_error(lambda: load_snapshot(bad_snapshot_path), "E_SNAPSHOT")
    append_structured_log(
        log_path,
        {
            "phase": "INJECTION",
            "injection_id": "snapshot-digest-mismatch",
            "error": snapshot_error.to_dict(),
        },
    )
    checks.append(
        {
            "injection_id": "snapshot-digest-mismatch",
            "code": snapshot_error.code,
            "field": snapshot_error.field,
        }
    )
    return {
        "status": "PASS",
        "checks": checks,
        "structured_log": str(log_path),
        "log_records": len(log_path.read_text(encoding="utf-8").splitlines()),
    }


def main() -> int:
    reset_output()
    scenarios: list[dict[str, Any]] = []
    try:
        normal = scenario_normal()
        scenarios.append(normal)
        print("=== 场景 1：常规抢占 ===")
        print(canonical_json(normal))

        unsat = scenario_unsat()
        scenarios.append(unsat)
        print("=== 场景 2：缺口不可满足 ===")
        print(canonical_json(unsat))

        replay = scenario_snapshot_replay()
        scenarios.append(replay)
        print("=== 场景 3：快照保存与恢复重放 ===")
        print(canonical_json(replay))

        anomalies = anomaly_checks(OUT / "session.cursor-2.snapshot.json")
        print("=== 异常注入自检 ===")
        print(canonical_json(anomalies))
    except AssertionError as exc:
        failure = {
            "status": "FAIL",
            "reason": f"assertion failed: {exc}",
        }
        print(canonical_json(failure))
        return 1
    except WorkbenchError as exc:
        failure = {"status": "FAIL", "error": exc.to_dict()}
        print(canonical_json(failure))
        return exc.exit_code

    summary = {
        "format_version": 1,
        "status": "PASS",
        "scenarios": scenarios,
        "anomaly_checks": anomalies,
    }
    summary["summary_digest"] = digest_value(summary)
    write_canonical(OUT / "results.json", summary)
    print("=== 汇总 ===")
    print(
        canonical_json(
            {
                "status": summary["status"],
                "summary_digest": summary["summary_digest"],
                "normal_head": scenarios[0]["ledger_head_digest"],
                "unsat_code": scenarios[1]["business_code"],
                "replay_head": scenarios[2]["restored_replay_after_remaining"]["head_digest"],
                "anomaly_codes": [item["code"] for item in anomalies["checks"]],
                "results_file": str(OUT / "results.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
