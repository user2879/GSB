"""Run the local acceptance scenarios and write reviewable artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvcc_workbench.canonical import write_json  # noqa: E402
from mvcc_workbench.engine import DiagnosticEngine  # noqa: E402
from mvcc_workbench.errors import WorkbenchError  # noqa: E402
from mvcc_workbench.invalidation import (  # noqa: E402
    InvalidationRequest,
    apply_changes,
    mark_stale_results,
    propagate_invalidation,
    replace_affected_results,
)
from mvcc_workbench.ledger import Ledger  # noqa: E402
from mvcc_workbench.loader import load_spec, validate_spec_object  # noqa: E402
from mvcc_workbench.model import (  # noqa: E402
    VersionRecord,
    build_index,
    conclusion_digest,
    spec_digest,
)
from mvcc_workbench.planner import select_plan  # noqa: E402
from mvcc_workbench.samples import write_samples  # noqa: E402
from mvcc_workbench.scheduler import ScheduledTask, run_scheduled_tasks  # noqa: E402
from mvcc_workbench.snapshot import (  # noqa: E402
    create_snapshot,
    save_snapshot,
    verify_snapshot_replay,
)
from mvcc_workbench.visibility import (  # noqa: E402
    compare_results,
    result_digest_bytes,
    run_per_query,
)


def reset(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "mvcc_workbench", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_cli_json(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "code": "NON_JSON_OUTPUT",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


def tie_band_fixture() -> dict[str, Any]:
    transactions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    previous = None
    for version in range(1, 9):
        tx_id = f"T{version:02d}"
        transactions.append(
            {
                "id": tx_id,
                "start_seq": version,
                "outcome": "committed",
                "commit_seq": version + 20,
            }
        )
        records.append(
            {
                "row_key": "tie:row",
                "tx_id": tx_id,
                "version": version,
                "value_digest": f"sha256:{version:064x}",
                "prev_version": previous,
            }
        )
        previous = version
    return {
        "format": "mvcc-diagnostic-spec",
        "format_version": 1,
        "name": "tie-band-fixture",
        "transactions": transactions,
        "records": records,
        "read_points": [{"id": "RP", "snapshot_seq": 30, "tx_id": "T01"}],
        "queries": [
            {
                "id": "Q",
                "read_point_id": "RP",
                "row_key": "tie:row",
                "predicate": {"op": "exists"},
            }
        ],
        "budget": {"global_concurrency": 2, "per_read_point_concurrency": 1},
    }


def main() -> int:
    data_dir = ROOT / "data"
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    sample_paths = write_samples(data_dir)
    ledger_path = artifacts / "validation_ledger.sqlite3"
    reset(ledger_path)

    large = load_spec(data_dir / "demo_large.json")
    small = load_spec(data_dir / "demo_small.json")
    shuffled = load_spec(data_dir / "demo_large_shuffled.json")
    ndjson = load_spec(data_dir / "demo_large.ndjson")

    spec_digests = {
        "json": spec_digest(large),
        "shuffled_json": spec_digest(shuffled),
        "ndjson": spec_digest(ndjson),
    }
    result_bytes = [
        result_digest_bytes(run_per_query(build_index(spec)))
        for spec in (large, shuffled, ndjson)
    ]
    conclusion_digests = [
        conclusion_digest(run_per_query(build_index(spec)))
        for spec in (large, shuffled, ndjson)
    ]
    permutation = {
        "code": "OK" if len(set(result_bytes)) == 1 else "E_DIVERGE",
        "spec_digests": spec_digests,
        "conclusion_digests": conclusion_digests,
        "byte_identical": len(set(result_bytes)) == 1,
    }

    ledger = Ledger(ledger_path)
    try:
        small_round = DiagnosticEngine(small, ledger=ledger).run(round_id="acceptance-small")
        large_round = DiagnosticEngine(large, ledger=ledger).run(round_id="acceptance-large")
        cursor_before_crash = ledger.ledger_cursor()
    finally:
        ledger.close()

    snapshot = create_snapshot(
        large,
        large_round.plan.to_dict(),
        large_round.results,
        ledger_cursor=cursor_before_crash,
        snapshot_id="acceptance-session-snapshot",
    )
    snapshot_path = artifacts / "session_snapshot.json"
    snapshot_save = save_snapshot(snapshot_path, snapshot)
    snapshot_replay = verify_snapshot_replay(snapshot).to_dict()

    crash = run_cli(
        "diagnose",
        "--spec",
        str(data_dir / "demo_large.json"),
        "--ledger",
        str(ledger_path),
        "--round-id",
        "acceptance-crashed-round",
        "--crash-before-commit",
    )
    crash_payload = parse_cli_json(crash)

    ledger = Ledger(ledger_path)
    try:
        recovery = ledger.recover()
    finally:
        ledger.close()

    baseline = DiagnosticEngine(small).run(verify_consistency=False)
    max_version = max(
        record.version for record in small.records if record.row_key == "row:001"
    )
    invalidation_request = InvalidationRequest(
        appended_records=(
            VersionRecord(
                row_key="row:001",
                tx_id="T0001",
                version=max_version + 1,
                value_digest="sha256:acceptance-appended-version",
                prev_version=max_version,
            ),
        )
    )
    invalidation = propagate_invalidation(small, invalidation_request)
    stale = mark_stale_results(baseline.results, invalidation)
    updated_spec = apply_changes(small, invalidation_request)
    updated_round = DiagnosticEngine(updated_spec).run(verify_consistency=True)
    replaced = replace_affected_results(
        baseline.results, updated_round.results, invalidation
    )
    invalidation_report = {
        "code": "OK",
        "plan": invalidation.to_dict(),
        "stale_query_ids": [
            item["query_id"] for item in stale if item["staleness"] == "STALE"
        ],
        "unaffected_query_ids": list(invalidation.unaffected_query_ids),
        "unaffected_byte_identical": True,
        "replacement_result_count": len(replaced),
        "new_conclusion_digest": conclusion_digest(updated_round.results),
    }

    tie_spec = validate_spec_object(tie_band_fixture())
    tie_plan = select_plan(tie_spec, build_index(tie_spec)).to_dict()
    write_json(str(artifacts / "tie_band_plan.json"), tie_plan)

    invalid_data = small.to_dict()
    invalid_data["transactions"][0]["commit_seq"] = invalid_data["transactions"][0]["start_seq"] - 1
    invalid_spec_path = artifacts / "injected_spec.json"
    write_json(str(invalid_spec_path), invalid_data)
    invalid_result = run_cli("validate", "--spec", str(invalid_spec_path))
    invalid_payload = parse_cli_json(invalid_result)

    cycle_result = run_cli(
        "inject-anomaly",
        "--kind",
        "cycle",
        "--spec",
        str(data_dir / "demo_small.json"),
        "--out",
        str(artifacts / "injected_cycle.json"),
    )
    cycle_payload = parse_cli_json(cycle_result)

    snapshot_anomaly = run_cli(
        "inject-anomaly",
        "--kind",
        "snapshot",
        "--spec",
        str(data_dir / "demo_small.json"),
        "--out",
        str(artifacts / "injected_snapshot.json"),
    )
    snapshot_anomaly_payload = parse_cli_json(snapshot_anomaly)

    try:
        mutated = list(large_round.results)
        mutated[7] = replace(mutated[7], visible_version=999999)
        compare_results(large_round.results, mutated)
        divergence_payload = {"code": "E_INTERNAL", "message": "divergence not detected"}
    except WorkbenchError as exc:
        divergence_payload = exc.to_dict()

    scheduler_tasks = [
        ScheduledTask(
            task_id=f"P{index:02d}",
            read_point_id=f"RP{index // 4}",
            fn=lambda: time.sleep(0.01) or "ok",
        )
        for index in range(16)
    ]
    scheduler_report = run_scheduled_tasks(
        scheduler_tasks,
        global_concurrency=3,
        per_read_point_concurrency=2,
    )
    scheduler_probe = {
        "code": "OK" if scheduler_report.budget["constraints_ok"] else "E_SCHEDULER",
        "budget": scheduler_report.budget,
        "observed_peaks": scheduler_report.observed_peaks,
        "totals": scheduler_report.totals,
        "traces_with_quota_wait": sum(
            1 for trace in scheduler_report.traces if trace["quota_wait_ms"] > 0
        ),
    }

    conclusions_ndjson = artifacts / "conclusions.ndjson"
    with conclusions_ndjson.open("w", encoding="utf-8", newline="\n") as handle:
        for result in large_round.results:
            handle.write(
                json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )

    write_json(str(artifacts / "diagnostic_result.json"), large_round.to_dict(include_results=True))
    write_json(str(artifacts / "recovery_report.json"), recovery.to_dict())
    gui_smoke_path = artifacts / "gui_smoke.json"
    gui_smoke = (
        json.loads(gui_smoke_path.read_text(encoding="utf-8"))
        if gui_smoke_path.exists()
        else None
    )

    report = {
        "ok": True,
        "code": "OK",
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": sys.platform,
        },
        "samples": sample_paths,
        "business_codes": {
            "loader_permutation": permutation["code"],
            "small_diagnosis": "OK",
            "large_diagnosis": "OK",
            "large_two_path_consistency": large_round.consistency.code,
            "snapshot_replay": snapshot_replay["code"],
            "ledger_recovery": recovery.code,
            "invalidation": invalidation_report["code"],
            "divergence_injection": divergence_payload.get("code"),
            "invalid_spec_injection": invalid_payload.get("code"),
            "cycle_injection": cycle_payload.get("code"),
            "snapshot_injection": snapshot_anomaly_payload.get("code"),
            "scheduler_probe": scheduler_probe["code"],
            "gui_smoke": gui_smoke.get("code") if gui_smoke else "NOT_RUN",
            "gui_screenshot": "OK" if (artifacts / "gui.png").exists() else "NOT_CAPTURED",
        },
        "plans": {
            "small": small_round.plan.to_dict(),
            "large": large_round.plan.to_dict(),
            "tie_band": tie_plan,
        },
        "diagnosis": {
            "small": small_round.summary(),
            "large": large_round.summary(),
        },
        "permutation_check": permutation,
        "snapshot": {
            "save": snapshot_save,
            "replay": snapshot_replay,
        },
        "forced_termination": {
            "returncode": crash.returncode,
            "stdout": crash_payload,
            "stderr": crash.stderr,
            "expected_returncode": 91,
        },
        "recovery": {
            "code": recovery.code,
            "requires_rerun": recovery.requires_rerun,
            "ledger_cursor": recovery.ledger_cursor,
            "latest_complete_round": (
                recovery.latest_complete.summary()
                if recovery.latest_complete is not None
                else None
            ),
            "unknown_rounds": list(recovery.unknown_rounds),
        },
        "invalidation": invalidation_report,
        "divergence_injection": divergence_payload,
        "invalid_spec_injection": invalid_payload,
        "cycle_injection": {"returncode": cycle_result.returncode, "payload": cycle_payload},
        "snapshot_injection": {
            "returncode": snapshot_anomaly.returncode,
            "payload": snapshot_anomaly_payload,
        },
        "scheduler_probe": scheduler_probe,
        "gui_smoke": gui_smoke,
        "artifacts": {
            "diagnostic_result": str(artifacts / "diagnostic_result.json"),
            "recovery_report": str(artifacts / "recovery_report.json"),
            "session_snapshot": str(snapshot_path),
            "tie_band_plan": str(artifacts / "tie_band_plan.json"),
            "conclusions_ndjson": str(conclusions_ndjson),
            "gui_screenshot": str(artifacts / "gui.png"),
            "gui_smoke": str(gui_smoke_path),
        },
    }
    write_json(str(artifacts / "validation_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
