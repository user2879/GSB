from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .blocks import ContentStore, MaterializedBlocks
from .engine import replay_spec
from .invalidation import deletion_events_to_ledger_keys
from .ledger import Ledger, open_session_ledger
from .scheduler import BlockKey, make_tasks, run_quota_schedule


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(".")

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def ledger_db(self) -> Path:
        return self.root / "ledger" / "blockledger.sqlite"

    @property
    def session(self) -> Path:
        return self.root / "ledger" / "session.json"

    @property
    def content_store(self) -> Path:
        return self.root / "snapshots" / "content"

    @property
    def live_blocks(self) -> Path:
        return self.root / "snapshots" / "live"

    @property
    def work_root(self) -> Path:
        return self.root / "snapshots" / "work"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"


def _status_counts(ledger: Ledger) -> dict[str, int]:
    counts: dict[str, int] = {}
    selected = 0
    for row in ledger.latest_rows():
        counts[row["verification_status"]] = counts.get(row["verification_status"], 0) + 1
        selected += int(row["sample_selected"])
    counts["sample_selected_blocks"] = selected
    counts["total_latest_blocks"] = len(ledger.latest_rows())
    return counts


def _corrupt_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        for outcome in result["outcomes"]:
            if outcome["verification_status"] != "corrupt":
                continue
            findings.append(
                {
                    "code": "E_CORRUPT",
                    "volume_id": outcome["volume_id"],
                    "block_index": outcome["block_index"],
                    "expected_digest": outcome["expected_digest"],
                    "actual_digest": outcome.get("restored_digest"),
                    "rollback_id": result["task"]["rollback_id"],
                    "batch_id": result["task"]["batch_id"],
                    "field": "digest",
                }
            )
    findings.sort(key=lambda item: (item["volume_id"], item["block_index"], item["rollback_id"]))
    return findings


def _verification_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "candidate_blocks": 0,
        "sampled_blocks": 0,
        "subprocess_attempted_blocks": 0,
        "passed_blocks": 0,
        "corrupt_blocks": 0,
        "unknown_blocks": 0,
        "unchecked_blocks": 0,
    }
    for result in results:
        for outcome in result["outcomes"]:
            summary["candidate_blocks"] += 1
            if outcome.get("sample_selected"):
                summary["sampled_blocks"] += 1
            status = outcome["verification_status"]
            if status == "unchecked":
                summary["unchecked_blocks"] += 1
            elif status == "pass":
                summary["passed_blocks"] += 1
                summary["subprocess_attempted_blocks"] += 1
            elif status == "corrupt":
                summary["corrupt_blocks"] += 1
                summary["subprocess_attempted_blocks"] += 1
            elif status == "unknown":
                summary["unknown_blocks"] += 1
                if outcome.get("worker_pid") is not None:
                    summary["subprocess_attempted_blocks"] += 1
    return summary


def execute_run(
    spec: dict[str, Any],
    *,
    ledger_path: str | Path,
    session_path: str | Path,
    content_root: str | Path,
    live_root: str | Path,
    work_root: str | Path,
    attempt: int = 1,
    only_rollback: str | None = None,
    fault_kill_blocks: set[BlockKey] | None = None,
    corrupt_blocks: set[BlockKey] | None = None,
    crash_after_lease_batch: str | None = None,
    apply_deletions: bool = True,
    python_executable: str | None = None,
) -> dict[str, Any]:
    ledger, replacement = open_session_ledger(spec, ledger_path, session_path)
    try:
        recovered_before_run = ledger.recover_interrupted()
        store = ContentStore(content_root)
        live = MaterializedBlocks(live_root)
        replay = replay_spec(spec, content_store=store, materialized=live)
        replay_digest = ledger.save_replay(replay)

        plans = replay.rollback_plans
        if only_rollback is not None:
            plans = [plan for plan in plans if plan["rollback_id"] == only_rollback]
            if not plans:
                raise ValueError(f"unknown rollback_id: {only_rollback}")
        tasks = make_tasks(plans, spec["budget"], attempt=attempt)

        selected_fault: BlockKey | None = None
        if fault_kill_blocks == {"__FIRST_SELECTED__"}:
            from .sampling import is_selected

            for task in tasks:
                for block in sorted(task["blocks"], key=lambda item: item["block_index"]):
                    if is_selected(
                        task["rollback_id"], task["volume_id"], block["block_index"],
                        int(spec["budget"]["verification_sample_permille"]),
                    ):
                        selected_fault = (task["volume_id"], int(block["block_index"]))
                        fault_kill_blocks = {selected_fault}
                        break
                if selected_fault is not None:
                    break

        results = run_quota_schedule(
            tasks,
            ledger=ledger,
            content_store=store,
            work_root=work_root,
            sample_permille=int(spec["budget"]["verification_sample_permille"]),
            timeout_ms=int(spec["budget"]["verification_timeout_ms"]),
            fault_kill_blocks=fault_kill_blocks,
            corrupt_blocks=corrupt_blocks,
            crash_after_lease_batch=crash_after_lease_batch,
            python_executable=python_executable,
        )

        invalidation_result = None
        if apply_deletions and only_rollback is None:
            affected, steps = deletion_events_to_ledger_keys(replay)
            if affected:
                invalidation_result = ledger.record_invalidation(
                    "event-delete-propagation", affected, steps
                )

        report = {
            "status": "ok",
            "session_replacement": replacement,
            "recovered_before_run": recovered_before_run,
            "replay_fingerprint": replay.fingerprint(),
            "replay_digest_ledger": replay_digest,
            "task_count": len(tasks),
            "batch_count": len(results),
            "selected_fault_block": list(selected_fault) if selected_fault else None,
            "fault_kill_blocks": [list(item) for item in sorted(fault_kill_blocks or set())],
            "corrupt_blocks": [list(item) for item in sorted(corrupt_blocks or set())],
            "commits": [item["commit"] for item in results],
            "wait_metrics": ledger.batch_wait_metrics(),
            "invalidation": invalidation_result,
            "verification_before_invalidation": _verification_summary(results),
            "counts": _status_counts(ledger),
            "corrupt_findings": _corrupt_findings(results),
            "ledger_content_digest": ledger.ledger_content_digest(),
        }
        if report["corrupt_findings"]:
            report["status"] = "E_CORRUPT"
        return report
    finally:
        ledger.close()


def next_rerun_attempt(ledger: Ledger) -> int:
    row = ledger.conn.execute(
        "SELECT COUNT(*) AS n FROM batches WHERE batch_id LIKE '%:rerun-a%'"
    ).fetchone()
    return int(row["n"]) + 1


def recover_and_rerun_unknown(
    spec: dict[str, Any],
    *,
    ledger_path: str | Path,
    session_path: str | Path,
    content_root: str | Path,
    work_root: str | Path,
    rerun: bool,
    python_executable: str | None = None,
) -> dict[str, Any]:
    ledger, replacement = open_session_ledger(spec, ledger_path, session_path)
    try:
        recovered = ledger.recover_interrupted()
        unknown_rows = ledger.unknown_tasks()
        rerun_results: list[dict[str, Any]] = []
        if rerun and unknown_rows:
            store = ContentStore(content_root)
            # Replay repopulates CAS after a fresh checkout or cache loss.
            replay_spec(spec, content_store=store)
            attempt = next_rerun_attempt(ledger)
            tasks = make_tasks([], spec["budget"], attempt=attempt, unknown_rows=unknown_rows)
            rerun_results = run_quota_schedule(
                tasks,
                ledger=ledger,
                content_store=store,
                work_root=work_root,
                sample_permille=int(spec["budget"]["verification_sample_permille"]),
                timeout_ms=int(spec["budget"]["verification_timeout_ms"]),
                python_executable=python_executable,
            )
        return {
            "status": "ok",
            "session_replacement": replacement,
            "recovered_interrupted": recovered,
            "unknown_before_rerun": len(unknown_rows),
            "rerun_batch_count": len(rerun_results),
            "rerun_commits": [item["commit"] for item in rerun_results],
            "counts": _status_counts(ledger),
            "durable_authority": {
                "snapshot_count": ledger.authoritative_state()["snapshot_count"],
                "block_count": ledger.authoritative_state()["block_count"],
                "ledger_content_digest": ledger.ledger_content_digest(),
            },
            "wait_metrics": ledger.batch_wait_metrics(),
            "ledger_content_digest": ledger.ledger_content_digest(),
        }
    finally:
        ledger.close()


def write_json_report(payload: dict[str, Any], path: str | Path | None) -> Path | None:
    if path is None:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    return output
