"""End-to-end orchestration, recovery, conservation audit and snapshots."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import FORMAT_VERSION
from .canonical import content_digest
from .errors import ConserveError, EdgeError, SpecError
from .invalidation import plan_current_invalidations
from .jsonlog import StructuredLogger
from .ledger import Ledger
from .processes import NodeProcessManager
from .scheduler import DispatchResult, DispatchScheduler, initial_node_quotas
from .spec import EdgeSpec, load_spec
from .water import (
    MICRO,
    WaterCandidate,
    attribute_water_error,
    select_water_candidate,
)


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


class EdgeController:
    def __init__(
        self,
        spec: EdgeSpec,
        ledger: Ledger,
        *,
        log_path: str | Path = "logs/edgecache.jsonl",
        exit_report_path: str | Path = "artifacts/exit_report.json",
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.logger = StructuredLogger(log_path)
        self.exit_report_path = Path(exit_report_path)
        self.run_id: str | None = None
        self.attempt_batch_ids: set[str] = set()
        self.pre_conclusion: dict[str, Any] | None = None
        self.post_conclusion: dict[str, Any] | None = None
        self.recovery_unknown_ids: list[str] = []

    def _new_run_id(self) -> str:
        return f"run-{self.spec.digest[:12]}-{int(time.time() * 1000)}"

    def _select_or_create_run(self) -> tuple[str, bool]:
        open_runs = self.ledger.find_open_runs(self.spec.digest)
        if open_runs:
            run_id = open_runs[0]
            self.recovery_unknown_ids = self.ledger.mark_interrupted_pending_unknown(run_id)
            if self.recovery_unknown_ids:
                self.logger.emit(
                    "RECOVERY_UNKNOWN",
                    level="WARN",
                    run_id=run_id,
                    batch_ids=self.recovery_unknown_ids,
                    requires_rerun=True,
                )
                self.ledger.add_log(
                    run_id=run_id,
                    level="WARN",
                    code="RECOVERY_UNKNOWN",
                    details={
                        "batch_ids": self.recovery_unknown_ids,
                        "requires_rerun": True,
                    },
                )
            return run_id, True
        run_id = self._new_run_id()
        self.ledger.create_run(
            run_id,
            self.spec.digest,
            initial_node_quotas(self.spec),
            self.spec.global_spec.available_quota,
        )
        return run_id, False

    def _uncommitted_jobs(self) -> list[tuple[Any, Any, Any]]:
        committed = {
            str(batch["batch_id"])
            for batch in self.ledger.iter_batches(self.run_id, states=("committed",))
        }
        jobs = []
        revision_by_id = self.spec.revision_by_id()
        node_by_id = self.spec.node_by_id()
        for fragment in self.spec.ordered_fragments():
            if fragment.batch_id in committed:
                continue
            jobs.append((fragment, node_by_id[fragment.node_id], revision_by_id[fragment.quota_revision]))
        return jobs

    def _pre_conservation(self) -> dict[str, Any]:
        assert self.run_id is not None
        self.ledger.clear_conservation_checks(self.run_id)
        quota = self.ledger.quota_snapshot(self.run_id)
        node_available = {item["node_id"]: int(item["available"]) for item in quota["nodes"]}
        global_available = int(quota["global_available"])
        jobs = self._uncommitted_jobs()
        self.attempt_batch_ids = {fragment.batch_id for fragment, _, _ in jobs}
        checks: list[dict[str, Any]] = []
        for fragment, node, revision in jobs:
            expected = fragment.evict_blocks * self.spec.global_spec.block_weight
            before_node = node_available.get(node.node_id, 0)
            before_global = global_available
            check = self.ledger.record_conservation_check(
                self.run_id,
                "pre",
                batch_id=fragment.batch_id,
                node_id=node.node_id,
                quota_version=revision.revision_id,
                evicted_blocks=fragment.evict_blocks,
                block_weight=self.spec.global_spec.block_weight,
                node_delta=expected,
                global_delta=expected,
                node_before=before_node,
                global_before=before_global,
            )
            checks.append(check)
            node_available[node.node_id] = before_node - expected
            global_available = before_global - expected
            if check["node_diff"] != 0 or check["global_diff"] != 0:
                raise ConserveError(
                    fragment.batch_id,
                    int(check["node_diff"]),
                    int(check["global_diff"]),
                    revision.revision_id,
                    phase="pre",
                )
        self.pre_conclusion = self.ledger.conservation_conclusion(self.run_id, "pre")
        self.pre_conclusion["planned_batch_ids"] = sorted(self.attempt_batch_ids)
        return self.pre_conclusion

    def _post_conservation(self) -> dict[str, Any]:
        assert self.run_id is not None
        post_checks: list[dict[str, Any]] = []
        for batch in self.ledger.committed_batches(self.run_id):
            if str(batch["batch_id"]) not in self.attempt_batch_ids:
                continue
            check = self.ledger.record_conservation_check(
                self.run_id,
                "post",
                batch_id=str(batch["batch_id"]),
                node_id=str(batch["node_id"]),
                quota_version=int(batch["quota_version"]),
                evicted_blocks=int(batch["evict_blocks"]),
                block_weight=int(batch["block_weight"]),
                node_delta=int(batch["node_delta"]),
                global_delta=int(batch["global_delta"]),
                node_before=int(batch["node_before"]),
                global_before=int(batch["global_before"]),
            )
            post_checks.append(check)
        self.post_conclusion = self.ledger.conservation_conclusion(self.run_id, "post")
        self.post_conclusion["observed_batch_ids"] = sorted(item["batch_id"] for item in post_checks)
        pre_ok = bool(self.pre_conclusion["all_ok"]) if self.pre_conclusion else True
        post_ok = bool(self.post_conclusion["all_ok"])
        common = set(self.pre_conclusion.get("planned_batch_ids", [])) & set(
            self.post_conclusion.get("observed_batch_ids", [])
        )
        pre_by_batch = {
            check["batch_id"]: check for check in self.pre_conclusion.get("checks", [])
        }
        post_by_batch = {
            check["batch_id"]: check for check in self.post_conclusion.get("checks", [])
        }
        mismatch_batch = None
        for batch_id in sorted(common):
            pre = pre_by_batch.get(batch_id)
            post = post_by_batch.get(batch_id)
            if pre is None or post is None:
                mismatch_batch = batch_id
                break
            if (
                pre["ok"] != post["ok"]
                or pre["node_diff"] != post["node_diff"]
                or pre["global_diff"] != post["global_diff"]
            ):
                mismatch_batch = batch_id
                break
        consistent = pre_ok == post_ok and mismatch_batch is None
        self.post_conclusion["pre_post_consistent"] = consistent
        if not post_ok:
            bad = next(check for check in post_checks if not check["ok"])
            raise ConserveError(
                bad["batch_id"],
                int(bad["node_diff"]),
                int(bad["global_diff"]),
                int(bad["quota_version"]),
                phase="post",
            )
        if mismatch_batch is not None:
            pre = pre_by_batch[mismatch_batch]
            post = post_by_batch[mismatch_batch]
            raise ConserveError(
                mismatch_batch,
                int(post["node_diff"]),
                int(post["global_diff"]),
                int(post["quota_version"]),
                phase="pre_post_mismatch",
                pre_node_diff=pre["node_diff"],
                pre_global_diff=pre["global_diff"],
            )
        return self.post_conclusion

    def _water_report(self, results: list[DispatchResult]) -> dict[str, Any]:
        assert self.run_id is not None
        committed_batches = self.ledger.committed_batches(self.run_id)
        sample_by_batch = {
            str(batch["batch_id"]): json.loads(str(batch["sample_json"]))
            for batch in committed_batches
            if batch["sample_json"] is not None
        }
        total_blocks = sum(fragment.block_count for fragment in self.spec.fragments)
        uncovered_blocks = 0
        for batch in committed_batches:
            payload = sample_by_batch.get(str(batch["batch_id"]), {})
            uncovered_blocks += int(payload.get("uncovered_blocks", 0))
        unknown = self.ledger.unknown_batches(self.run_id)
        for batch in unknown:
            fragment = self.spec.fragment_by_id().get(str(batch["fragment_id"]))
            if fragment is not None:
                uncovered_blocks += fragment.block_count
        unknown_evictions = sum(int(batch["evict_blocks"]) for batch in unknown)
        attribution = attribute_water_error(
            total_blocks,
            min(uncovered_blocks, total_blocks),
            len(unknown),
            self.spec.global_spec,
        )
        attribution_payload = attribution.to_dict()
        quota = self.ledger.quota_snapshot(self.run_id)
        actual_water = int(quota["global_available"])
        error_bound = int(attribution_payload["bound_micro"])
        candidate_decision = None
        if unknown:
            unknown_water = actual_water - unknown_evictions * self.spec.global_spec.block_weight
            decision = select_water_candidate(
                WaterCandidate("known_committed_water", actual_water, unknown_evictions),
                WaterCandidate("unknown_not_deducted_water", unknown_water, unknown_evictions),
                error_bound,
            )
            candidate_decision = decision.to_dict()
            attribution_payload["affected_by_error"] = decision.affected_by_error
        return {
            "initial_available": self.spec.global_spec.available_quota,
            "actual_available": actual_water,
            "committed_evictions": sum(
                int(batch["evict_blocks"]) for batch in committed_batches
            ),
            "unknown_evictions": unknown_evictions,
            "unknown_batches": len(unknown),
            "error_attribution": attribution_payload,
            "candidate_decision": candidate_decision,
        }

    def _snapshot_payload(
        self,
        *,
        status: str,
        results: list[DispatchResult],
        scheduler_summary: dict[str, Any],
        process_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        assert self.run_id is not None
        quota = self.ledger.quota_snapshot(self.run_id)
        batches = []
        for batch in self.ledger.iter_batches(self.run_id):
            batches.append(
                {
                    "batch_id": str(batch["batch_id"]),
                    "fragment_id": str(batch["fragment_id"]),
                    "node_id": str(batch["node_id"]),
                    "revision_id": int(batch["revision_id"]),
                    "sequence": int(batch["sequence"]),
                    "state": str(batch["state"]),
                    "evict_blocks": int(batch["evict_blocks"]),
                    "node_delta": int(batch["node_delta"]),
                    "global_delta": int(batch["global_delta"]),
                    "content_digest": batch["content_digest"],
                    "error_code": batch["error_code"],
                    "requires_rerun": bool(batch["requires_rerun"]),
                }
            )
        water = self._water_report(results)
        payload = {
            "format_version": FORMAT_VERSION,
            "run_id": self.run_id,
            "spec_digest": self.spec.digest,
            "status": status,
            "ledger_digest": self.ledger.ledger_digest(self.run_id),
            "quota": quota,
            "water": water,
            "batches": sorted(batches, key=lambda item: (item["node_id"], item["sequence"], item["batch_id"])),
            "unknown_batches": [
                {
                    "batch_id": str(batch["batch_id"]),
                    "fragment_id": str(batch["fragment_id"]),
                    "node_id": str(batch["node_id"]),
                    "evict_blocks": int(batch["evict_blocks"]),
                    "requires_rerun": True,
                    "error_code": batch["error_code"],
                }
                for batch in self.ledger.unknown_batches(self.run_id)
            ],
            "conservation": {
                "pre": self.pre_conclusion,
                "post": self.post_conclusion,
            },
            "invalidations": plan_current_invalidations(self.spec),
            "dispatch": {
                "summary": scheduler_summary,
                "results": [result.to_dict() for result in results],
            },
            "processes": process_snapshot,
            "recovery_unknown_batch_ids": self.recovery_unknown_ids,
        }
        return payload

    def _write_exit_report(self, payload: dict[str, Any], error: EdgeError | None = None) -> None:
        report = {
            "format_version": FORMAT_VERSION,
            "complete": True,
            "spec_digest": self.spec.digest,
            "run_id": self.run_id,
            "generated_at_ms": int(time.time() * 1000),
            "error": None if error is None else error.to_dict(),
            "snapshot_digest": content_digest(payload),
            "snapshot": payload,
        }
        _atomic_json(self.exit_report_path, report)

    def run(
        self,
        *,
        injection: str = "none",
        target_node_id: str | None = None,
        log_dir: str | Path = "logs",
        exit_report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if exit_report_path is not None:
            self.exit_report_path = Path(exit_report_path)
        run_id, recovered = self._select_or_create_run()
        self.run_id = run_id
        pre = self._pre_conservation()
        processes = NodeProcessManager(self.spec, log_dir)
        results: list[DispatchResult] = []
        scheduler_summary: dict[str, Any] = {}
        status = "open"
        fatal_error: EdgeError | None = None
        try:
            processes.start(injection=injection, target_node_id=target_node_id)
            scheduler = DispatchScheduler(self.spec, self.ledger, run_id, processes, self.logger)
            results, scheduler_summary = scheduler.run()
            post = self._post_conservation()
            committed_count = len(self.ledger.committed_batches(run_id))
            expected_count = len(self.spec.fragments)
            status = "complete" if committed_count == expected_count else "interrupted"
            runner: dict[str, Any] = {
                "ok": status == "complete",
                "status": status,
                "run_id": run_id,
                "recovered": recovered,
                "recovery_unknown_batch_ids": self.recovery_unknown_ids,
                "pre_conservation": pre,
                "post_conservation": post,
                "scheduler": scheduler_summary,
                "results": [result.to_dict() for result in results],
            }
            snapshot = self._snapshot_payload(
                status=status,
                results=results,
                scheduler_summary=scheduler_summary,
                process_snapshot=processes.process_snapshot(),
            )
            snapshot_save = self.ledger.save_snapshot(run_id, snapshot)
            run_digest = content_digest(snapshot)
            self.ledger.set_run_status(run_id, status, run_digest)
            runner["snapshot"] = snapshot_save
            runner["snapshot_payload"] = snapshot
            runner["ledger_digest"] = snapshot["ledger_digest"]
            self._write_exit_report(snapshot)
            return runner
        except BaseException as exc:
            status = "interrupted"
            if isinstance(exc, EdgeError):
                fatal_error = exc
            try:
                self.ledger.mark_interrupted_pending_unknown(run_id, reason="graceful_or_fatal_stop")
            except Exception:
                pass
            try:
                self.ledger.set_run_status(run_id, status, self.ledger.ledger_digest(run_id))
            except Exception:
                pass
            snapshot = self._snapshot_payload(
                status=status,
                results=results,
                scheduler_summary=scheduler_summary,
                process_snapshot=processes.process_snapshot(),
            )
            try:
                self.ledger.save_snapshot(run_id, snapshot)
            except Exception:
                pass
            self._write_exit_report(snapshot, fatal_error)
            if isinstance(exc, EdgeError):
                raise
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise EdgeError(
                "E_INTERNAL",
                f"controller failed: {type(exc).__name__}",
                details={"error": str(exc)},
                exit_code=20,
            ) from exc
        finally:
            processes.cleanup()


def run_from_paths(
    spec_path: str | Path,
    ledger_path: str | Path,
    *,
    injection: str = "none",
    target_node_id: str | None = None,
    fresh: bool = False,
    log_dir: str | Path = "logs",
    exit_report_path: str | Path = "artifacts/exit_report.json",
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    ledger = Ledger(ledger_path, reset=fresh)
    try:
        return EdgeController(
            spec,
            ledger,
            log_path=Path(log_dir) / "edgecache.jsonl",
            exit_report_path=exit_report_path,
        ).run(injection=injection, target_node_id=target_node_id, log_dir=log_dir)
    finally:
        ledger.close()

