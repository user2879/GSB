"""Batch scheduling, transactional execution, recovery, and report generation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import (
    assignment_identifier,
    canonical_json_bytes,
    digest_inputs,
    digest_model,
    sha256_hex,
)
from .constants import (
    DEFAULT_SUPPORTED_SLOTS,
    MODEL_ARRAY_ORDER,
    NUMBERING_VERSION,
    REPORT_FORMAT_VERSION,
)
from .dataset import GeneratedAssets, load_assets
from .feed import FeedPlan, SlotGroup, load_feed
from .forward import BatchOutput, forward_batch
from .io_utils import write_json_atomic
from .ledger import Ledger, LedgerError, RecoveryInfo
from .logging_utils import JsonlLogger
from .snapshot import Snapshot, read_snapshot, snapshot_from_slots, write_snapshot


@dataclass(frozen=True)
class WorkspacePaths:
    workspace: Path
    feed: Path
    dataset: Path
    model: Path
    ledger: Path
    snapshot: Path
    report: Path
    structured_log: Path

    @classmethod
    def under(cls, workspace: Path) -> "WorkspacePaths":
        root = Path(workspace)
        return cls(
            workspace=root,
            feed=root / "feed.json",
            dataset=root / "dataset.npz",
            model=root / "model.npz",
            ledger=root / "ledger.sqlite3",
            snapshot=root / "snapshot.json",
            report=root / "report.json",
            structured_log=root / "structured.jsonl",
        )


@dataclass(frozen=True)
class RunOptions:
    supported_slots: tuple[int, ...] = DEFAULT_SUPPORTED_SLOTS
    numbering_version: str = NUMBERING_VERSION
    slot_order: tuple[int, ...] | None = None
    crash_after_commits: int | None = None


@dataclass
class ReconcileResult:
    generation_id: int
    resumed: bool
    replacement_reason: str | None
    recovery: RecoveryInfo = field(default_factory=lambda: RecoveryInfo((), ()))
    previous_generation: Mapping[str, Any] | None = None


class Scheduler:
    def __init__(self, paths: WorkspacePaths, options: RunOptions | None = None) -> None:
        self.paths = paths
        self.options = options or RunOptions()
        self.logger = JsonlLogger(paths.structured_log)
        self.assets: GeneratedAssets | None = None
        self.feed: FeedPlan | None = None
        self.config: dict[str, Any] | None = None
        self.fingerprint: dict[str, Any] | None = None
        self.input_digest: str | None = None
        self.param_digest: str | None = None
        self.numbering_version: str = NUMBERING_VERSION

    def run(self) -> dict[str, Any]:
        slots = _validate_slots(self.options.supported_slots)
        slot_order = _validate_slot_order(self.options.slot_order, slots)
        numbering_version = _validate_numbering_version(self.options.numbering_version)
        self.numbering_version = numbering_version
        self.assets = load_assets(self.paths.dataset, self.paths.model)
        self.feed = load_feed(
            self.paths.feed,
            supported_slots=slots,
            known_sample_ids=self.assets.sample_ids,
        )
        self.input_digest = digest_inputs(self.assets.sample_ids, self.assets.inputs)
        self.param_digest = digest_model(self.assets.model, MODEL_ARRAY_ORDER)
        self.config = {
            **dict(self.assets.config),
            "digest_contract": "batch-audit-v1",
            "numbering_version": numbering_version,
            "output_round_decimals": 12,
            "supported_slots": list(slots),
        }
        config_digest = sha256_hex(canonical_json_bytes(self.config))
        self.fingerprint = {
            "feed_digest": self.feed.feed_digest,
            "input_digest": self.input_digest,
            "param_digest": self.param_digest,
            "numbering_version": numbering_version,
            "config_digest": config_digest,
        }

        with Ledger(self.paths.ledger) as ledger:
            reconcile = self._reconcile_generation(ledger)
            generation = ledger.generation(reconcile.generation_id)
            if generation is None:
                raise LedgerError("generation disappeared during reconciliation")
            current_slots = ledger.slots(generation["generation_id"])
            incomplete = [
                row["batch_slot"]
                for row in current_slots
                if row["status"] not in {"committed", "empty"}
            ]

            if not reconcile.resumed:
                _unlink_if_exists(self.paths.report)
            elif incomplete:
                _unlink_if_exists(self.paths.report)

            self.logger.event(
                "run_reconciled",
                generation=generation["generation_id"],
                resumed=reconcile.resumed,
                replacement_reason=reconcile.replacement_reason,
                interrupted_attempts=list(reconcile.recovery.interrupted_attempts),
                unknown_slots_before_redo=list(reconcile.recovery.unknown_slots),
            )

            if not incomplete:
                if generation["status"] != "complete":
                    ledger.complete_generation(generation["generation_id"])
                generation = ledger.generation(generation["generation_id"])
                assert generation is not None
                self._write_snapshot(ledger, generation, status="complete")
                report = self._build_report(ledger, generation, reconcile)
                write_json_atomic(self.paths.report, report)
                self.logger.event(
                    "run_complete",
                    generation=generation["generation_id"],
                    report=str(self.paths.report),
                    idempotent_replay=True,
                )
                return report

            attempt_id = ledger.start_attempt(generation["generation_id"])
            redone: list[int] = []
            commits_this_run = 0
            try:
                for slot in slot_order:
                    rows = {row["batch_slot"]: row for row in ledger.slots(generation["generation_id"])}
                    current = rows[slot]
                    if current["status"] in {"committed", "empty"}:
                        self.logger.event(
                            "slot_skipped",
                            generation=generation["generation_id"],
                            slot=slot,
                            status=current["status"],
                        )
                        continue
                    group = next(
                        item for item in self.feed.groups if item.batch_slot == slot
                    )
                    self._commit_group(ledger, generation["generation_id"], attempt_id, group)
                    commits_this_run += 1
                    redone.append(slot)
                    refreshed = ledger.generation(generation["generation_id"])
                    assert refreshed is not None
                    self._write_snapshot(ledger, refreshed, status="active")
                    self.logger.event(
                        "slot_committed",
                        generation=generation["generation_id"],
                        slot=slot,
                        member_count=len(group.members),
                        overwritten_count=len(group.overwritten),
                        failed_count=len(group.failed),
                    )
                    if (
                        self.options.crash_after_commits is not None
                        and commits_this_run >= self.options.crash_after_commits
                    ):
                        self.logger.event(
                            "crash_injected",
                            generation=generation["generation_id"],
                            after_commits=commits_this_run,
                            slot=slot,
                            note="os._exit intentionally bypasses cleanup and exit report",
                        )
                        os._exit(99)

                ledger.finish_attempt(attempt_id, status="completed")
                ledger.complete_generation(generation["generation_id"])
                generation = ledger.generation(generation["generation_id"])
                assert generation is not None
                recovery = RecoveryInfo(
                    interrupted_attempts=reconcile.recovery.interrupted_attempts,
                    unknown_slots=reconcile.recovery.unknown_slots,
                    redone_slots=tuple(
                        slot for slot in redone if slot in reconcile.recovery.unknown_slots
                    ),
                )
                self._write_snapshot(ledger, generation, status="complete")
                report = self._build_report(ledger, generation, reconcile, recovery=recovery)
                write_json_atomic(self.paths.report, report)
                self.logger.event(
                    "run_complete",
                    generation=generation["generation_id"],
                    report=str(self.paths.report),
                    redone_slots=list(redone),
                )
                return report
            except Exception as exc:
                try:
                    ledger.finish_attempt(attempt_id, status="failed", reason=f"{type(exc).__name__}: {exc}")
                finally:
                    self.logger.event(
                        "run_failed",
                        generation=generation["generation_id"],
                        error=type(exc).__name__,
                        message=str(exc),
                    )
                raise

    def _reconcile_generation(self, ledger: Ledger) -> ReconcileResult:
        assert self.fingerprint is not None
        assert self.assets is not None
        assert self.feed is not None
        assert self.config is not None

        latest = ledger.latest_generation()
        try:
            snapshot = read_snapshot(self.paths.snapshot)
        except ValueError as exc:
            raise LedgerError(str(exc), reason="snapshot_unreadable") from exc

        resume = False
        if latest is not None and snapshot is not None:
            latest_fingerprint = _fingerprint_from_generation(latest)
            resume = (
                latest["status"] in {"active", "complete"}
                and int(snapshot.generation) == int(latest["generation_id"])
                and dict(snapshot.fingerprint) == self.fingerprint
                and latest_fingerprint == self.fingerprint
            )

        if resume:
            assert latest is not None
            recovery = ledger.recover_interrupted(latest["generation_id"])
            refreshed = ledger.generation(latest["generation_id"])
            assert refreshed is not None
            self._write_snapshot(
                ledger,
                refreshed,
                status="complete" if refreshed["status"] == "complete" else "active",
            )
            self.logger.event(
                "generation_resumed",
                generation=latest["generation_id"],
                interrupted_attempts=list(recovery.interrupted_attempts),
                unknown_slots=list(recovery.unknown_slots),
            )
            return ReconcileResult(
                generation_id=latest["generation_id"],
                resumed=True,
                replacement_reason=None,
                recovery=recovery,
            )

        replacement_reason = _replacement_reason(latest, snapshot, self.fingerprint)
        old_recovery = RecoveryInfo((), ())
        affected_samples = 0
        affected_slots: list[int] = []
        supersedes = None
        if latest is not None:
            supersedes = int(latest["generation_id"])
            old_recovery = ledger.recover_interrupted(supersedes)
            affected_samples = len(ledger.assignments(supersedes))
            affected_slots = sorted(
                int(row["batch_slot"])
                for row in ledger.slots(supersedes)
                if row["status"] in {"committed", "empty"} or row["assignment_count"] > 0
            )

        snapshot_version = int(snapshot.generation) if snapshot is not None else 0
        generation_id = ledger.next_generation_id(snapshot_version)
        arrays = {"inputs": self.assets.inputs, **self.assets.model}
        ledger.create_generation(
            generation_id=generation_id,
            feed_digest=self.feed.feed_digest,
            input_digest=self.input_digest or "",
            param_digest=self.param_digest or "",
            numbering_version=self.numbering_version,
            config=self.config,
            sample_ids=self.assets.sample_ids,
            supported_slots=self.options.supported_slots,
            arrays=arrays,
            replacement_reason=replacement_reason,
            supersedes_generation=supersedes,
            affected_samples=affected_samples,
            affected_slots=affected_slots,
        )
        generation = ledger.generation(generation_id)
        assert generation is not None
        self._write_snapshot(ledger, generation, status="active")
        self.logger.event(
            "generation_created",
            generation=generation_id,
            replacement_reason=replacement_reason,
            supersedes=supersedes,
            affected_samples=affected_samples,
            affected_slots=affected_slots,
            previous_unknown_slots=list(old_recovery.unknown_slots),
        )
        return ReconcileResult(
            generation_id=generation_id,
            resumed=False,
            replacement_reason=replacement_reason,
            recovery=old_recovery,
            previous_generation=latest,
        )

    def _commit_group(
        self,
        ledger: Ledger,
        generation_id: int,
        attempt_id: int,
        group: SlotGroup,
    ) -> None:
        assert self.assets is not None
        assert self.fingerprint is not None
        inputs_by_id = {
            sample_id: self.assets.inputs[index]
            for index, sample_id in enumerate(self.assets.sample_ids)
        }
        overwritten_by_sample: dict[str, int] = {}
        for item in group.overwritten:
            overwritten_by_sample[item.sample_id] = overwritten_by_sample.get(item.sample_id, 0) + 1

        members_payload = [
            {
                "sample_id": member.sample_id,
                "arrival_seq": member.arrival_seq,
                "rank": member.rank,
            }
            for member in group.members
        ]
        overwritten_payload = [item.as_dict() for item in group.overwritten]
        failed_payload = [item.as_dict() for item in group.failed]

        if not group.members:
            ledger.commit_slot(
                generation_id=generation_id,
                attempt_id=attempt_id,
                batch_slot=group.batch_slot,
                status="empty",
                members=(),
                overwritten=overwritten_payload,
                failed=failed_payload,
                output_digest=None,
                assignments=(),
            )
            return

        member_ids = [member.sample_id for member in group.members]
        batch: BatchOutput = forward_batch(member_ids, inputs_by_id, self.assets.model)
        assignments: list[dict[str, Any]] = []
        for member in group.members:
            assignment_id, assignment_digest = assignment_identifier(
                numbering_version=self.numbering_version,
                batch_slot=group.batch_slot,
                rank=member.rank,
                sample_id=member.sample_id,
                param_digest=self.param_digest or "",
            )
            assignments.append(
                {
                    "rank": member.rank,
                    "sample_id": member.sample_id,
                    "arrival_seq": member.arrival_seq,
                    "assignment_id": assignment_id,
                    "assignment_digest": assignment_digest,
                    "sample_output_digest": batch.sample_digests[member.sample_id],
                    "batch_output_digest": batch.batch_output_digest,
                    "overwritten_count": overwritten_by_sample.get(member.sample_id, 0),
                }
            )
        ledger.commit_slot(
            generation_id=generation_id,
            attempt_id=attempt_id,
            batch_slot=group.batch_slot,
            status="committed",
            members=members_payload,
            overwritten=overwritten_payload,
            failed=failed_payload,
            output_digest=batch.batch_output_digest,
            assignments=assignments,
        )

    def _write_snapshot(self, ledger: Ledger, generation: Mapping[str, Any], *, status: str) -> None:
        assert self.fingerprint is not None
        snapshot = snapshot_from_slots(
            generation=generation["generation_id"],
            fingerprint=self.fingerprint,
            status=status,
            slots=ledger.slots(generation["generation_id"]),
        )
        write_snapshot(self.paths.snapshot, snapshot)

    def _build_report(
        self,
        ledger: Ledger,
        generation: Mapping[str, Any],
        reconcile: ReconcileResult,
        *,
        recovery: RecoveryInfo | None = None,
    ) -> dict[str, Any]:
        assert self.feed is not None
        assert self.assets is not None
        generation_id = int(generation["generation_id"])
        slot_rows = ledger.slots(generation_id)
        assignments = ledger.assignments(generation_id)
        assignments_by_slot: dict[int, list[dict[str, Any]]] = {}
        for row in assignments:
            assignments_by_slot.setdefault(int(row["batch_slot"]), []).append(row)

        slots_report: list[dict[str, Any]] = []
        assigned_ids: set[str] = set()
        for row in slot_rows:
            slot = int(row["batch_slot"])
            slot_assignments = assignments_by_slot.get(slot, [])
            assigned_ids.update(str(item["sample_id"]) for item in slot_assignments)
            members = [
                {
                    "sample_id": str(item["sample_id"]),
                    "rank": int(item["rank"]),
                    "arrival_seq": int(item["arrival_seq"]),
                    "assignment_id": str(item["assignment_id"]),
                    "sample_output_digest": str(item["sample_output_digest"]),
                    "batch_output_digest": str(item["batch_output_digest"]),
                }
                for item in sorted(slot_assignments, key=lambda item: int(item["rank"]))
            ]
            slots_report.append(
                {
                    "batch_slot": slot,
                    "status": row["status"],
                    "error_code": row["error_code"],
                    "param_digest": generation["param_digest"],
                    "members": members,
                    "participant_sample_ids": [item["sample_id"] for item in members],
                    "output_digest": row["output_digest"],
                    "overwritten": [
                        {
                            "sample_id": str(item["sample_id"]),
                            "arrival_seq": int(item["arrival_seq"]),
                            "batch_slot": int(item["batch_slot"]),
                            "kept_arrival_seq": int(item["kept_arrival_seq"]),
                        }
                        for item in row["overwritten"]
                    ],
                    "failed": row["failed"],
                }
            )

        failed_entries = [item.as_dict() for item in self.feed.failed_entries]
        failed_by_slot: dict[str, list[dict[str, Any]]] = {}
        for item in failed_entries:
            failed_by_slot.setdefault(str(item["batch_slot"]), []).append(item)
        failed_by_reason: dict[str, int] = {}
        for item in failed_entries:
            reason = str(item["reason"])
            failed_by_reason[reason] = failed_by_reason.get(reason, 0) + 1

        failures_by_sample: dict[str, list[dict[str, Any]]] = {}
        for item in self.feed.entries_by_arrival:
            if not item.admitted:
                failures_by_sample.setdefault(item.sample_id, []).append(
                    {
                        "slot": item.batch_slot,
                        "arrival_seq": item.arrival_seq,
                        "reason": item.reject_reason or "admission_rejected",
                    }
                )
        unassigned: list[dict[str, Any]] = []
        known_sample_ids = set(self.assets.sample_ids)
        candidate_ids = list(self.assets.sample_ids)
        for sample_id in failures_by_sample:
            if sample_id not in known_sample_ids:
                candidate_ids.append(sample_id)
        for sample_id in sorted(candidate_ids):
            if sample_id in assigned_ids:
                continue
            failures = failures_by_sample.get(sample_id, [])
            if failures:
                unassigned.append(
                    {
                        "sample_id": sample_id,
                        "reason": "ADMISSION_FAILED:" + str(failures[0]["reason"]),
                        "failures": failures,
                    }
                )
            else:
                unassigned.append(
                    {
                        "sample_id": sample_id,
                        "reason": "NOT_DELIVERED",
                        "failures": [],
                    }
                )

        if recovery is None:
            recovery = reconcile.recovery
        report = {
            "report_format_version": REPORT_FORMAT_VERSION,
            "status": "complete" if all(row["status"] in {"committed", "empty"} for row in slot_rows) else "incomplete",
            "generation": generation_id,
            "version": generation_id,
            "generation_status": generation["status"],
            "fingerprints": {
                "feed_digest": generation["feed_digest"],
                "input_digest": generation["input_digest"],
                "param_digest": generation["param_digest"],
                "numbering_version": generation["numbering_version"],
                "config_digest": self.fingerprint["config_digest"] if self.fingerprint else None,
            },
            "version_evolution": {
                "replacement_reason": generation["replacement_reason"],
                "supersedes_generation": generation["supersedes_generation"],
                "affected_samples": generation["affected_samples"],
                "affected_slots": generation["affected_slots"],
            },
            "progress": {
                "cursor": generation["progress_cursor"],
                "completed_slot_count": generation["completed_slot_count"],
                "total_slots": len(slot_rows),
            },
            "recovery": {
                "interrupted_attempts": list(recovery.interrupted_attempts),
                "unknown_slots_before_redo": list(recovery.unknown_slots),
                "redone_slots": list(recovery.redone_slots),
            },
            "slots": slots_report,
            "empty_slots": [int(item) for item in self.feed.empty_slots],
            "failed_entries": failed_entries,
            "failed_by_slot": failed_by_slot,
            "failed_reason_counts": failed_by_reason,
            "overwritten_entries": [item.as_dict() for item in self.feed.overwritten_entries],
            "unassigned_samples": unassigned,
        }
        return report


def _validate_slots(slots: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(int(item) for item in slots))
    if not normalized:
        raise LedgerError("at least one batch slot must be configured")
    if len(set(normalized)) != len(normalized):
        raise LedgerError("supported batch slots contain duplicates")
    if any(item < 0 for item in normalized):
        raise LedgerError("batch slots must be non-negative integers")
    return normalized


def _validate_slot_order(order: Sequence[int] | None, slots: Sequence[int]) -> tuple[int, ...]:
    if order is None:
        return tuple(slots)
    normalized = tuple(int(item) for item in order)
    if sorted(normalized) != sorted(int(item) for item in slots) or len(set(normalized)) != len(normalized):
        raise LedgerError("slot_order must be a permutation of supported_slots")
    return normalized


def _validate_numbering_version(value: str) -> str:
    normalized = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", normalized):
        raise LedgerError(
            "numbering_version must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
    return normalized


def _fingerprint_from_generation(generation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feed_digest": str(generation["feed_digest"]),
        "input_digest": str(generation["input_digest"]),
        "param_digest": str(generation["param_digest"]),
        "numbering_version": str(generation["numbering_version"]),
        "config_digest": sha256_hex(canonical_json_bytes(generation["config"])),
    }


def _replacement_reason(
    latest: Mapping[str, Any] | None,
    snapshot: Snapshot | None,
    fingerprint: Mapping[str, Any],
) -> str | None:
    if latest is None and snapshot is None:
        return None
    if latest is None:
        return "ledger_generation_missing"
    if snapshot is None:
        return "snapshot_missing"
    if int(snapshot.generation) != int(latest["generation_id"]):
        return "snapshot_version_mismatch"
    if _fingerprint_from_generation(latest) != dict(fingerprint):
        return "fingerprint_changed"
    return "generation_not_resumable"


def _unlink_if_exists(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
