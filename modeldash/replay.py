"""Snapshot/ledger consistency replay and first-mismatch reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import LedgerError, WorkbenchError
from .inference import aggregate_output_summaries
from .nodes import SocketDispatcher
from .receipts import Receipt, ReceiptBook, build_node_untrust
from .spec import Spec
from .storage import Ledger
from .util import digest_json


@dataclass
class ReplaySummary:
    session_id: str
    snapshot_id: str | None
    redispatch_count: int
    remaining_task_ids: list[str]
    checks: int
    result_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "code": "REPLAY_OK",
            "session_id": self.session_id,
            "snapshot_id": self.snapshot_id,
            "redispatch_count": self.redispatch_count,
            "remaining_task_ids": self.remaining_task_ids,
            "checks": self.checks,
            "result_digest": self.result_digest,
        }


class ReplayVerifier:
    def __init__(
        self,
        spec: Spec,
        ledger: Ledger,
        session_id: str,
        *,
        snapshot_id: str | None = None,
        dispatcher: SocketDispatcher | None = None,
        redispatch: bool = False,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.session_id = session_id
        self.snapshot_id = snapshot_id
        self.dispatcher = dispatcher
        self.redispatch = redispatch
        self.checks = 0

    def _mismatch(
        self,
        *,
        task_id: str,
        node_id: str,
        field: str,
        expected: Any,
        actual: Any,
        detail: str,
    ) -> None:
        raise WorkbenchError(
            "E_REPLAY",
            field,
            detail,
            {
                "first_task_id": task_id,
                "node_id": node_id,
                "field": field,
                "expected": expected,
                "actual": actual,
            },
        )

    def verify(self) -> ReplaySummary:
        rounds = self.ledger.load_rounds(self.session_id)
        if not rounds:
            raise LedgerError("E_REPLAY", "rounds", "session has no committed rounds")
        previous_cursor = 0
        previous_round = 0
        for item in rounds:
            self.checks += 1
            if item["round_seq"] <= previous_round:
                self._mismatch(
                    task_id="",
                    node_id="",
                    field=f"rounds[{item['round_seq']}].round_seq",
                    expected=previous_round + 1,
                    actual=item["round_seq"],
                    detail="round sequence is not strictly increasing",
                )
            if item["cursor"] <= previous_cursor:
                self._mismatch(
                    task_id="",
                    node_id="",
                    field=f"rounds[{item['round_seq']}].cursor",
                    expected=previous_cursor + 1,
                    actual=item["cursor"],
                    detail="progress cursor is not strictly increasing",
                )
            previous_round = int(item["round_seq"])
            previous_cursor = int(item["cursor"])

        final_state = rounds[-1]["payload"]["state"]
        snapshot_state: dict[str, Any] = final_state
        if self.snapshot_id:
            result = self.ledger.load_snapshot(self.snapshot_id, self.spec)
            if result.action != "RESTORED" or result.state is None:
                raise WorkbenchError(
                    "E_REPLAY",
                    "snapshot",
                    f"snapshot {self.snapshot_id} is not restorable: {result.reason}",
                )
            snapshot_state = result.state

        self._verify_task_outputs(final_state)
        self._verify_wait_totals(final_state)
        self._verify_preemption_sequence(final_state)
        self._verify_receipts_and_untrust(final_state)
        remaining = self._remaining_tasks(snapshot_state)
        redispatch_count = 0
        if self.dispatcher is not None and self.redispatch:
            redispatch_count = self._redispatch_receipts(final_state)

        result_digest = digest_json(
            {
                "session_id": self.session_id,
                "rounds": [item["content_digest"] for item in rounds],
                "tasks": [
                    {
                        "task_id": task_id,
                        "state": final_state["tasks"][task_id]["state"],
                        "output_summary": final_state["tasks"][task_id]["output_summary"],
                    }
                    for task_id in sorted(final_state["tasks"])
                ],
                "node_untrust": final_state.get("node_untrust", []),
                "redispatched": redispatch_count,
            }
        )
        return ReplaySummary(
            session_id=self.session_id,
            snapshot_id=self.snapshot_id,
            redispatch_count=redispatch_count,
            remaining_task_ids=remaining,
            checks=self.checks,
            result_digest=result_digest,
        )

    def _verify_task_outputs(self, state: dict[str, Any]) -> None:
        for task_id in sorted(state.get("tasks", {})):
            task = state["tasks"][task_id]
            shards = sorted(task.get("completed_shards", []), key=lambda item: int(item["shard_index"]))
            expected_index = 0
            for shard in shards:
                self.checks += 1
                if int(shard["shard_index"]) != expected_index:
                    self._mismatch(
                        task_id=task_id,
                        node_id=str(task["node_id"]),
                        field="completed_shards.shard_index",
                        expected=expected_index,
                        actual=shard["shard_index"],
                        detail="completed shards are not contiguous",
                    )
                expected_index += 1
            if task["state"] == "DONE":
                aggregate = aggregate_output_summaries([str(item["output_summary"]) for item in shards])
                self.checks += 1
                if aggregate != task.get("output_summary"):
                    self._mismatch(
                        task_id=task_id,
                        node_id=str(task["node_id"]),
                        field="output_summary",
                        expected=task.get("output_summary"),
                        actual=aggregate,
                        detail="aggregate output summary does not match completed shard summaries",
                    )
                if len(shards) != int(task["total_shards"]):
                    self._mismatch(
                        task_id=task_id,
                        node_id=str(task["node_id"]),
                        field="completed_shards",
                        expected=int(task["total_shards"]),
                        actual=len(shards),
                        detail="DONE task does not contain every shard",
                    )

    def _verify_wait_totals(self, state: dict[str, Any]) -> None:
        for task_id in sorted(state.get("tasks", {})):
            task = state["tasks"][task_id]
            recomputed = sum(int(window.get("duration_ms", 0)) for window in task.get("wait_windows", []))
            self.checks += 1
            if recomputed != int(task.get("wait_ms", 0)):
                self._mismatch(
                    task_id=task_id,
                    node_id=str(task["node_id"]),
                    field="wait_ms",
                    expected=int(task.get("wait_ms", 0)),
                    actual=recomputed,
                    detail="recorded quota wait does not equal its monotonic-clock windows",
                )

    def _verify_preemption_sequence(self, state: dict[str, Any]) -> None:
        preemptions = sorted(state.get("preemptions", []), key=lambda item: str(item["preemption_id"]))
        for item in preemptions:
            self.checks += 1
            if item.get("status") in {"SUSPENDED", "SUSPENDED_AT_FAILURE"}:
                if int(item["suspended_shard"]) < 0:
                    self._mismatch(
                        task_id=str(item["task_id"]),
                        node_id="",
                        field="suspended_shard",
                        expected=">= 0 at a shard boundary",
                        actual=item["suspended_shard"],
                        detail="suspended checkpoint is invalid",
                    )
        for task_id in sorted(state.get("tasks", {})):
            task = state["tasks"][task_id]
            indices = [int(item["shard_index"]) for item in task.get("completed_shards", [])]
            self.checks += 1
            if len(indices) != len(set(indices)):
                self._mismatch(
                    task_id=task_id,
                    node_id=str(task["node_id"]),
                    field="completed_shards",
                    expected="unique shard indices",
                    actual=indices,
                    detail="a completed shard was duplicated",
                )

    def _verify_receipts_and_untrust(self, state: dict[str, Any]) -> None:
        book = ReceiptBook()
        for item in state.get("receipts", []):
            book.add(Receipt.from_dict(item))
        audited = book.audit(self.spec.current_version)
        expected_receipts = {str(item["receipt_key"]): item for item in state.get("receipts", [])}
        for receipt in audited:
            self.checks += 1
            expected = expected_receipts.get(receipt.receipt_key)
            if expected is None:
                self._mismatch(
                    task_id=receipt.task_id,
                    node_id=receipt.node_id,
                    field="receipts.receipt_key",
                    expected=sorted(expected_receipts),
                    actual=receipt.receipt_key,
                    detail="replayed receipt is absent from the ledger",
                )
            for field in ("status", "untrust_score", "decay_count", "output_summary"):
                self.checks += 1
                if expected.get(field) != getattr(receipt, field):
                    self._mismatch(
                        task_id=receipt.task_id,
                        node_id=receipt.node_id,
                        field=f"receipts.{field}",
                        expected=expected.get(field),
                        actual=getattr(receipt, field),
                        detail="receipt decay or output did not replay identically",
                    )
        recomputed_nodes = build_node_untrust(
            audited, [node.node_id for node in self.spec.nodes], self.spec.current_version
        )
        self.checks += 1
        if recomputed_nodes != state.get("node_untrust", []):
            expected = state.get("node_untrust", [])
            actual = recomputed_nodes
            for index in range(max(len(expected), len(actual))):
                exp = expected[index] if index < len(expected) else None
                act = actual[index] if index < len(actual) else None
                if exp != act:
                    self._mismatch(
                        task_id="",
                        node_id=str((exp or act or {}).get("node_id", "")),
                        field=f"node_untrust[{index}]",
                        expected=exp,
                        actual=act,
                        detail="per-node untrust ranking did not replay identically",
                    )

    def _remaining_tasks(self, state: dict[str, Any]) -> list[str]:
        remaining = []
        for task_id in sorted(state.get("tasks", {})):
            task = state["tasks"][task_id]
            if task["state"] not in {"DONE", "EXHAUSTED", "FAILED"}:
                remaining.append(task_id)
            elif task.get("unknown_shards"):
                remaining.append(task_id)
        return remaining

    def _redispatch_receipts(self, state: dict[str, Any]) -> int:
        assert self.dispatcher is not None
        count = 0
        for item in sorted(state.get("receipts", []), key=lambda row: str(row["receipt_key"])):
            task_id = str(item["task_id"])
            try:
                definition = self.spec.task(task_id)
            except KeyError:
                continue
            request_id = "replay-" + digest_json({"receipt": item["receipt_key"], "session": self.session_id})[:28]
            result = self.dispatcher.dispatch(definition, int(item["shard_index"]), request_id)
            count += 1
            self.checks += 1
            if result.output_summary != item["output_summary"]:
                self._mismatch(
                    task_id=task_id,
                    node_id=str(item["node_id"]),
                    field="receipts.output_summary",
                    expected=item["output_summary"],
                    actual=result.output_summary,
                    detail="live re-dispatch produced a different output summary",
                )
            self.checks += 1
            if result.input_digest != item["input_digest"]:
                self._mismatch(
                    task_id=task_id,
                    node_id=str(item["node_id"]),
                    field="receipts.input_digest",
                    expected=item["input_digest"],
                    actual=result.input_digest,
                    detail="live re-dispatch produced a different input digest",
                )
        return count
