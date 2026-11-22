"""Versioned ledger snapshots with verified wholesale restoration."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .content import (
    ledger_digest as compute_ledger_digest,
    normalise_event,
    normalise_visibility,
    snapshot_digest,
)
from .errors import BusinessError, SnapshotError, SpecError
from .models import (
    FORMAT_VERSION,
    SNAPSHOT_FORMAT_VERSION,
    Partition,
    PrefixReport,
    RestoreDiff,
    RestoreResult,
    Transaction,
    UnknownTransaction,
)
from .observability import StructuredLogger, utc_now_text
from .replay import ReplayMismatch, replay_events
from .store import Ledger
from .validation import parse_partitions, parse_transactions


@dataclass(frozen=True)
class ParsedSnapshot:
    path: str
    created_at: str
    snapshot_format_version: int
    ledger_format_version: int
    partition_set_version: int
    transaction_set_version: int
    event_cursor: int
    ledger_digest: str
    content_digest: str
    partitions: tuple[Partition, ...]
    transactions: tuple[Transaction, ...]
    events: tuple[dict[str, Any], ...]
    visibility: tuple[dict[str, Any], ...]
    prefixes: tuple[PrefixReport, ...]
    unknown_transactions: tuple[UnknownTransaction, ...]


@dataclass(frozen=True)
class SnapshotInfo:
    name: str
    path: str
    snapshot_format_version: int | None
    ledger_format_version: int | None
    partition_set_version: int | None
    transaction_set_version: int | None
    event_cursor: int | None
    ledger_digest: str | None
    content_digest: str | None
    readable: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "snapshot_format_version": self.snapshot_format_version,
            "ledger_format_version": self.ledger_format_version,
            "partition_set_version": self.partition_set_version,
            "transaction_set_version": self.transaction_set_version,
            "event_cursor": self.event_cursor,
            "ledger_digest": self.ledger_digest,
            "content_digest": self.content_digest,
            "readable": self.readable,
            "error": self.error,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _snapshot_error(message: str, **kwargs: Any) -> SnapshotError:
    details = kwargs.pop("details", {})
    return SnapshotError(message, details=details, **kwargs)


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not _is_int(value):
        raise _snapshot_error("快照整数字段非法", field=key)
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise _snapshot_error("快照字符串字段非法", field=key)
    return value


def _parse_prefixes(raw: Any) -> tuple[PrefixReport, ...]:
    if not isinstance(raw, list):
        raise _snapshot_error("快照 prefixes 必须为数组", field="prefixes")
    result: list[PrefixReport] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _snapshot_error("快照前缀项必须为对象", field=f"prefixes[{index}]")
        try:
            unknown = tuple(
                (int(interval[0]), int(interval[1]))
                for interval in item.get("unknown_intervals", [])
                if isinstance(interval, list) and len(interval) == 2
            )
            result.append(
                PrefixReport(
                    partition_id=str(item["partition_id"]),
                    partition_version=int(item["partition_version"]),
                    max_position=int(item["max_position"]),
                    prefix=int(item["prefix"]),
                    truncated_position=None
                    if item.get("truncated_position") is None
                    else int(item["truncated_position"]),
                    truncated_reason=item.get("truncated_reason"),
                    truncated_transaction_id=item.get("truncated_transaction_id"),
                    unknown_intervals=unknown,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _snapshot_error(
                "快照前缀项字段非法",
                field=f"prefixes[{index}]",
                details={"error": str(exc)},
            ) from exc
    return tuple(sorted(result, key=lambda item: item.partition_id))


def _parse_unknown(raw: Any) -> tuple[UnknownTransaction, ...]:
    if not isinstance(raw, list):
        raise _snapshot_error(
            "快照 unknown_transactions 必须为数组",
            field="unknown_transactions",
        )
    result: list[UnknownTransaction] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _snapshot_error(
                "未知事务项必须为对象", field=f"unknown_transactions[{index}]"
            )
        try:
            intervals = tuple(
                (int(interval[0]), int(interval[1]))
                for interval in item.get("intervals", [])
                if isinstance(interval, list) and len(interval) == 2
            )
            result.append(
                UnknownTransaction(
                    transaction_id=str(item["transaction_id"]),
                    partition_id=str(item["partition_id"]),
                    intervals=intervals,
                    reason=str(item.get("reason", "committing_after_forced_termination")),
                    requires_rerun=bool(item.get("requires_rerun", True)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _snapshot_error(
                "未知事务项字段非法",
                field=f"unknown_transactions[{index}]",
                details={"error": str(exc)},
            ) from exc
    return tuple(sorted(result, key=lambda item: (item.partition_id, item.transaction_id)))


def load_snapshot(path: str | os.PathLike[str]) -> ParsedSnapshot:
    snapshot_path = Path(path)
    try:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise _snapshot_error(
            "无法读取快照 JSON",
            field="file",
            details={"path": str(snapshot_path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise _snapshot_error("快照根节点必须为对象", field="snapshot")

    stored_content_digest = payload.get("content_digest")
    if not isinstance(stored_content_digest, str):
        raise _snapshot_error("快照缺少 content_digest", field="content_digest")
    actual_content_digest = snapshot_digest(payload)
    if stored_content_digest != actual_content_digest:
        raise _snapshot_error(
            "快照内容摘要不符",
            field="content_digest",
            details={"stored": stored_content_digest, "actual": actual_content_digest},
        )

    snapshot_format_version = _require_int(payload, "snapshot_format_version")
    if snapshot_format_version != SNAPSHOT_FORMAT_VERSION:
        raise _snapshot_error(
            "快照格式版本不受支持",
            field="snapshot_format_version",
            details={
                "supported": SNAPSHOT_FORMAT_VERSION,
                "stored": snapshot_format_version,
            },
        )
    ledger_format_version = _require_int(payload, "ledger_format_version")
    if ledger_format_version < 1 or ledger_format_version > FORMAT_VERSION:
        raise _snapshot_error(
            "快照账本格式版本不受支持",
            field="ledger_format_version",
            details={"supported": FORMAT_VERSION, "stored": ledger_format_version},
        )
    partition_set_version = _require_int(payload, "partition_set_version")
    transaction_set_version = _require_int(payload, "transaction_set_version")
    event_cursor = _require_int(payload, "event_cursor")
    stored_ledger_digest = _require_str(payload, "ledger_digest")
    created_at = _require_str(payload, "created_at")

    raw_partitions = payload.get("partitions")
    if not isinstance(raw_partitions, list):
        raise _snapshot_error("快照 partitions 必须为数组", field="partitions")
    try:
        partitions = parse_partitions({"partitions": raw_partitions})
    except BusinessError as exc:
        raise _snapshot_error(
            "快照分区声明非法",
            field=exc.field or "partitions",
            details=exc.to_dict(),
        ) from exc

    raw_transactions = payload.get("transactions")
    if not isinstance(raw_transactions, list):
        raise _snapshot_error("快照 transactions 必须为数组", field="transactions")
    try:
        transactions = parse_transactions({"transactions": raw_transactions}, partitions)
    except SpecError as exc:
        raise _snapshot_error(
            "快照事务目录非法",
            transaction_id=exc.transaction_id,
            field=exc.field or "transactions",
            details=exc.to_dict(),
        ) from exc
    except BusinessError as exc:
        raise _snapshot_error(
            "快照事务目录非法",
            transaction_id=exc.transaction_id,
            field=exc.field or "transactions",
            details=exc.to_dict(),
        ) from exc

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise _snapshot_error("快照 events 必须为数组", field="events")
    events: list[dict[str, Any]] = []
    for index, item in enumerate(raw_events):
        if not isinstance(item, dict):
            raise _snapshot_error("快照事件项必须为对象", field=f"events[{index}]")
        normalised = normalise_event(item)
        if not _is_int(normalised["event_seq"]):
            raise _snapshot_error(
                "快照事件序号非法", field=f"events[{index}].event_seq"
            )
        events.append(normalised)
    events.sort(key=lambda item: item["event_seq"])
    if len(events) != event_cursor:
        raise _snapshot_error(
            "快照事件游标与事件数量不一致",
            field="event_cursor",
            details={"event_cursor": event_cursor, "events": len(events)},
        )

    raw_visibility = payload.get("visibility")
    if not isinstance(raw_visibility, list):
        raise _snapshot_error("快照 visibility 必须为数组", field="visibility")
    visibility: list[dict[str, Any]] = []
    for index, item in enumerate(raw_visibility):
        if not isinstance(item, dict):
            raise _snapshot_error(
                "快照可见性项必须为对象", field=f"visibility[{index}]"
            )
        visibility.append(normalise_visibility(item))

    prefixes = _parse_prefixes(payload.get("prefixes"))
    unknown = _parse_unknown(payload.get("unknown_transactions"))

    embedded_digest = compute_ledger_digest(
        partitions,
        transactions,
        events,
        visibility,
        partition_set_version=partition_set_version,
        transaction_set_version=transaction_set_version,
        ledger_format_version=ledger_format_version,
    )
    if stored_ledger_digest != embedded_digest:
        raise _snapshot_error(
            "快照内账本内容摘要与序列化状态不符",
            field="ledger_digest",
            details={"stored": stored_ledger_digest, "actual": embedded_digest},
        )

    try:
        outcome = replay_events(
            partitions,
            transactions,
            events,
            visibility=visibility,
            expected_cursor=event_cursor,
        )
    except ReplayMismatch as mismatch:
        raise _snapshot_error(
            "快照事件流无法一致重放",
            transaction_id=mismatch.transaction_id,
            event_seq=mismatch.event_seq,
            field=mismatch.field,
            details={"expected": mismatch.expected, "actual": mismatch.actual},
        ) from mismatch

    if tuple(prefixes) != outcome.partitions:
        raise _snapshot_error(
            "快照可消费前缀与事件重放结论不符",
            field="prefixes",
        )
    if tuple(unknown) != outcome.unknown_transactions:
        raise _snapshot_error(
            "快照未知事务标注与事件重放结论不符",
            field="unknown_transactions",
        )

    transaction_map = {
        transaction.transaction_id: transaction for transaction in transactions
    }
    for index, item in enumerate(raw_transactions):
        if not isinstance(item, dict):
            continue
        transaction_id = item.get("transaction_id")
        expected_state = outcome.states.get(transaction_id)
        expected_seq = outcome.final_event_seq.get(transaction_id)
        if item.get("current_state") != expected_state:
            raise _snapshot_error(
                "快照事务最终状态与事件重放结论不符",
                transaction_id=transaction_id if isinstance(transaction_id, str) else None,
                event_seq=expected_seq,
                field=f"transactions[{index}].current_state",
                details={"stored": item.get("current_state"), "derived": expected_state},
            )
        if item.get("last_event_seq") != expected_seq:
            raise _snapshot_error(
                "快照事务最后事件序号与事件重放结论不符",
                transaction_id=transaction_id if isinstance(transaction_id, str) else None,
                event_seq=expected_seq,
                field=f"transactions[{index}].last_event_seq",
                details={"stored": item.get("last_event_seq"), "derived": expected_seq},
            )
        transaction = transaction_map.get(transaction_id)
        if transaction is not None and item.get("spec_digest") != transaction.spec_digest:
            raise _snapshot_error(
                "快照事务规格摘要不符",
                transaction_id=transaction.transaction_id,
                field=f"transactions[{index}].spec_digest",
            )

    return ParsedSnapshot(
        path=str(snapshot_path),
        created_at=created_at,
        snapshot_format_version=snapshot_format_version,
        ledger_format_version=ledger_format_version,
        partition_set_version=partition_set_version,
        transaction_set_version=transaction_set_version,
        event_cursor=event_cursor,
        ledger_digest=stored_ledger_digest,
        content_digest=stored_content_digest,
        partitions=partitions,
        transactions=transactions,
        events=tuple(events),
        visibility=tuple(visibility),
        prefixes=prefixes,
        unknown_transactions=unknown,
    )


def create_snapshot(ledger: Ledger, path: str | os.PathLike[str]) -> ParsedSnapshot:
    ledger.verify_records()
    conclusion = ledger.conclusion()
    transaction_states = ledger.latest_states()
    raw_transactions: list[dict[str, Any]] = []
    for transaction in ledger.transactions():
        item = transaction.as_spec_dict(include_digest=True)
        item["current_state"] = transaction_states.get(transaction.transaction_id)
        row = ledger.connection.execute(
            "SELECT last_event_seq FROM transactions WHERE transaction_id=?",
            (transaction.transaction_id,),
        ).fetchone()
        item["last_event_seq"] = None if row is None else row["last_event_seq"]
        raw_transactions.append(item)
    payload: dict[str, Any] = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "created_at": utc_now_text(),
        "ledger_format_version": conclusion.ledger_format_version,
        "partition_set_version": conclusion.partition_set_version,
        "transaction_set_version": conclusion.transaction_set_version,
        "event_cursor": conclusion.event_cursor,
        "ledger_digest": conclusion.ledger_digest,
        "partitions": [partition.as_dict() for partition in ledger.partitions()],
        "transactions": raw_transactions,
        "events": [normalise_event(event) for event in ledger.events()],
        "visibility": [normalise_visibility(row) for row in ledger.visibility()],
        "prefixes": [report.as_dict() for report in conclusion.partitions],
        "unknown_transactions": [
            item.as_dict() for item in conclusion.unknown_transactions
        ],
    }
    payload["content_digest"] = snapshot_digest(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return load_snapshot(destination)


def snapshots_directory(workdir: str | os.PathLike[str]) -> Path:
    return Path(workdir) / "snapshots"


def backups_directory(workdir: str | os.PathLike[str]) -> Path:
    return Path(workdir) / "backups"


def next_snapshot_path(workdir: str | os.PathLike[str]) -> Path:
    directory = snapshots_directory(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    maximum = 0
    pattern = re.compile(r"^snapshot-(\d+)\.json$")
    for path in directory.glob("snapshot-*.json"):
        match = pattern.match(path.name)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return directory / f"snapshot-{maximum + 1:04d}.json"


def list_snapshots(workdir: str | os.PathLike[str]) -> tuple[SnapshotInfo, ...]:
    directory = snapshots_directory(workdir)
    if not directory.exists():
        return ()
    result: list[SnapshotInfo] = []
    for path in sorted(directory.glob("snapshot-*.json"), key=lambda item: item.name):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            result.append(
                SnapshotInfo(
                    name=path.name,
                    path=str(path),
                    snapshot_format_version=payload.get("snapshot_format_version")
                    if _is_int(payload.get("snapshot_format_version"))
                    else None,
                    ledger_format_version=payload.get("ledger_format_version")
                    if _is_int(payload.get("ledger_format_version"))
                    else None,
                    partition_set_version=payload.get("partition_set_version")
                    if _is_int(payload.get("partition_set_version"))
                    else None,
                    transaction_set_version=payload.get("transaction_set_version")
                    if _is_int(payload.get("transaction_set_version"))
                    else None,
                    event_cursor=payload.get("event_cursor")
                    if _is_int(payload.get("event_cursor"))
                    else None,
                    ledger_digest=payload.get("ledger_digest")
                    if isinstance(payload.get("ledger_digest"), str)
                    else None,
                    content_digest=payload.get("content_digest")
                    if isinstance(payload.get("content_digest"), str)
                    else None,
                    readable=True,
                )
            )
        except (OSError, ValueError) as exc:
            result.append(
                SnapshotInfo(
                    name=path.name,
                    path=str(path),
                    snapshot_format_version=None,
                    ledger_format_version=None,
                    partition_set_version=None,
                    transaction_set_version=None,
                    event_cursor=None,
                    ledger_digest=None,
                    content_digest=None,
                    readable=False,
                    error=str(exc),
                )
            )
    return tuple(result)


def _state_snapshot(ledger: Ledger) -> tuple[dict[str, str | None], dict[str, Any]]:
    states = ledger.latest_states()
    states = {transaction.transaction_id: states.get(transaction.transaction_id) for transaction in ledger.transactions()}
    versions = {
        "partition_set_version": ledger._meta_int("partition_set_version"),
        "transaction_set_version": ledger._meta_int("transaction_set_version"),
        "ledger_format_version": ledger._meta_int("ledger_format_version"),
        "event_cursor": ledger._event_cursor(),
    }
    return states, versions


def _restore_diff(ledger: Ledger, snapshot: ParsedSnapshot) -> RestoreDiff:
    current_partitions = {partition.partition_id: partition for partition in ledger.partitions()}
    snapshot_partitions = {partition.partition_id: partition for partition in snapshot.partitions}
    current_transactions = {transaction.transaction_id: transaction for transaction in ledger.transactions()}
    snapshot_transactions = {transaction.transaction_id: transaction for transaction in snapshot.transactions}
    current_states, current_versions = _state_snapshot(ledger)
    snapshot_states = {transaction.transaction_id: None for transaction in snapshot.transactions}
    snapshot_states.update({key: value for key, value in _snapshot_states(snapshot).items()})

    version_changes: list[dict[str, Any]] = []
    for key in (
        "ledger_format_version",
        "partition_set_version",
        "transaction_set_version",
        "event_cursor",
    ):
        snapshot_value = getattr(snapshot, key)
        if current_versions[key] != snapshot_value:
            version_changes.append(
                {
                    "kind": key,
                    "old": current_versions[key],
                    "new": snapshot_value,
                }
            )
    for partition_id in sorted(set(current_partitions) & set(snapshot_partitions)):
        old = current_partitions[partition_id].partition_version
        new = snapshot_partitions[partition_id].partition_version
        if old != new:
            version_changes.append(
                {
                    "kind": "partition_version",
                    "id": partition_id,
                    "old": old,
                    "new": new,
                }
            )
    version_changes.sort(
        key=lambda item: (item["kind"], str(item.get("id", ""))),
    )

    state_changes: list[dict[str, Any]] = []
    for transaction_id in sorted(set(current_transactions) & set(snapshot_transactions)):
        old_state = current_states.get(transaction_id)
        new_state = snapshot_states.get(transaction_id)
        if old_state != new_state:
            state_changes.append(
                {
                    "transaction_id": transaction_id,
                    "old_state": old_state,
                    "new_state": new_state,
                }
            )
    return RestoreDiff(
        partition_added=tuple(sorted(set(snapshot_partitions) - set(current_partitions))),
        partition_removed=tuple(sorted(set(current_partitions) - set(snapshot_partitions))),
        transaction_added=tuple(sorted(set(snapshot_transactions) - set(current_transactions))),
        transaction_removed=tuple(sorted(set(current_transactions) - set(snapshot_transactions))),
        version_changes=tuple(version_changes),
        transaction_state_changes=tuple(state_changes),
    )


def _snapshot_states(snapshot: ParsedSnapshot) -> dict[str, str]:
    states: dict[str, str] = {}
    for event in snapshot.events:
        transaction_id = event["transaction_id"]
        state = event["state"]
        if isinstance(transaction_id, str) and isinstance(state, str):
            states[transaction_id] = state
    return states


def _backup_name(digest: str) -> str:
    timestamp = utc_now_text().replace("-", "").replace(":", "").replace(".", "")
    suffix = digest.removeprefix("sha256:")[:16]
    return f"ledger-{timestamp}-{suffix}.sqlite"


def restore_snapshot(
    ledger: Ledger,
    path: str | os.PathLike[str],
    *,
    logger: StructuredLogger | None = None,
) -> RestoreResult:
    active_logger = logger or StructuredLogger(ledger.path.parent / "logs" / "business.jsonl")
    try:
        snapshot = load_snapshot(path)
        prior_digest = ledger.computed_ledger_digest()
        diff = _restore_diff(ledger, snapshot)
        backup_directory = backups_directory(ledger.path.parent)
        backup_path = ledger.backup_to(backup_directory / _backup_name(prior_digest))
        ledger.replace_all_state(
            snapshot.partitions,
            snapshot.transactions,
            snapshot.events,
            snapshot.visibility,
            partition_set_version=snapshot.partition_set_version,
            transaction_set_version=snapshot.transaction_set_version,
            ledger_format_version=snapshot.ledger_format_version,
        )
        ledger.verify_records()
        restored_digest = ledger.computed_ledger_digest()
        if restored_digest != snapshot.ledger_digest:
            raise _snapshot_error(
                "快照恢复后账本摘要不一致",
                field="ledger_digest",
                details={"snapshot": snapshot.ledger_digest, "restored": restored_digest},
            )
        ledger.recover()
        manifest = {
            "restored_at": utc_now_text(),
            "snapshot_path": str(Path(path)),
            "snapshot_content_digest": snapshot.content_digest,
            "snapshot_ledger_digest": snapshot.ledger_digest,
            "prior_ledger_digest": prior_digest,
            "backup_path": backup_path,
            "diff": diff.as_dict(),
        }
        manifest_path = backup_directory / (
            "restore-" + Path(backup_path).stem.removeprefix("ledger-") + ".json"
        )
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return RestoreResult(
            restored=True,
            replaced=True,
            snapshot_path=str(Path(path)),
            backup_path=backup_path,
            snapshot_ledger_digest=snapshot.ledger_digest,
            prior_ledger_digest=prior_digest,
            diff=diff,
            event_cursor=ledger._event_cursor(),
            partitions=ledger.query_prefixes(),
            format_version_changed=snapshot.ledger_format_version != FORMAT_VERSION,
        )
    except BusinessError as error:
        active_logger.write_error(error)
        raise


def validate_snapshot_file(
    path: str | os.PathLike[str],
    *,
    logger: StructuredLogger | None = None,
) -> ParsedSnapshot:
    try:
        return load_snapshot(path)
    except BusinessError as error:
        if logger is not None:
            logger.write_error(error)
        raise
