"""Versioned session snapshots, whole-state replacement, and replay checks."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, sha256_digest
from .errors import SnapshotError
from .loader import validate_spec_object
from .model import (
    Spec,
    budget_version,
    build_index,
    chain_digest,
    conclusion_digest,
    read_point_set_version,
    transaction_set_version,
)
from .visibility import QueryResult, run_per_query

SNAPSHOT_FORMAT = "mvcc-session-snapshot"
SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SessionIdentity:
    transaction_set_version: str
    chain_digest: str
    budget_version: str
    read_point_set_version: str
    conclusion_digest: str
    ledger_cursor: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_set_version": self.transaction_set_version,
            "chain_digest": self.chain_digest,
            "budget_version": self.budget_version,
            "read_point_set_version": self.read_point_set_version,
            "conclusion_digest": self.conclusion_digest,
            "ledger_cursor": self.ledger_cursor,
        }


@dataclass(frozen=True)
class SessionSnapshot:
    snapshot_id: str
    created_wall_ms: int
    identity: SessionIdentity
    spec: dict[str, Any]
    plan: dict[str, Any]
    results: tuple[dict[str, Any], ...]
    format_version: int = SNAPSHOT_FORMAT_VERSION

    def payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": SNAPSHOT_FORMAT,
            "format_version": self.format_version,
            "snapshot_id": self.snapshot_id,
            "created_wall_ms": self.created_wall_ms,
            "identity": self.identity.to_dict(),
            "spec": self.spec,
            "plan": self.plan,
            "results": [dict(item) for item in sorted(self.results, key=lambda x: x["query_id"])],
        }

    @property
    def content_digest(self) -> str:
        return sha256_digest(self.payload_without_digest())

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload_without_digest()
        payload["content_digest"] = self.content_digest
        return payload


@dataclass(frozen=True)
class ReplayVerification:
    ok: bool
    expected_digest: str
    actual_digest: str
    query_count: int
    code: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "query_count": self.query_count,
        }


def session_identity(
    spec: Spec,
    results: Iterable[QueryResult | Mapping[str, Any]],
    *,
    ledger_cursor: int,
) -> SessionIdentity:
    if type(ledger_cursor) is not int or ledger_cursor < 0:
        raise SnapshotError(
            "ledger_cursor 必须是非负整数",
            details={"ledger_cursor": ledger_cursor},
        )
    return SessionIdentity(
        transaction_set_version=transaction_set_version(spec),
        chain_digest=chain_digest(spec),
        budget_version=budget_version(spec),
        read_point_set_version=read_point_set_version(spec),
        conclusion_digest=conclusion_digest(results),
        ledger_cursor=ledger_cursor,
    )


def create_snapshot(
    spec: Spec,
    plan: Mapping[str, Any],
    results: Iterable[QueryResult],
    *,
    ledger_cursor: int,
    snapshot_id: str | None = None,
) -> SessionSnapshot:
    result_tuple = tuple(sorted(results, key=lambda item: item.query_id))
    identity = session_identity(spec, result_tuple, ledger_cursor=ledger_cursor)
    timestamp = int(time.time() * 1000)
    return SessionSnapshot(
        snapshot_id=snapshot_id or f"snapshot-{timestamp}",
        created_wall_ms=timestamp,
        identity=identity,
        spec=spec.to_dict(),
        plan=dict(plan),
        results=tuple(item.to_dict() for item in result_tuple),
    )


def save_snapshot(path: str | Path, snapshot: SessionSnapshot) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            os.chmod(destination, 0o644)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {
        "path": str(destination),
        "content_digest": snapshot.content_digest,
        "snapshot_id": snapshot.snapshot_id,
        **(snapshot.identity.to_dict()),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(
            f"快照读取失败: {exc}",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(value, dict):
        raise SnapshotError("快照根节点必须是 JSON 对象", details={"path": str(path)})
    return value


def _parse_identity(value: Any) -> SessionIdentity:
    if not isinstance(value, dict):
        raise SnapshotError("快照 identity 必须是对象")
    required = {
        "transaction_set_version",
        "chain_digest",
        "budget_version",
        "read_point_set_version",
        "conclusion_digest",
        "ledger_cursor",
    }
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise SnapshotError(
            "快照 identity 字段不完整",
            details={"missing": missing, "extra": extra},
        )
    for key in sorted(required - {"ledger_cursor"}):
        if not isinstance(value[key], str) or not value[key]:
            raise SnapshotError(
                f"快照 identity.{key} 必须是非空字符串",
                details={"field": key},
            )
    cursor = value["ledger_cursor"]
    if type(cursor) is not int or cursor < 0:
        raise SnapshotError(
            "快照 identity.ledger_cursor 必须是非负整数",
            details={"field": "ledger_cursor"},
        )
    return SessionIdentity(
        transaction_set_version=value["transaction_set_version"],
        chain_digest=value["chain_digest"],
        budget_version=value["budget_version"],
        read_point_set_version=value["read_point_set_version"],
        conclusion_digest=value["conclusion_digest"],
        ledger_cursor=cursor,
    )


def load_snapshot(path: str | Path) -> SessionSnapshot:
    source = Path(path)
    value = _load_json(source)
    if value.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotError(
            "快照 format 非法",
            details={"field": "format", "value": value.get("format")},
        )
    format_version = value.get("format_version")
    if type(format_version) is not int or format_version != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotError(
            "快照格式版本不兼容",
            details={
                "field": "format_version",
                "expected": SNAPSHOT_FORMAT_VERSION,
                "actual": format_version,
            },
        )
    if not isinstance(value.get("snapshot_id"), str) or not value["snapshot_id"]:
        raise SnapshotError("快照 snapshot_id 非法")
    if type(value.get("created_wall_ms")) is not int:
        raise SnapshotError("快照 created_wall_ms 必须是整数")
    identity = _parse_identity(value.get("identity"))
    if not isinstance(value.get("spec"), dict):
        raise SnapshotError("快照 spec 必须是对象")
    if not isinstance(value.get("plan"), dict):
        raise SnapshotError("快照 plan 必须是对象")
    results = value.get("results")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise SnapshotError("快照 results 必须是对象数组")
    result_ids = [item.get("query_id") for item in results]
    if not all(isinstance(item, str) and item for item in result_ids):
        raise SnapshotError("快照 results 的 query_id 非法")
    if len(result_ids) != len(set(result_ids)):
        raise SnapshotError("快照 results 的 query_id 重复")

    supplied_digest = value.get("content_digest")
    if not isinstance(supplied_digest, str):
        raise SnapshotError("快照缺少 content_digest")
    provisional = SessionSnapshot(
        snapshot_id=value["snapshot_id"],
        created_wall_ms=value["created_wall_ms"],
        identity=identity,
        spec=value["spec"],
        plan=value["plan"],
        results=tuple(sorted(results, key=lambda x: x["query_id"])),
        format_version=format_version,
    )
    actual_digest = provisional.content_digest
    if supplied_digest != actual_digest:
        raise SnapshotError(
            "快照内容摘要不一致",
            details={
                "anomaly": "SNAPSHOT_DIGEST_MISMATCH",
                "expected": supplied_digest,
                "actual": actual_digest,
                "path": str(source),
            },
        )
    try:
        validate_spec_object(value["spec"], source_path=str(source) + ":spec")
    except Exception as exc:
        raise SnapshotError(
            "快照中的规格无法通过校验",
            details={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "business_code": getattr(exc, "code", None),
            },
        ) from exc
    return provisional


def identity_mismatches(
    expected: SessionIdentity,
    actual: SessionIdentity,
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    left = expected.to_dict()
    right = actual.to_dict()
    for key in sorted(left):
        if left[key] != right[key]:
            result.append({"field": key, "expected": left[key], "actual": right[key]})
    return tuple(result)


def verify_snapshot_replay(snapshot: SessionSnapshot) -> ReplayVerification:
    spec = validate_spec_object(snapshot.spec, source_path=f"snapshot:{snapshot.snapshot_id}")
    index = build_index(spec)
    results = run_per_query(index)
    actual_digest = conclusion_digest(results)
    expected_digest = snapshot.identity.conclusion_digest
    if actual_digest != expected_digest:
        raise SnapshotError(
            "快照重放结论摘要不一致",
            details={
                "snapshot_id": snapshot.snapshot_id,
                "expected": expected_digest,
                "actual": actual_digest,
                "query_count": len(results),
            },
        )
    stored_ids = tuple(sorted(item["query_id"] for item in snapshot.results))
    replayed_ids = tuple(sorted(item.query_id for item in results))
    if stored_ids != replayed_ids:
        raise SnapshotError(
            "快照重放查询集合不一致",
            details={
                "snapshot_id": snapshot.snapshot_id,
                "expected": list(stored_ids),
                "actual": list(replayed_ids),
            },
        )
    return ReplayVerification(
        ok=True,
        expected_digest=expected_digest,
        actual_digest=actual_digest,
        query_count=len(results),
    )


def restore_snapshot(
    path: str | Path,
    *,
    expected_identity: SessionIdentity | None = None,
    force_replace: bool = False,
    verify_replay: bool = True,
) -> tuple[SessionSnapshot, ReplayVerification | None]:
    snapshot = load_snapshot(path)
    mismatches = (
        identity_mismatches(expected_identity, snapshot.identity)
        if expected_identity is not None
        else ()
    )
    if mismatches and not force_replace:
        raise SnapshotError(
            "快照版本或摘要与当前会话不一致，必须整体替换",
            details={
                "anomaly": "SESSION_IDENTITY_MISMATCH",
                "action": "REPLACE_ALL",
                "mismatches": list(mismatches),
                "snapshot_id": snapshot.snapshot_id,
            },
        )
    replay = verify_snapshot_replay(snapshot) if verify_replay else None
    return snapshot, replay


def snapshot_identity_json(snapshot: SessionSnapshot) -> str:
    return canonical_json(snapshot.identity.to_dict())
