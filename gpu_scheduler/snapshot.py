from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import SNAPSHOT_FORMAT_VERSION
from .errors import SnapshotDigestError
from .hashing import canonical_text, digest_obj, sha256_hex
from .ledger import Ledger
from .spec import Spec


def outcome_summary(receipt_row: dict[str, Any]) -> dict[str, Any]:
    receipt = json.loads(receipt_row["canonical_receipt_json"])
    resource = receipt.get("resource_usage", {})
    summary = {
        "shard_id": receipt["shard_id"],
        "outcome_status": receipt["status"],
        "receipt_digest": receipt["receipt_digest"],
        "checksum": (receipt.get("compute") or {}).get("checksum"),
        "resource_digest": digest_obj(resource),
        "resource_usage": resource,
        "round_index": receipt_row["round_index"],
    }
    summary["outcome_digest"] = digest_obj(
        {
            "shard_id": summary["shard_id"],
            "outcome_status": summary["outcome_status"],
            "receipt_digest": summary["receipt_digest"],
            "checksum": summary["checksum"],
            "resource_digest": summary["resource_digest"],
        }
    )
    return summary


def replay_ledger(ledger: Ledger) -> dict[str, Any]:
    verification = ledger.verify_digests()
    rounds = ledger.list_rounds()
    watermarks: list[dict[str, Any]] = []
    preemption_sequence: list[dict[str, Any]] = []
    propagation_sets: list[dict[str, Any]] = []
    for row in rounds:
        payload = json.loads(row["payload_json"])
        watermarks.append({"round_index": row["round_index"], "watermarks": payload["watermarks"]})
        for preemption in payload.get("preemptions", []):
            preemption_sequence.append({"round_index": row["round_index"], **preemption})
        propagations = payload.get("propagations")
        if propagations:
            for propagation in propagations:
                propagation_sets.append({"round_index": row["round_index"], **propagation})
        elif payload.get("propagation"):
            propagation_sets.append({"round_index": row["round_index"], **payload["propagation"]})

    outcomes = [outcome_summary(row) for row in sorted(ledger.latest_receipts().values(), key=lambda r: r["shard_id"])]
    unknown_rows = ledger.conn.execute(
        """
        SELECT shard_id, MAX(round_index) AS round_index
        FROM shard_events WHERE phase='unknown' GROUP BY shard_id ORDER BY shard_id
        """
    ).fetchall()
    unknown = [dict(row) for row in unknown_rows]
    return {
        "verification": verification,
        "ledger_cursor": ledger.head_round,
        "ledger_head_digest": ledger.head_chain_digest,
        "watermarks": watermarks,
        "preemption_sequence": preemption_sequence,
        "propagation_sets": propagation_sets,
        "shard_outcomes": outcomes,
        "unknown_shards": unknown,
    }


def build_snapshot_content(spec: Spec, ledger: Ledger, *, session_id: str) -> dict[str, Any]:
    replay = replay_ledger(ledger)
    return {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "integer_format_version": 1,
        "session_id": session_id,
        "spec_digest": spec.spec_digest,
        "spec": spec.raw,
        "pool_version": spec.pool_version,
        "task_version": spec.task_version,
        "scale_version": spec.scale_version,
        "ledger_cursor": replay["ledger_cursor"],
        "ledger_head_digest": replay["ledger_head_digest"],
        "round_watermarks": replay["watermarks"],
        "preemption_sequence": replay["preemption_sequence"],
        "propagation_sets": replay["propagation_sets"],
        "shard_outcomes": replay["shard_outcomes"],
        "unknown_shards": replay["unknown_shards"],
    }


def write_snapshot(path: Path | str, content: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content_digest = digest_obj(content)
    outer = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "content_sha256": content_digest,
        "content": content,
    }
    path.write_text(canonical_text(outer) + "\n", encoding="utf-8")
    return path


def read_snapshot(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        outer = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SnapshotDigestError(f"snapshot not found: {path}", locator={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise SnapshotDigestError(
            f"snapshot JSON invalid: {exc.msg}",
            locator={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(outer, dict) or outer.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotDigestError(
            "unsupported snapshot format version",
            locator={"path": str(path), "field": "snapshot_format_version"},
        )
    content = outer.get("content")
    actual_digest = outer.get("content_sha256")
    if not isinstance(content, dict) or not isinstance(actual_digest, str):
        raise SnapshotDigestError("snapshot content or digest missing", locator={"field": "content/content_sha256"})
    recomputed = digest_obj(content)
    if actual_digest != recomputed:
        raise SnapshotDigestError(
            "snapshot content summary mismatch",
            locator={"path": str(path), "field": "content_sha256", "expected": recomputed, "actual": actual_digest},
        )
    return outer


def compare_outcomes(snapshot_outcomes: list[dict[str, Any]], replayed_outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    old = {item["shard_id"]: item for item in snapshot_outcomes}
    new = {item["shard_id"]: item for item in replayed_outcomes}
    fields = ("outcome_status", "receipt_digest", "outcome_digest", "resource_digest", "checksum")
    for shard_id in sorted(set(old) | set(new)):
        old_item = old.get(shard_id)
        new_item = new.get(shard_id)
        if old_item is None or new_item is None:
            return {
                "match": False,
                "first_difference": {
                    "shard_id": shard_id,
                    "field": "presence",
                    "expected": "present" if old_item is not None else "absent",
                    "actual": "present" if new_item is not None else "absent",
                },
            }
        for field in fields:
            if old_item.get(field) != new_item.get(field):
                return {
                    "match": False,
                    "first_difference": {
                        "shard_id": shard_id,
                        "field": field,
                        "expected": old_item.get(field),
                        "actual": new_item.get(field),
                    },
                }
    return {"match": True, "compared_shards": len(old)}


def restore_and_verify(
    snapshot_path: Path | str,
    workdir: Path | str,
    current_spec: Spec,
    *,
    restored_spec_path: Path | str | None = None,
) -> dict[str, Any]:
    outer = read_snapshot(snapshot_path)
    content = outer["content"]
    ledger = Ledger(Path(workdir) / "ledger.sqlite3")
    try:
        replay = replay_ledger(ledger)
        replacement = content.get("spec")
        declaration_matches = content.get("spec_digest") == current_spec.spec_digest
        version_matches = (
            content.get("pool_version", {}).get("digest") == current_spec.pool_version["digest"]
            and content.get("task_version", {}).get("digest") == current_spec.task_version["digest"]
            and content.get("scale_version", {}).get("digest") == current_spec.scale_version["digest"]
        )
        result: dict[str, Any] = {
            "merge_mode": "match_no_merge_needed" if declaration_matches and version_matches else "whole_replace",
            "declaration_matches": declaration_matches,
            "version_matches": version_matches,
            "replayed_rounds": replay["verification"]["rounds"],
        }
        if not declaration_matches and isinstance(replacement, dict):
            target = Path(restored_spec_path or Path(workdir) / "restored-spec.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(canonical_text(replacement) + "\n", encoding="utf-8")
            result["restored_spec_path"] = str(target)
            result["replaced_versions"] = {
                "pool_version": content.get("pool_version"),
                "task_version": content.get("task_version"),
                "scale_version": content.get("scale_version"),
            }
        if int(content.get("ledger_cursor", -1)) != replay["ledger_cursor"]:
            raise SnapshotDigestError(
                "snapshot ledger cursor does not match committed ledger head",
                locator={
                    "field": "ledger_cursor",
                    "expected": replay["ledger_cursor"],
                    "actual": content.get("ledger_cursor"),
                },
            )
        comparison = compare_outcomes(content.get("shard_outcomes", []), replay["shard_outcomes"])
        if not comparison["match"]:
            diff = comparison["first_difference"]
            raise SnapshotDigestError(
                "snapshot outcome differs from ledger replay",
                locator=diff,
            )
        result["comparison"] = comparison
        result["ok"] = True
        return result
    finally:
        ledger.close()


def snapshot_file_digest(path: Path | str) -> str:
    return sha256_hex(Path(path).read_bytes())
