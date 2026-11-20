"""Snapshot capture, integrity validation, replay checks and wholesale restore."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import EdgeError
from .ledger import Ledger, _normalize_entries, _validate_entries_against, utc_now
from .model import (
    ContentSnapshot,
    Policy,
    cache_key,
    canonical_json_bytes,
    canonical_trace_bytes,
)
from .spec import parse_content_records, parse_policy


SNAPSHOT_SCHEMA = "edgecache.snapshot/v1"


@dataclass(frozen=True)
class LoadedSnapshot:
    path: Path
    document: dict[str, Any]
    content: ContentSnapshot
    policy: Policy
    content_version: str
    policy_version: str
    comparison: dict[str, Any]


def _json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


class SnapshotStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, ledger: Ledger) -> dict[str, Any]:
        row = ledger.current_row()
        if row is None:
            raise EdgeError(code="E_SNAPSHOT", message="cannot save a snapshot without a complete ledger version")
        content_records = json.loads(row["content_json"])
        policy = json.loads(row["policy_json"])
        content = ContentSnapshot(tuple(parse_content_records(content_records).records))
        entries = ledger.current_entries()
        evictions = ledger.current_evictions()
        propagation = ledger.current_propagation()
        propagation_triggers = sorted({str(row["trigger_path"]) for row in propagation})
        document: dict[str, Any] = {
            "schema": SNAPSHOT_SCHEMA,
            "snapshot_id": f"snapshot-{row['ledger_version']}",
            "saved_at": utc_now(),
            "ledger_version": row["ledger_version"],
            "content_version": row["content_version"],
            "policy_version": row["policy_version"],
            "content_digest": row["content_digest"],
            "policy_digest": row["policy_digest"],
            "content_versions": content.etag_set,
            "content_records": content_records,
            "policy": policy,
            "graph": content.graph_dict(),
            "entries": entries,
            "evictions": evictions,
            "propagation": propagation,
            "propagation_triggers": propagation_triggers,
            "execution_trace": canonical_trace_bytes(propagation).decode("utf-8"),
            "progress_cursor": row["progress_cursor"],
        }
        prior_files = sorted(self.directory.glob("snapshot-*.json"))
        counter = 1
        while True:
            target = self.directory / f"snapshot-{row['ledger_version']}-{counter:03d}.json"
            if not target.exists():
                break
            counter += 1
        temporary = self.directory / f".{target.name}.tmp"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
        # Existing snapshots become immutable backups.  The new snapshot stays
        # writable until the next capture makes it a backup as well.
        for prior in prior_files:
            try:
                os.chmod(prior, stat.S_IREAD)
            except OSError:
                pass
        ledger.register_snapshot(
            snapshot_id=f"{document['snapshot_id']}-{counter:03d}",
            path=str(target.resolve()),
            ledger_version=int(row["ledger_version"]),
            content_digest=str(row["content_digest"]),
            policy_digest=str(row["policy_digest"]),
            read_only=False,
        )
        return {
            "ok": True,
            "snapshot_path": str(target.resolve()),
            "snapshot_id": f"{document['snapshot_id']}-{counter:03d}",
            "ledger_version": row["ledger_version"],
            "content_version": row["content_version"],
            "policy_version": row["policy_version"],
            "content_digest": row["content_digest"],
            "entry_count": len(entries),
        }

    def load(self, path: str | Path, ledger: Ledger | None = None) -> LoadedSnapshot:
        source = Path(path)
        try:
            raw = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="unable to read snapshot file",
                field_path="path",
                details={"file": str(source), "error": str(exc)},
            ) from exc
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot is not valid JSON",
                field_path=f"json.line[{exc.lineno}].column[{exc.colno}]",
                details={"error": exc.msg},
            ) from exc
        if not isinstance(document, dict) or document.get("schema") != SNAPSHOT_SCHEMA:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="unsupported or missing snapshot schema",
                field_path="schema",
                details={"expected": SNAPSHOT_SCHEMA, "actual": document.get("schema") if isinstance(document, dict) else None},
            )
        return self.validate(document, source=source, ledger=ledger)

    def validate(
        self,
        document: dict[str, Any],
        *,
        source: Path | None = None,
        ledger: Ledger | None = None,
    ) -> LoadedSnapshot:
        try:
            content = ContentSnapshot(tuple(parse_content_records(document["content_records"]).records))
        except (KeyError, TypeError, ValueError, EdgeError) as exc:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot content records are invalid",
                field_path="content_records",
                details={"error": str(exc)},
            ) from exc
        try:
            policy = parse_policy(document["policy"])
        except (KeyError, TypeError, ValueError, EdgeError) as exc:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot policy is invalid",
                field_path="policy",
                details={"error": str(exc)},
            ) from exc

        declared_content_digest = document.get("content_digest")
        if content.digest != declared_content_digest:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot content digest does not match its content records",
                field_path="content_digest",
                details={"declared": declared_content_digest, "computed": content.digest},
            )
        if policy.digest != document.get("policy_digest"):
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot policy digest does not match its policy document",
                field_path="policy_digest",
                details={"declared": document.get("policy_digest"), "computed": policy.digest},
            )
        content_version = str(document.get("content_version"))
        policy_version = str(document.get("policy_version"))
        if content.version != content_version:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot content_version does not match its content records",
                field_path="content_version",
                details={"declared": content_version, "computed": content.version},
            )
        if policy.version != policy_version:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot policy_version does not match its policy document",
                field_path="policy_version",
                details={"declared": policy_version, "computed": policy.version},
            )
        declared_versions = document.get("content_versions")
        if declared_versions != content.etag_set:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot version set does not match its content records",
                field_path="content_versions",
                details={"declared": declared_versions, "computed": content.etag_set},
            )
        if document.get("graph") != content.graph_dict():
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot derivation graph does not match its content records",
                field_path="graph",
            )

        raw_entries = document.get("entries", [])
        if not isinstance(raw_entries, list):
            raise EdgeError(code="E_SNAPSHOT", message="snapshot entries must be an array", field_path="entries")
        entries = _normalize_entries(raw_entries)
        for entry in entries:
            expected_key = cache_key(
                policy,
                node_id=entry["node_id"],
                path=entry["path"],
                etag=entry["etag"],
            )
            if entry["cache_key"] != expected_key:
                raise EdgeError(
                    code="E_SNAPSHOT",
                    message="snapshot cache key does not match the declared policy key fields",
                    path=entry["path"],
                    field_path=f"entries[node={entry['node_id']},key={entry['cache_key']}].cache_key",
                    details={"expected": expected_key, "actual": entry["cache_key"]},
                )
        _validate_entries_against(content, policy, content_version, entries)

        propagation = document.get("propagation", [])
        if not isinstance(propagation, list):
            raise EdgeError(code="E_SNAPSHOT", message="snapshot propagation must be an array", field_path="propagation")
        declared_triggers = document.get("propagation_triggers")
        if declared_triggers is None:
            declared_triggers = sorted({str(row.get("trigger_path")) for row in propagation if isinstance(row, dict)})
        _validate_replay(
            content,
            propagation,
            content_version,
            policy_version,
            declared_triggers=declared_triggers,
        )

        snapshot_set = {
            "ledger_version": document.get("ledger_version"),
            "content_version": content_version,
            "policy_version": policy_version,
            "content_versions": content.etag_set,
            "content_digest": content.digest,
            "policy_digest": policy.digest,
        }
        disk_set = ledger.version_set() if ledger is not None else None
        if disk_set is None:
            comparison = {"same_version_set": False, "disk": None, "snapshot": snapshot_set, "requires_replace": True}
        else:
            same = all(
                disk_set.get(key) == snapshot_set.get(key)
                for key in ("content_version", "policy_version", "content_versions", "content_digest", "policy_digest")
            )
            comparison = {
                "same_version_set": same,
                "disk": disk_set,
                "snapshot": snapshot_set,
                "requires_replace": not same,
            }
        return LoadedSnapshot(
            path=source or Path(str(document.get("snapshot_id", "<memory>"))),
            document=document,
            content=content,
            policy=policy,
            content_version=content_version,
            policy_version=policy_version,
            comparison=comparison,
        )

    def restore(self, loaded: LoadedSnapshot, ledger: Ledger) -> dict[str, Any]:
        # Replay validation has already run in ``validate``; restore is a new
        # ledger version containing only snapshot state, never a merge.
        version = ledger.restore_snapshot_document(loaded.document)
        state = ledger.version_row(version)
        return {
            "ok": True,
            "restored_ledger_version": version,
            "source_snapshot_ledger_version": loaded.document.get("ledger_version"),
            "content_version": state["content_version"],
            "policy_version": state["policy_version"],
            "comparison": loaded.comparison,
            "entries": ledger.entries_for_version(version),
            "propagation": ledger.propagation_for_version(version),
            "replay": {"ok": True, "checked_rows": len(ledger.propagation_for_version(version))},
        }


def _validate_replay(
    content: ContentSnapshot,
    rows: Sequence[Mapping[str, Any]],
    content_version: str,
    policy_version: str,
    *,
    declared_triggers: Sequence[str] | None = None,
) -> None:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        try:
            normalized.append(
                {
                    "seq": int(raw["seq"]),
                    "trigger_path": str(raw["trigger_path"]),
                    "affected_path": str(raw["affected_path"]),
                    "depth": int(raw["depth"]),
                    "content_version": str(raw.get("content_version", content_version)),
                    "policy_version": str(raw.get("policy_version", policy_version)),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EdgeError(
                code="E_REPLAY",
                message="propagation trace row is malformed",
                field_path="propagation",
                details={"row": raw, "error": str(exc)},
            ) from exc
    normalized.sort(key=lambda row: row["seq"])
    for row in normalized:
        if row["content_version"] != content_version:
            raise EdgeError(
                code="E_REPLAY",
                message="propagation trace mixes content versions",
                path=row["trigger_path"],
                field_path=f"propagation[seq={row['seq']}].content_version",
                details={"expected": content_version, "actual": row["content_version"]},
            )
        if row["policy_version"] != policy_version:
            raise EdgeError(
                code="E_REPLAY",
                message="propagation trace mixes policy versions",
                path=row["trigger_path"],
                field_path=f"propagation[seq={row['seq']}].policy_version",
                details={"expected": policy_version, "actual": row["policy_version"]},
            )
    groups: dict[str, list[dict[str, Any]]] = {}
    first_seq: dict[str, int] = {}
    for row in normalized:
        trigger = row["trigger_path"]
        groups.setdefault(trigger, []).append(row)
        first_seq.setdefault(trigger, row["seq"])
    if declared_triggers is not None:
        if not isinstance(declared_triggers, list) or any(not isinstance(item, str) for item in declared_triggers):
            raise EdgeError(
                code="E_REPLAY",
                message="snapshot propagation trigger list is malformed",
                field_path="propagation_triggers",
            )
        declared = sorted(set(declared_triggers))
        observed = sorted(groups)
        if declared != observed:
            raise EdgeError(
                code="E_REPLAY",
                message="snapshot propagation trigger set does not match the saved trace",
                path=declared[0] if declared else (observed[0] if observed else None),
                field_path="propagation_triggers",
                details={
                    "trigger_path": declared[0] if declared else (observed[0] if observed else None),
                    "affected_path": None,
                    "expected_triggers": declared,
                    "actual_triggers": observed,
                },
            )
    trigger_order = sorted(groups, key=lambda item: (first_seq[item], item))
    for trigger in trigger_order:
        actual = [row["affected_path"] for row in groups[trigger]]
        try:
            expected = content.descendant_order(trigger)
        except EdgeError as exc:
            raise EdgeError(
                code="E_REPLAY",
                message="snapshot propagation trigger is absent from the replayed dependency graph",
                path=trigger,
                field_path=f"propagation[trigger={trigger}].trigger_path",
                details={
                    "trigger_path": trigger,
                    "affected_path": actual[0] if actual else None,
                    "cause": exc.message,
                },
            ) from exc
        if actual != expected:
            index = _first_difference(expected, actual)
            actual_value = actual[index] if index < len(actual) else None
            expected_value = expected[index] if index < len(expected) else None
            raise EdgeError(
                code="E_REPLAY",
                message="snapshot propagation trace does not match dependency-graph replay",
                path=trigger,
                field_path=f"propagation[trigger={trigger}][{index}].affected_path",
                details={
                    "reason": "affected-set-or-order-mismatch",
                    "trigger_path": trigger,
                    "first_mismatch_index": index,
                    "affected_path": actual_value if actual_value is not None else expected_value,
                    "expected_affected_path": expected_value,
                    "actual_affected_path": actual_value,
                    "expected": expected,
                    "actual": actual,
                },
            )
        for row in groups[trigger]:
            expected_depth = content.depth_from(trigger, row["affected_path"])
            if row["depth"] != expected_depth:
                raise EdgeError(
                    code="E_REPLAY",
                    message="snapshot propagation depth does not match graph replay",
                    path=trigger,
                    field_path=f"propagation[seq={row['seq']}].depth",
                    details={
                        "trigger_path": trigger,
                        "affected_path": row["affected_path"],
                        "expected_depth": expected_depth,
                        "actual_depth": row["depth"],
                    },
                )


def _first_difference(expected: Sequence[str], actual: Sequence[str]) -> int:
    limit = max(len(expected), len(actual))
    for index in range(limit):
        expected_value = expected[index] if index < len(expected) else object()
        actual_value = actual[index] if index < len(actual) else object()
        if expected_value != actual_value:
            return index
    return 0


def snapshot_summary(loaded: LoadedSnapshot) -> dict[str, Any]:
    document = loaded.document
    return {
        "ok": True,
        "snapshot_path": str(loaded.path),
        "ledger_version": document.get("ledger_version"),
        "content_version": loaded.content_version,
        "policy_version": loaded.policy_version,
        "content_digest": loaded.content.digest,
        "policy_digest": loaded.policy.digest,
        "entry_count": len(document.get("entries", [])),
        "propagation_count": len(document.get("propagation", [])),
        "comparison": loaded.comparison,
        "replay": {"ok": True, "checked_rows": len(document.get("propagation", []))},
    }
