"""Version-aware whole-ledger snapshots and replay recovery."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DiagnosticError, is_int
from .jsonio import atomic_write_json, canonical_json, load_json_file, sha256_canonical
from .ledger import (
    LEDGER_FORMAT_VERSION,
    Ledger,
    RecoveryReport,
    ledger_content_digest,
    validate_row_payload,
    write_ledger_rows_to_new_database,
)
from .model import parameter_digest
from .spec import WorkerSet


SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_KEYS = {
    "schema_version",
    "worker_set_version",
    "param_versions",
    "param_digests",
    "ledger_format_version",
    "ledger_digest",
    "cursor",
    "rows",
    "content_digest",
}


@dataclass(frozen=True)
class RestoreOutcome:
    whole_replacement: bool
    replacement_reasons: tuple[str, ...]
    recovery: RecoveryReport
    ledger_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "whole_replacement": self.whole_replacement,
            "replacement_reasons": list(self.replacement_reasons),
            "recovery": self.recovery.to_dict(),
            "ledger_digest": self.ledger_digest,
        }


def _snapshot_content_digest(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "content_digest"}
    return sha256_canonical(body)


def create_snapshot(worker_set: WorkerSet, ledger: Ledger, path: str | Path) -> dict[str, Any]:
    rows = ledger.snapshot_rows()
    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_FORMAT_VERSION,
        "worker_set_version": worker_set.worker_set_version,
        "param_versions": worker_set.param_versions(),
        "param_digests": {
            worker.worker_id: parameter_digest(worker)
            for worker in sorted(worker_set.workers, key=lambda item: item.worker_id)
        },
        "ledger_format_version": ledger.format_version,
        "ledger_digest": ledger.content_digest,
        "cursor": ledger.progress_cursor,
        "rows": rows,
    }
    payload["content_digest"] = _snapshot_content_digest(payload)
    atomic_write_json(path, payload)
    return payload


def load_snapshot(path: str | Path) -> dict[str, Any]:
    value = load_json_file(path, object_name=str(path), code="E_SNAPSHOT")
    if not isinstance(value, dict):
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="snapshot must be a JSON object",
            object=str(path),
            field="<root>",
        )
    for field in sorted(SNAPSHOT_KEYS):
        if field not in value:
            raise DiagnosticError(
                code="E_SNAPSHOT",
                message=f"snapshot is missing {field!r}",
                object=str(path),
                field=field,
            )
    for field in value:
        if field not in SNAPSHOT_KEYS:
            raise DiagnosticError(
                code="E_SNAPSHOT",
                message=f"unexpected snapshot field {field!r}",
                object=str(path),
                field=field,
            )
    for field in ("schema_version", "worker_set_version", "ledger_format_version", "cursor"):
        if not is_int(value[field]) or value[field] < 0:
            raise DiagnosticError(
                code="E_SNAPSHOT",
                message=f"{field} must be a non-negative integer",
                object=str(path),
                field=field,
            )
    if value["schema_version"] != SNAPSHOT_FORMAT_VERSION:
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message=f"unsupported snapshot schema {value['schema_version']}",
            object=str(path),
            field="schema_version",
        )
    versions = value["param_versions"]
    if not isinstance(versions, dict) or not all(
        isinstance(key, str) and key and is_int(item) and item >= 1
        for key, item in versions.items()
    ):
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="param_versions must map worker ids to positive integers",
            object=str(path),
            field="param_versions",
        )
    digests = value["param_digests"]
    if not isinstance(digests, dict) or not all(
        isinstance(key, str) and key and isinstance(item, str) and len(item) == 64
        for key, item in digests.items()
    ):
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="param_digests must map worker ids to SHA-256 digests",
            object=str(path),
            field="param_digests",
        )
    if not isinstance(value["ledger_digest"], str) or not value["ledger_digest"]:
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="ledger_digest must be a non-empty string",
            object=str(path),
            field="ledger_digest",
        )
    if not isinstance(value["rows"], list):
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="rows must be an array",
            object=str(path),
            field="rows",
        )
    expected_content = _snapshot_content_digest(value)
    if value["content_digest"] != expected_content:
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="snapshot content digest mismatch",
            object=str(path),
            field="content_digest",
            details={"expected": expected_content, "actual": value["content_digest"]},
        )
    rows = [validate_row_payload(row, code="E_SNAPSHOT") for row in value["rows"]]
    completed = sum(1 for row in rows if row["status"] == "completed")
    if completed != value["cursor"]:
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="snapshot cursor disagrees with completed rows",
            object=str(path),
            field="cursor",
        )
    if value["ledger_format_version"] != LEDGER_FORMAT_VERSION:
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="snapshot ledger format is unsupported",
            object=str(path),
            field="ledger_format_version",
        )
    expected_ledger = ledger_content_digest(
        rows,
        format_version=value["ledger_format_version"],
        progress_cursor=value["cursor"],
    )
    if expected_ledger != value["ledger_digest"]:
        raise DiagnosticError(
            code="E_SNAPSHOT",
            message="snapshot rows disagree with ledger_digest",
            object=str(path),
            field="ledger_digest",
            details={"expected": expected_ledger, "actual": value["ledger_digest"]},
        )
    return value


def _current_fingerprint(worker_set: WorkerSet) -> tuple[dict[str, int], dict[str, str]]:
    return (
        worker_set.param_versions(),
        {
            worker.worker_id: parameter_digest(worker)
            for worker in sorted(worker_set.workers, key=lambda item: item.worker_id)
        },
    )


def _target_identity(path: Path) -> tuple[str, int] | None:
    if not path.exists():
        return None
    try:
        with Ledger(path) as ledger:
            return (ledger.content_digest, ledger.progress_cursor)
    except Exception:
        return None


def _atomic_restore_ledger(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.restore.", suffix=".sqlite", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        write_ledger_rows_to_new_database(
            temporary,
            rows=snapshot["rows"],
            format_version=snapshot["ledger_format_version"],
            progress_cursor=snapshot["cursor"],
            content_digest=snapshot["ledger_digest"],
            code="E_SNAPSHOT",
        )
        for sidecar in (Path(str(path) + "-wal"), Path(str(path) + "-shm")):
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def restore_snapshot(
    snapshot_path: str | Path,
    ledger_path: str | Path,
    current_workers: WorkerSet,
) -> RestoreOutcome:
    snapshot = load_snapshot(snapshot_path)
    target = Path(ledger_path)
    current_versions, current_digests = _current_fingerprint(current_workers)
    reasons: list[str] = []
    if snapshot["worker_set_version"] != current_workers.worker_set_version:
        reasons.append("worker_set_version")
    if snapshot["param_versions"] != current_versions:
        reasons.append("param_versions")
    if snapshot["param_digests"] != current_digests:
        reasons.append("param_digests")
    identity = _target_identity(target)
    if identity is None:
        reasons.append("missing_or_invalid_target_ledger")
    else:
        if identity[0] != snapshot["ledger_digest"]:
            reasons.append("ledger_digest")
        if identity[1] != snapshot["cursor"]:
            reasons.append("cursor")
    if reasons:
        _atomic_restore_ledger(target, snapshot)
        replacement = True
    else:
        replacement = False
    with Ledger(target) as ledger:
        if ledger.content_digest != snapshot["ledger_digest"]:
            raise DiagnosticError(
                code="E_SNAPSHOT",
                message="restored ledger digest does not match snapshot",
                object=str(target),
                field="ledger_digest",
            )
        report = ledger.recovery_report()
        return RestoreOutcome(
            whole_replacement=replacement,
            replacement_reasons=tuple(reasons),
            recovery=report,
            ledger_digest=ledger.content_digest,
        )


def snapshot_json(snapshot: dict[str, Any]) -> str:
    return canonical_json(snapshot)
