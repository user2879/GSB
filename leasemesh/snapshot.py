from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from . import SNAPSHOT_FORMAT_VERSION
from .canonical import diff_flat, sha256_hex
from .errors import E_SPEC, MeshError
from .timeutil import iso_stamp


def build_snapshot(ledger: Any, spec: dict[str, Any]) -> dict[str, Any]:
    head_digest = ledger.meta("head_digest") if hasattr(ledger, "meta") else None
    projection = ledger.adjudicator.result() if hasattr(ledger, "adjudicator") else ledger
    declaration_digest = spec.get("declaration_digest") or sha256_hex(
        {key: value for key, value in spec.items() if not key.startswith("_")}
    )
    body = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "integer_format_version": int(SNAPSHOT_FORMAT_VERSION),
        "declaration_digest": declaration_digest,
        "ledger_head_digest": head_digest,
        "projection": projection,
    }
    body["content_digest"] = sha256_hex({key: value for key, value in body.items() if key != "content_digest"})
    return body


def write_snapshot_atomic(path: str | Path, snapshot: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _backup_readonly(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.readonly-{iso_stamp()}-{uuid.uuid4().hex[:8]}.bak")
    shutil.copy2(path, backup)
    os.chmod(backup, 0o444)
    return backup


def reconcile_snapshot(
    path: str | Path,
    spec: dict[str, Any],
    ledger: Any,
    *,
    diff_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    target = Path(path)
    fresh = build_snapshot(ledger, spec)
    expected_declaration = fresh["declaration_digest"]

    if not target.exists():
        write_snapshot_atomic(target, fresh)
        return {"action": "created", "path": str(target), "backup": None, "differences": []}

    raw_text = target.read_text(encoding="utf-8", errors="replace")
    try:
        old = json.loads(raw_text)
        if not isinstance(old, dict):
            raise ValueError("snapshot root is not an object")
        stored_digest = old.get("content_digest")
        actual_digest = sha256_hex({key: value for key, value in old.items() if key != "content_digest"})
        digest_ok = stored_digest == actual_digest
        version = old.get("format_version")
        version_ok = isinstance(version, int) and not isinstance(version, bool) and version == SNAPSHOT_FORMAT_VERSION
        declaration_ok = old.get("declaration_digest") == expected_declaration
        head_ok = old.get("ledger_head_digest") == fresh.get("ledger_head_digest")
        valid = digest_ok and version_ok and declaration_ok and head_ok
        parse_error = None
    except Exception as exc:
        old = None
        valid = False
        digest_ok = version_ok = declaration_ok = head_ok = False
        parse_error = str(exc)

    if valid:
        return {"action": "unchanged", "path": str(target), "backup": None, "differences": []}

    backup = _backup_readonly(target)
    if old is None:
        differences = [{"path": "$", "kind": "unparseable", "actual": parse_error}]
    else:
        differences = diff_flat(old, fresh)
        if not version_ok:
            differences.insert(0, {"path": "$.format_version", "kind": "version_mismatch"})
        if not declaration_ok:
            differences.insert(0, {"path": "$.declaration_digest", "kind": "declaration_mismatch"})
        if not digest_ok:
            differences.insert(0, {"path": "$.content_digest", "kind": "content_digest_mismatch"})
        if not head_ok:
            differences.insert(0, {"path": "$.ledger_head_digest", "kind": "head_mismatch"})

    diff_directory = Path(diff_dir) if diff_dir else target.parent / "snapshot-diffs"
    diff_directory.mkdir(parents=True, exist_ok=True)
    diff_path = diff_directory / f"{target.stem}-{iso_stamp()}-{uuid.uuid4().hex[:8]}.diff.json"
    diff_payload = {
        "snapshot": str(target),
        "backup": str(backup),
        "replacement": "wholesale",
        "difference_count": len(differences),
        "differences": differences,
    }
    diff_path.write_text(json.dumps(diff_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_snapshot_atomic(target, fresh)
    return {
        "action": "replaced_wholesale",
        "path": str(target),
        "backup": str(backup),
        "diff_path": str(diff_path),
        "difference_count": len(differences),
        "differences": differences[:20],
    }


def assert_snapshot_matches_declaration(path: str | Path, spec: dict[str, Any]) -> None:
    target = Path(path)
    if not target.exists():
        raise MeshError("snapshot does not exist", code=E_SPEC, field="snapshot")
    data = json.loads(target.read_text(encoding="utf-8"))
    expected = spec.get("declaration_digest")
    if data.get("declaration_digest") != expected:
        raise MeshError("snapshot declaration digest mismatch", code=E_SPEC, field="declaration_digest")
