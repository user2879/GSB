from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .canonical import canonical_text, sha256_hex, write_canonical_json
from .engine import scale_caliber
from .ledger import INTEGER_FORMAT_VERSION
from .spec import SessionSpec


def build_snapshot(spec: SessionSpec, result: dict[str, Any]) -> dict[str, Any]:
    sealed_windows = [window for window in result["windows"] if window["sealed"]]
    open_windows = [window for window in result["windows"] if not window["sealed"]]
    current_window = open_windows[-1] if open_windows else None
    current_items: list[dict[str, Any]] = []
    if current_window is not None:
        ids = set(current_window["item_ids"])
        current_items = [item for item in result["items"] if item["item_id"] in ids]
    body = {
        "integer_format_version": INTEGER_FORMAT_VERSION,
        "snapshot_format_version": 1,
        "session_id": spec.session_id,
        "script_version": spec.script_version,
        "scale_version": spec.scale_version,
        "script_digest": spec.script_digest,
        "scale_caliber": scale_caliber(spec),
        "scale_caliber_digest": sha256_hex(scale_caliber(spec)),
        "sealed_windows": sealed_windows,
        "current_window": current_window,
        "current_window_partial_items": current_items,
        "summary_digest": result["summary"]["content_digest"],
    }
    body["content_digest"] = sha256_hex({k: v for k, v in body.items() if k != "content_digest"})
    return body


def write_snapshot(path: str | Path, snapshot: dict[str, Any]) -> Path:
    return write_canonical_json(path, snapshot)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("snapshot must be an object")
    return value


def _force_writable_unlink(path: Path) -> None:
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        path.unlink()


def _make_readonly_backup(path: Path, old: dict[str, Any]) -> Path:
    digest = str(old.get("content_digest", "unknown"))[:12]
    backup = path.with_name(f"{path.stem}.replaced-{digest}.bak.json")
    if backup.exists():
        _force_writable_unlink(backup)
    shutil.copy2(path, backup)
    os.chmod(backup, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    stale_sidecar = path.with_name(f"{path.stem}.stale.json")
    _force_writable_unlink(stale_sidecar)
    stale = {
        "integer_format_version": INTEGER_FORMAT_VERSION,
        "stale": True,
        "reason": "version_or_content_digest_mismatch",
        "old_content_digest": old.get("content_digest"),
        "old_script_version": old.get("script_version"),
        "old_scale_version": old.get("scale_version"),
        "marked_stale_fields": ["sealed_windows", "error_bounds", "observation_view"],
        "backup_path": str(backup),
    }
    write_canonical_json(stale_sidecar, stale)
    os.chmod(stale_sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return backup


def open_or_replace_snapshot(
    path: str | Path,
    current_spec: SessionSpec,
    current_result: dict[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    current_identity = {
        "script_version": current_spec.script_version,
        "scale_version": current_spec.scale_version,
        "script_digest": current_spec.script_digest,
        "scale_caliber_digest": sha256_hex(scale_caliber(current_spec)),
    }
    if not path.exists():
        snapshot = build_snapshot(current_spec, current_result)
        write_snapshot(path, snapshot)
        return {"action": "created", "path": str(path), "snapshot": snapshot, "backup_path": None}

    old = _read_json(path)
    recomputed_digest = sha256_hex({k: v for k, v in old.items() if k != "content_digest"})
    digest_valid = recomputed_digest == old.get("content_digest")
    identity_matches = (
        old.get("script_version") == current_identity["script_version"]
        and old.get("scale_version") == current_identity["scale_version"]
        and old.get("script_digest") == current_identity["script_digest"]
        and old.get("scale_caliber_digest") == current_identity["scale_caliber_digest"]
        and digest_valid
    )
    if identity_matches:
        return {"action": "reused", "path": str(path), "snapshot": old, "backup_path": None}

    backup = _make_readonly_backup(path, old)
    replacement = build_snapshot(current_spec, current_result)
    tmp = path.with_suffix(".tmp.json")
    write_canonical_json(tmp, replacement)
    os.replace(tmp, path)
    return {
        "action": "replaced_whole_snapshot",
        "path": str(path),
        "backup_path": str(backup),
        "old_content_digest": old.get("content_digest"),
        "reasons": {
            "digest_valid": digest_valid,
            "identity_matches": identity_matches,
            "current_identity": current_identity,
            "old_identity": {
                "script_version": old.get("script_version"),
                "scale_version": old.get("scale_version"),
                "script_digest": old.get("script_digest"),
                "scale_caliber_digest": old.get("scale_caliber_digest"),
            },
        },
        "snapshot": replacement,
        "merge_performed": False,
    }


def assert_version_monotonic(
    old_script_version: int,
    new_script_version: int,
    old_scale_version: int | None = None,
    new_scale_version: int | None = None,
) -> None:
    from .errors import SpecError

    if new_script_version <= old_script_version:
        raise SpecError(
            "script_version must increase monotonically",
            field_name="script_version",
            details={"old": old_script_version, "new": new_script_version},
        )
    if old_scale_version is not None and new_scale_version is not None and new_scale_version < old_scale_version:
        raise SpecError(
            "scale_version must increase monotonically",
            field_name="scale_version",
            details={"old": old_scale_version, "new": new_scale_version},
        )


def visible_snapshot_view(path: str | Path) -> dict[str, Any]:
    """Return only the current replacement; stale backups are excluded by default."""

    path = Path(path)
    current = _read_json(path)
    stale_sidecar = path.with_name(f"{path.stem}.stale.json")
    stale = _read_json(stale_sidecar) if stale_sidecar.exists() else None
    return {
        "integer_format_version": INTEGER_FORMAT_VERSION,
        "visible": {
            "content_digest": current.get("content_digest"),
            "script_version": current.get("script_version"),
            "scale_version": current.get("scale_version"),
            "sealed_window_count": len(current.get("sealed_windows", [])),
            "current_window": current.get("current_window"),
        },
        "excluded_stale": stale,
    }
