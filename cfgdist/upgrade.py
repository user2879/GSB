"""Entry-by-entry manifest upgrade with checkpoints and resumable recovery.

The upgrade walks one format version at a time (v1 -> v2 -> v3). Within each
version step every entry is processed individually and a checkpoint plus an
append-only trail record is written after it. A forced termination therefore
resumes from the exact (step, entry) pair already completed; no entry is ever
upgraded twice. The source manifest is only rewritten atomically once the full
pipeline reaches the current version.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import CURRENT_FORMAT_VERSION
from .errors import BizError, ErrorCode
from .logging_setup import StructuredLogger, utc_now_iso
from .manifest import (
    DEFAULT_STRATEGY,
    atomic_write,
    load_manifest,
    load_json_with_duplicate_check,
    manifest_to_json,
)


def _utc_now() -> str:
    return utc_now_iso()


def resolve_manifest_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    if path.is_dir():
        candidates = sorted(
            item
            for item in path.glob("*.json")
            if item.is_file() and not item.name.endswith(".backup.json")
        )
        if not candidates:
            raise BizError(
                ErrorCode.E_FORMAT,
                f"no manifest json found in directory {path}",
                {"directory": str(path)},
            )
        return candidates[0]
    return path


def _read_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {
            "completed_steps": [],
            "from_version": None,
            "to_version": CURRENT_FORMAT_VERSION,
        }
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("completed_steps", [])
    return data


def _write_checkpoint(path: Path, payload: dict) -> None:
    atomic_write(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def upgrade_entry(entry: dict, from_version: int, first_node: str) -> dict:
    """Upgrade one entry by exactly one format version."""
    upgraded = dict(entry)
    if from_version == 1:
        upgraded["node"] = first_node
        upgraded["config_version"] = 2
    elif from_version == 2:
        upgraded["config_version"] = 3
        upgraded.setdefault("strategy", dict(DEFAULT_STRATEGY))
    else:  # pragma: no cover - guarded by caller
        raise BizError(
            ErrorCode.E_FORMAT,
            f"cannot upgrade entry from version {from_version}",
        )
    return upgraded


def _step_key(step: int, entry_key: str) -> str:
    return f"v{step}:{entry_key}"


def upgrade_manifest(
    manifest_path: str | Path,
    nodes: list[str],
    runtime_dir: str | Path,
    limit: int | None = None,
    logger: StructuredLogger | None = None,
) -> dict:
    if not nodes:
        raise BizError(ErrorCode.E_FORMAT, "node declaration must contain at least one node")
    source = resolve_manifest_path(manifest_path)
    runtime = Path(runtime_dir)
    checkpoint_dir = runtime / "checkpoints"
    backup_dir = runtime / "backups"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{source.stem}.checkpoint.json"
    trail_path = checkpoint_dir / f"{source.stem}.upgrade-trail.jsonl"
    backup_path = backup_dir / f"{source.stem}.v.backup.json"
    working_path = checkpoint_dir / f"{source.stem}.working.json"

    manifest = load_manifest(source)
    original_version = manifest["format_version"]
    if original_version >= CURRENT_FORMAT_VERSION:
        return {
            "status": "noop",
            "source": str(source),
            "format_version": original_version,
            "processed": 0,
            "checkpoint": str(checkpoint_path),
            "trail": str(trail_path),
        }

    checkpoint = _read_checkpoint(checkpoint_path)
    if checkpoint.get("from_version") not in (None, original_version):
        raise BizError(
            ErrorCode.E_FORMAT,
            "checkpoint does not match the manifest being upgraded",
            {
                "checkpoint_from_version": checkpoint.get("from_version"),
                "manifest_from_version": original_version,
            },
        )
    checkpoint["from_version"] = original_version
    checkpoint["to_version"] = CURRENT_FORMAT_VERSION
    completed_steps: list[str] = list(checkpoint.get("completed_steps", []))
    completed_set = set(completed_steps)

    # A previous run may have completed whole version steps; resume from its
    # persisted intermediate working document rather than the original source.
    if working_path.exists():
        # Intermediate working documents carry upgraded fields while still in
        # progress, so parse without strict version-specific field allowlists.
        working = load_json_with_duplicate_check(
            working_path.read_text(encoding="utf-8"), str(working_path)
        )
        if not (original_version < working["format_version"] <= CURRENT_FORMAT_VERSION):
            raise BizError(
                ErrorCode.E_FORMAT,
                "working manifest version is inconsistent with checkpoint",
                {
                    "working_version": working["format_version"],
                    "source_version": original_version,
                },
            )
    else:
        working = json.loads(json.dumps(manifest))

    if not backup_path.exists():
        atomic_write(backup_path, source.read_text(encoding="utf-8"), readonly=True)

    first_node = nodes[0]
    processed_now = 0
    interrupted = False

    for step in range(original_version, CURRENT_FORMAT_VERSION):
        for index, entry in enumerate(working["entries"]):
            key = entry["key"]
            marker = _step_key(step, key)
            if marker in completed_set:
                continue
            before = dict(entry)
            entry = upgrade_entry(entry, step, first_node)
            working["entries"][index] = entry
            completed_steps.append(marker)
            processed_now += 1
            checkpoint["completed_steps"] = completed_steps
            checkpoint["updated_at"] = _utc_now()
            _write_checkpoint(checkpoint_path, checkpoint)
            trail_record = {
                "ts": _utc_now(),
                "step": f"v{step}->v{step + 1}",
                "entry_index": index,
                "key": key,
                "from_version": step,
                "to_version": step + 1,
                "before": before,
                "after": entry,
            }
            with trail_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trail_record, sort_keys=True) + "\n")
            if logger is not None:
                logger.log("upgrade.entry", key=key, step=trail_record["step"])
            if limit is not None and processed_now >= limit:
                interrupted = True
                # Persist the partial intermediate document so resumed runs
                # keep fields produced by earlier (step, entry) upgrades.
                working["format_version"] = step + 1
                atomic_write(working_path, manifest_to_json(working))
                break
        if interrupted:
            break
        # Whole version step finished: persist the intermediate document so a
        # later resume starts from the correct format version.
        working["format_version"] = step + 1
        atomic_write(working_path, manifest_to_json(working))

    if interrupted:
        return {
            "status": "interrupted",
            "source": str(source),
            "format_version": original_version,
            "processed": processed_now,
            "total_entries": len(working["entries"]),
            "total_steps": (CURRENT_FORMAT_VERSION - original_version)
            * len(working["entries"]),
            "completed_steps": completed_steps,
            "checkpoint": str(checkpoint_path),
            "trail": str(trail_path),
            "backup": str(backup_path),
        }

    expected_steps = (CURRENT_FORMAT_VERSION - original_version) * len(working["entries"])
    if len(completed_steps) != expected_steps:
        raise BizError(
            ErrorCode.E_FORMAT,
            "internal upgrade error: checkpoint does not cover every entry step",
            {"completed": len(completed_steps), "expected": expected_steps},
        )

    working["format_version"] = CURRENT_FORMAT_VERSION
    working.setdefault("derivations", [])
    atomic_write(source, manifest_to_json(working))
    if working_path.exists():
        working_path.unlink()
    checkpoint["rewritten_at"] = _utc_now()
    _write_checkpoint(checkpoint_path, checkpoint)
    try:
        os.chmod(backup_path, 0o444)
    except OSError:  # pragma: no cover - Windows read-only is best effort
        pass
    if logger is not None:
        logger.log(
            "upgrade.complete",
            source=str(source),
            from_version=original_version,
            to_version=CURRENT_FORMAT_VERSION,
        )
    return {
        "status": "completed",
        "source": str(source),
        "format_version": CURRENT_FORMAT_VERSION,
        "processed": processed_now,
        "total_entries": len(working["entries"]),
        "total_steps": expected_steps,
        "completed_steps": completed_steps,
        "checkpoint": str(checkpoint_path),
        "trail": str(trail_path),
        "backup": str(backup_path),
    }


def read_upgrade_trail(trail_path: str | Path) -> list[dict]:
    path = Path(trail_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
