"""Session snapshots: version-aware full replacement recovery."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import BizError, ErrorCode
from .logging_setup import StructuredLogger, utc_now_iso
from .manifest import atomic_write, canonical_json
from .session import SessionStore

SNAPSHOT_FORMAT = 1


def build_snapshot(
    manifest: dict,
    plan: dict,
    store: SessionStore,
    session_id: str,
) -> dict:
    return {
        "snapshot_format": SNAPSHOT_FORMAT,
        "captured_at": utc_now_iso(),
        "manifest_version": manifest["format_version"],
        "config_version": plan["config_version"],
        "manifest": {
            key: value for key, value in manifest.items() if not key.startswith("__")
        },
        "plan": plan,
        "receipts": store.list_receipts(session_id),
        "state_transitions": store.list_transitions(session_id),
        "committed_versions": store.all_committed_versions(),
        "session_id": session_id,
    }


def save_snapshot(snapshot: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write(
        target,
        json.dumps(snapshot, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )


def load_snapshot(path: str | Path) -> dict:
    snapshot_path = Path(path)
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BizError(
            ErrorCode.E_FORMAT,
            f"cannot read snapshot {snapshot_path}: {exc}",
            {"field_path": str(snapshot_path)},
        ) from exc
    required = {
        "snapshot_format",
        "manifest_version",
        "config_version",
        "manifest",
        "plan",
        "receipts",
        "state_transitions",
        "committed_versions",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise BizError(
            ErrorCode.E_FORMAT,
            "snapshot missing required fields",
            {"field_path": "$", "missing": missing},
        )
    return payload


def restore_snapshot(
    snapshot_path: str | Path,
    current_manifest: dict,
    store: SessionStore,
    session_id: str,
    archive_dir: str | Path,
    logger: StructuredLogger | None = None,
) -> dict:
    """Restore a snapshot.

    Version mismatch triggers E_VERSION *and* a full replacement: the current
    manifest, plan and receipt/state rows are wholesale replaced by snapshot
    contents (never incrementally merged). The replaced snapshot/current state
    is archived read-only and labelled with both versions.
    """
    snapshot = load_snapshot(snapshot_path)
    current_version = current_manifest["format_version"]
    snapshot_version = snapshot["manifest_version"]
    replaced = snapshot_version != current_version

    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)
    label = f"v{snapshot_version}-replaced-current-v{current_version}"

    if replaced:
        # Archive the incoming snapshot read-only, labelled with both versions.
        archived_snapshot = archive / f"{Path(snapshot_path).stem}.{label}.json"
        atomic_write(
            archived_snapshot,
            json.dumps(snapshot, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            readonly=True,
        )
        # Archive current database state before wholesale replacement.
        archived_state = archive / f"state-before-restore.{label}.json"
        current_state = _dump_current_state(store)
        atomic_write(
            archived_state,
            json.dumps(current_state, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            readonly=True,
        )
        if logger is not None:
            logger.biz_error(
                "snapshot.version_conflict",
                ErrorCode.E_VERSION,
                "snapshot manifest version differs from current; full replacement applied",
                snapshot_version=snapshot_version,
                current_version=current_version,
            )

    _replace_store_state(store, snapshot, session_id)
    restored_manifest = snapshot["manifest"]

    report = {
        "replaced": replaced,
        "mode": "full_replacement" if replaced else "compatible_restore",
        "snapshot_manifest_version": snapshot_version,
        "current_manifest_version": current_version,
        "restored_config_version": snapshot["config_version"],
        "session_id": session_id,
        "archive_dir": str(archive),
        "restored_manifest": restored_manifest,
    }
    if replaced:
        report["archived_snapshot"] = str(archive)
        report["label"] = label
        report["error_code"] = ErrorCode.E_VERSION
        raise BizError(
            ErrorCode.E_VERSION,
            (
                f"snapshot version v{snapshot_version} conflicts with current "
                f"v{current_version}; state fully replaced from snapshot"
            ),
            report,
        )
    return report


def _dump_current_state(store: SessionStore) -> dict:
    rows = store.conn.execute("SELECT session_id FROM sessions").fetchall()
    sessions = {}
    for row in rows:
        session_id = row["session_id"]
        sessions[session_id] = {
            "session": store.get_session(session_id),
            "transitions": store.list_transitions(session_id),
            "receipts": store.list_receipts(session_id),
        }
    return {
        "captured_at": utc_now_iso(),
        "committed_versions": [
            store.latest_committed_version()
        ],
        "sessions": sessions,
    }


def _replace_store_state(store: SessionStore, snapshot: dict, session_id: str) -> None:
    # Wholesale replacement: clear current rows, then insert snapshot rows.
    conn = store.conn
    with conn:
        conn.execute("DELETE FROM receipts")
        conn.execute("DELETE FROM state_transitions")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM committed_versions")

        plan = snapshot["plan"]
        now = utc_now_iso()
        final_state = (
            snapshot["state_transitions"][-1]["to_state"]
            if snapshot["state_transitions"]
            else "prepared"
        )
        conn.execute(
            """INSERT INTO sessions
               (session_id, config_version, manifest_version, state, plan_json,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                snapshot["config_version"],
                snapshot["manifest_version"],
                final_state,
                canonical_json(plan),
                now,
                now,
            ),
        )
        for transition in snapshot["state_transitions"]:
            conn.execute(
                """INSERT INTO state_transitions
                   (session_id, from_state, to_state, config_version,
                    confirmed_nodes_json, ts)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    transition.get("from_state"),
                    transition["to_state"],
                    transition["config_version"],
                    canonical_json(transition.get("confirmed_nodes", [])),
                    transition.get("ts", now),
                ),
            )
        for receipt in snapshot["receipts"]:
            raw = receipt.get("raw", {})
            conn.execute(
                """INSERT INTO receipts
                   (session_id, config_version, node, attempt, receipt_id, ok,
                    retry, duplicate, digest, raw_json, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    receipt["config_version"],
                    receipt["node"],
                    receipt["attempt"],
                    receipt.get("receipt_id"),
                    receipt["ok"],
                    receipt["retry"],
                    receipt["duplicate"],
                    receipt.get("digest"),
                    canonical_json(raw),
                    receipt.get("ts", now),
                ),
            )
        for config_version in snapshot.get("committed_versions", []):
            conn.execute(
                """INSERT OR IGNORE INTO committed_versions
                   (config_version, session_id, plan_json, committed_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    config_version,
                    session_id,
                    canonical_json(plan),
                    now,
                ),
            )
