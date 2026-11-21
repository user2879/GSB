"""Deterministic scenario execution over local sample files."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .clock import Clock
from .codec import decode_records
from .engine import LeaseEngine
from .errors import MiddlewareError
from .spec import Spec, load_spec
from .storage import Storage
from .upgrade import UpgradeManager


def import_legacy_leases(storage: Storage, spec: Spec,
                         legacy_files: list[Path],
                         already_expired: bool = True) -> list[str]:
    """Decode legacy record files; bad records never overwrite good ones.

    Legacy records carry only a wall-clock deadline with no monotonic anchor,
    so by default they are imported already expired (their view entries get
    reclaimed); set ``already_expired=False`` to give them a fresh monotonic
    deadline based on the config TTL.
    """
    clock = Clock()
    imported: list[str] = []
    for path in legacy_files:
        raws = json.loads(path.read_text(encoding="utf-8"))
        records = decode_records(raws, spec)  # raises E_LEASE on first bad row
        for record in records:
            if storage.get_lease_row(record.lease_id) is not None:
                continue
            state = "expired" if already_expired else "granted"
            mono_deadline = None if already_expired else clock.monotonic() +                 spec.configs[record.config_key].ttl_seconds
            with storage.transaction():
                storage.upsert_lease(record, state, mono_deadline=mono_deadline)
                storage.append_ledger(
                    "lease_granted", created_at_ms=record.expires_at_ms,
                    client_id=record.client_id, lease_id=record.lease_id,
                    config_key=record.config_key,
                    lease_version=record.lease_version, state=state,
                    renewal_count=record.renewal_count,
                    digest=record.value_digest,
                    payload={"source": "legacy_import",
                             "expires_at_ms": record.expires_at_ms})
                if state == "expired":
                    storage.append_ledger(
                        "lease_expired", created_at_ms=record.expires_at_ms,
                        client_id=record.client_id, lease_id=record.lease_id,
                        config_key=record.config_key, state="expired",
                        payload={"source": "legacy_import"})
            imported.append(record.lease_id)
    return imported


def run_changes(engine: LeaseEngine, changes_path: str | Path) -> list[dict[str, Any]]:
    steps = json.loads(Path(changes_path).read_text(encoding="utf-8"))["steps"]
    trace: list[dict[str, Any]] = []
    lease_index: dict[tuple[str, str], str] = {}

    for step in steps:
        action = step["action"]
        if action == "grant":
            record = engine.grant(step["client_id"], step["config_key"],
                                  config_version=step.get("config_version"))
            lease_index[(step["client_id"], step["config_key"])] = record.lease_id
            trace.append({"action": action, "lease_id": record.lease_id,
                          "lease_version": record.lease_version})
        elif action == "renew":
            lid = lease_index[(step["client_id"], step["config_key"])]
            record = engine.renew(lid, step["expected_version"])
            trace.append({"action": action, "lease_id": lid,
                          "renewal_count": record.renewal_count})
        elif action == "publish":
            result = engine.publish_config(step["config_key"], step["version"],
                                           step["value"])
            trace.append({"action": action, **result})
        elif action == "sweep":
            expired = engine.sweep_expired()
            trace.append({"action": action, "expired": expired})
        elif action == "sleep_ms":
            time.sleep(step["duration_ms"] / 1000.0)
            trace.append({"action": action, "slept_ms": step["duration_ms"]})
        else:
            raise MiddlewareError("E_SPEC", f"unknown scenario action {action!r}",
                                  "steps.action")
    return trace


def build_fresh(spec_path: Path, db_path: Path, legacy_files: list[Path],
                changes_path: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    storage = Storage(db_path)
    imported = import_legacy_leases(storage, spec, legacy_files)
    upgrade = UpgradeManager(storage, spec).run()
    engine = LeaseEngine(spec, storage)
    # Bootstrap a view, then exercise the whole-view replacement contract.
    engine.rebuild_subscription_view(reason="initial_bootstrap")
    trace = run_changes(engine, changes_path)
    diff = engine.rebuild_subscription_view(reason="post_run_rebuild")
    return {"imported": imported, "upgrade": upgrade.to_dict(),
            "trace": trace,
            "rebuild_diff": _diff_dict(diff),
            "view": [dict(r) for r in storage.list_subscription_view()],
            "cursors": storage.get_all_cursors()}


def _diff_dict(diff: Any) -> dict[str, Any]:
    return {"added": diff.added, "removed": diff.removed, "upgraded": diff.upgraded,
            "cursor_changes": diff.cursor_changes,
            "unknown_renewals": diff.unknown_renewals}
