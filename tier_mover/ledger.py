"""SQLite metadata ledger with versioned snapshots and per-object transactions."""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .errors import VersionConflictError
from .models import Manifest

SCHEMA_VERSION = 1


def ledger_path(workdir: str | Path) -> Path:
    return Path(workdir) / "ledger.db"


def snapshot_dir(workdir: str | Path) -> Path:
    return Path(workdir) / "snapshots"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # Full durability: every committed per-object transaction is crash safe.
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class Ledger:
    def __init__(self, conn: sqlite3.Connection, workdir: Path) -> None:
        self.conn = conn
        self.workdir = workdir

    # ------------------------------------------------------------------ schema
    @staticmethod
    def create(workdir: str | Path, manifest: Manifest) -> "Ledger":
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        path = ledger_path(workdir)
        if path.exists():
            raise VersionConflictError(
                "ledger already exists; use open() or replace_snapshot()",
                {"path": str(path)},
            )
        conn = _connect(path)
        ledger = Ledger(conn, workdir)
        ledger._init_schema()
        ledger._write_manifest(manifest, state="fresh")
        for tier in sorted(manifest.tiers, key=lambda t: t.tier_id):
            conn.execute(
                "INSERT INTO tier_state(tier_id, role, capacity_bytes, unit_cost, "
                "read_weight, write_weight, evict_priority, used_bytes, free_bytes, "
                "object_count, storage_cost) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tier.tier_id,
                    tier.role,
                    tier.capacity_bytes,
                    tier.unit_cost,
                    tier.read_weight,
                    tier.write_weight,
                    tier.evict_priority,
                    0,
                    tier.capacity_bytes,
                    0,
                    0,
                ),
            )
        conn.execute(
            "INSERT INTO progress(cursor_kind, last_committed_seq, last_committed_key, "
            "unknown_object) VALUES (?,?,?,?)",
            ("none", 0, None, None),
        )
        conn.commit()
        return ledger

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE tier_state (
                tier_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                capacity_bytes INTEGER NOT NULL,
                unit_cost INTEGER NOT NULL,
                read_weight INTEGER NOT NULL,
                write_weight INTEGER NOT NULL,
                evict_priority INTEGER NOT NULL,
                used_bytes INTEGER NOT NULL,
                free_bytes INTEGER NOT NULL,
                object_count INTEGER NOT NULL,
                storage_cost INTEGER NOT NULL
            );
            CREATE TABLE object_state (
                key TEXT PRIMARY KEY,
                size_bytes INTEGER NOT NULL,
                digest TEXT NOT NULL,
                tier_id TEXT,
                heat INTEGER NOT NULL,
                last_access_ms INTEGER NOT NULL,
                access_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                updated_cycle INTEGER NOT NULL
            );
            CREATE TABLE cycle (
                cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                tier_param_version TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE TABLE migration (
                cycle_id INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                key TEXT NOT NULL,
                action TEXT NOT NULL,
                direction TEXT NOT NULL,
                source_tier TEXT,
                target_tier TEXT,
                reason TEXT NOT NULL,
                digest TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                move_cost INTEGER NOT NULL,
                src_used_before INTEGER NOT NULL,
                dst_used_before INTEGER NOT NULL,
                src_used_after INTEGER NOT NULL,
                dst_used_after INTEGER NOT NULL,
                PRIMARY KEY (cycle_id, seq)
            );
            CREATE TABLE progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cursor_kind TEXT NOT NULL,
                last_committed_seq INTEGER NOT NULL,
                last_committed_key TEXT,
                unknown_object TEXT
            );
            CREATE TABLE event_log (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER,
                event TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self.conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?,?)",
            [
                ("schema_version", str(SCHEMA_VERSION)),
                ("created_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            ],
        )
        self.conn.commit()

    def _write_manifest(self, manifest: Manifest, state: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
            ("policy_version", manifest.policy_version),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
            ("tier_param_version", manifest.tier_param_version),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
            ("manifest_json", json.dumps(manifest.to_dict(), sort_keys=True)),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)",
            ("state", state),
        )
        self.conn.commit()

    # ------------------------------------------------------------------- open
    @staticmethod
    def open(workdir: str | Path) -> "Ledger":
        path = ledger_path(workdir)
        if not path.exists():
            raise FileNotFoundError(path)
        conn = _connect(path)
        return Ledger(conn, Path(workdir))

    def close(self) -> None:
        self.conn.close()

    def versions(self) -> Dict[str, str]:
        rows = {
            r["key"]: r["value"]
            for r in self.conn.execute("SELECT key, value FROM meta").fetchall()
        }
        return {
            "policy_version": rows.get("policy_version", ""),
            "tier_param_version": rows.get("tier_param_version", ""),
            "schema_version": rows.get("schema_version", ""),
            "state": rows.get("state", ""),
        }

    def stored_manifest(self) -> Manifest:
        from .manifest import _validate  # reuse the same strong validator

        raw = json.loads(
            self.conn.execute("SELECT value FROM meta WHERE key='manifest_json'").fetchone()[0]
        )
        tiers, objects, rule, policy_version, tier_param_version, injections = _validate(raw)
        return Manifest(
            policy_version=policy_version,
            tier_param_version=tier_param_version,
            tiers=tiers,
            objects=objects,
            rule=rule,
            injection_sequence=injections,
        )

    def check_versions(self, manifest: Manifest) -> None:
        current = self.versions()
        if (
            current["policy_version"] != manifest.policy_version
            or current["tier_param_version"] != manifest.tier_param_version
        ):
            raise VersionConflictError(
                "ledger snapshot version does not match session manifest",
                {
                    "snapshot_policy_version": current["policy_version"],
                    "session_policy_version": manifest.policy_version,
                    "snapshot_tier_param_version": current["tier_param_version"],
                    "session_tier_param_version": manifest.tier_param_version,
                },
            )

    def diff_versions(self, manifest: Manifest) -> Dict[str, Any]:
        """Stable list of changed tiers and objects vs the stored snapshot."""
        old = self.stored_manifest()
        tier_changes: List[Dict[str, Any]] = []
        old_tiers = {t.tier_id: t for t in old.tiers}
        new_tiers = {t.tier_id: t for t in manifest.tiers}
        for tier_id in sorted(set(old_tiers) | set(new_tiers)):
            if tier_id not in old_tiers:
                tier_changes.append({"tier_id": tier_id, "change": "added"})
            elif tier_id not in new_tiers:
                tier_changes.append({"tier_id": tier_id, "change": "removed"})
            else:
                before = old_tiers[tier_id].to_dict()
                after = new_tiers[tier_id].to_dict()
                fields = sorted(key for key in before if before[key] != after[key])
                if fields:
                    tier_changes.append(
                        {
                            "tier_id": tier_id,
                            "change": "modified",
                            "fields": fields,
                            "before": {key: before[key] for key in fields},
                            "after": {key: after[key] for key in fields},
                        }
                    )
        old_objs = {o.key: o for o in old.objects}
        new_objs = {o.key: o for o in manifest.objects}
        object_changes: List[Dict[str, Any]] = []
        for key in sorted(set(old_objs) | set(new_objs)):
            if key not in old_objs:
                object_changes.append({"key": key, "change": "added"})
            elif key not in new_objs:
                object_changes.append({"key": key, "change": "removed"})
            else:
                before = old_objs[key].to_dict()
                after = new_objs[key].to_dict()
                fields = sorted(k for k in before if before[k] != after[k])
                if fields:
                    object_changes.append(
                        {
                            "key": key,
                            "change": "modified",
                            "fields": fields,
                            "before": {k: before[k] for k in fields},
                            "after": {k: after[k] for k in fields},
                        }
                    )
        return {
            "snapshot_policy_version": old.policy_version,
            "session_policy_version": manifest.policy_version,
            "snapshot_tier_param_version": old.tier_param_version,
            "session_tier_param_version": manifest.tier_param_version,
            "tier_changes": tier_changes,
            "object_changes": object_changes,
        }

    def replace_snapshot(self, manifest: Manifest) -> Dict[str, Any]:
        """Archive the old snapshot read-only and install a fresh replacement.

        This is a whole-state replacement (never an incremental merge): tier
        parameters, object ownership and ledger state are rebuilt from the
        current manifest while the previous database is preserved read-only.
        """
        diff = self.diff_versions(manifest)
        self.conn.close()
        source = ledger_path(self.workdir)
        snapshot_dir(self.workdir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = snapshot_dir(self.workdir) / f"ledger.{stamp}.db"
        shutil.copy2(source, archived)
        archived.chmod(0o444)
        source.unlink()
        sidecars = [
            source.with_suffix(".db-wal"),
            source.with_suffix(".db-shm"),
        ]
        for sidecar in sidecars:
            if sidecar.exists():
                sidecar.unlink()
        new_ledger = Ledger.create(self.workdir, manifest)
        new_ledger.log_event(
            "snapshot_replaced",
            archived_snapshot=str(archived),
            diff=diff,
        )
        return {"archived_snapshot": str(archived), "diff": diff}

    # --------------------------------------------------------------- mutation
    def log_event(self, event: str, **payload: Any) -> None:
        self.conn.execute(
            "INSERT INTO event_log(cycle_id, event, payload_json) VALUES (?,?,?)",
            (payload.get("cycle_id"), event, json.dumps(payload, sort_keys=True)),
        )
        self.conn.commit()

    def start_cycle(self, phase: str, manifest: Manifest, plan: Mapping[str, Any]) -> int:
        cur = self.conn.execute(
            "INSERT INTO cycle(phase, policy_version, tier_param_version, plan_json, started_at) "
            "VALUES (?,?,?,?,?)",
            (
                phase,
                manifest.policy_version,
                manifest.tier_param_version,
                json.dumps(plan, sort_keys=True),
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
        )
        cycle_id = int(cur.lastrowid)
        self.conn.commit()
        return cycle_id

    def finish_cycle(self, cycle_id: int) -> None:
        self.conn.execute(
            "UPDATE cycle SET finished_at=? WHERE cycle_id=?",
            (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), cycle_id),
        )
        self.conn.commit()

    def get_tier(self, tier_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM tier_state WHERE tier_id=?", (tier_id,)
        ).fetchone()
        if row is None:
            raise KeyError(tier_id)
        return row

    def _upsert_object(self, manifest: Manifest, key: str, tier_id: Optional[str],
                       status: str, cycle_id: int) -> None:
        from .heat import heat_value

        obj = manifest.object_by_key(key)
        self.conn.execute(
            "INSERT INTO object_state(key, size_bytes, digest, tier_id, heat, "
            "last_access_ms, access_count, status, updated_cycle) "
            "VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET size_bytes=excluded.size_bytes, "
            "digest=excluded.digest, tier_id=excluded.tier_id, heat=excluded.heat, "
            "last_access_ms=excluded.last_access_ms, access_count=excluded.access_count, "
            "status=excluded.status, updated_cycle=excluded.updated_cycle",
            (
                obj.key,
                obj.size_bytes,
                obj.digest,
                tier_id,
                heat_value(obj),
                obj.last_access_ms,
                obj.access_count,
                status,
                cycle_id,
            ),
        )

    def _refresh_tier_cost(self, tier_id: str) -> None:
        row = self.get_tier(tier_id)
        used = row["used_bytes"]
        capacity = row["capacity_bytes"]
        self.conn.execute(
            "UPDATE tier_state SET free_bytes=?, storage_cost=? WHERE tier_id=?",
            (capacity - used, used * row["unit_cost"], tier_id),
        )

    def record_placement(
        self,
        cycle_id: int,
        seq: int,
        manifest: Manifest,
        action: Mapping[str, Any],
    ) -> None:
        """One committed initial-placement object (single transaction)."""
        target_id = action["target_tier"]
        target = self.get_tier(target_id)
        size_bytes = int(action["size_bytes"])
        with self.conn:
            dst_before = target["used_bytes"]
            self.conn.execute(
                "UPDATE tier_state SET used_bytes=?, object_count=? WHERE tier_id=?",
                (dst_before + size_bytes, target["object_count"] + 1, target_id),
            )
            self._refresh_tier_cost(target_id)
            move_cost = size_bytes * target["unit_cost"] * target["write_weight"]
            self.conn.execute(
                "INSERT INTO migration(cycle_id, seq, key, action, direction, "
                "source_tier, target_tier, reason, digest, size_bytes, move_cost, "
                "src_used_before, dst_used_before, src_used_after, dst_used_after) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cycle_id, seq, action["key"], action["action"], action["direction"],
                    None, target_id, action["reason"], action["digest"], size_bytes,
                    move_cost, 0, dst_before, 0, dst_before + size_bytes,
                ),
            )
            self._upsert_object(manifest, action["key"], target_id, "committed", cycle_id)
            self.conn.execute(
                "UPDATE progress SET cursor_kind=?, last_committed_seq=?, "
                "last_committed_key=?, unknown_object=? WHERE id=1",
                ("placement", seq, action["key"], None),
            )

    def mark_inflight(
        self, cycle_id: int, manifest: Manifest, action: Mapping[str, Any]
    ) -> None:
        """Record an in-flight object *before* its physical switch.

        This is its own commit, deliberately separate from the per-object
        completion transaction.  A forced kill leaves status='in_flight' and an
        unknown progress cursor for recovery to resolve.
        """
        self._upsert_object(manifest, action["key"], None, "in_flight", cycle_id)
        self.conn.execute(
            "UPDATE progress SET cursor_kind=?, last_committed_key=?, unknown_object=? WHERE id=1",
            ("in_flight", action["key"], action["key"]),
        )
        self.conn.commit()

    def record_migration(
        self,
        cycle_id: int,
        seq: int,
        manifest: Manifest,
        action: Mapping[str, Any],
    ) -> None:
        """One committed object move: record, occupancy and cursor in one txn."""
        source_id = action["source_tier"]
        target_id = action["target_tier"]
        size_bytes = int(action["size_bytes"])
        source = self.get_tier(source_id)
        target = self.get_tier(target_id)
        with self.conn:
            src_before = source["used_bytes"]
            dst_before = target["used_bytes"]
            src_after = src_before - size_bytes
            dst_after = dst_before + size_bytes
            self.conn.execute(
                "UPDATE tier_state SET used_bytes=?, object_count=? WHERE tier_id=?",
                (src_after, max(0, source["object_count"] - 1), source_id),
            )
            self.conn.execute(
                "UPDATE tier_state SET used_bytes=?, object_count=? WHERE tier_id=?",
                (dst_after, target["object_count"] + 1, target_id),
            )
            self._refresh_tier_cost(source_id)
            self._refresh_tier_cost(target_id)
            move_cost = (
                size_bytes
                * target["unit_cost"]
                * (source["read_weight"] + target["write_weight"])
            )
            self.conn.execute(
                "INSERT INTO migration(cycle_id, seq, key, action, direction, "
                "source_tier, target_tier, reason, digest, size_bytes, move_cost, "
                "src_used_before, dst_used_before, src_used_after, dst_used_after) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cycle_id, seq, action["key"], action["action"], action["direction"],
                    source_id, target_id, action["reason"], action["digest"], size_bytes,
                    move_cost, src_before, dst_before, src_after, dst_after,
                ),
            )
            self._upsert_object(manifest, action["key"], target_id, "committed", cycle_id)
            self.conn.execute(
                "UPDATE progress SET cursor_kind=?, last_committed_seq=?, "
                "last_committed_key=?, unknown_object=? WHERE id=1",
                ("migration", seq, action["key"], None),
            )

    # -------------------------------------------------------------- queries
    def tier_rows(self) -> List[sqlite3.Row]:
        return list(
            self.conn.execute("SELECT * FROM tier_state ORDER BY tier_id").fetchall()
        )

    def object_rows(self) -> List[sqlite3.Row]:
        return list(
            self.conn.execute("SELECT * FROM object_state ORDER BY key").fetchall()
        )

    def ownership(self) -> Dict[str, Optional[str]]:
        return {
            row["key"]: row["tier_id"]
            for row in self.conn.execute("SELECT key, tier_id FROM object_state ORDER BY key")
        }

    def in_flight_objects(self) -> List[str]:
        return [
            row["key"]
            for row in self.conn.execute(
                "SELECT key FROM object_state WHERE status='in_flight' ORDER BY key"
            ).fetchall()
        ]

    def unfinished_cycle(self) -> Optional[Dict[str, Any]]:
        """Return the most recent cycle that never reached finish_cycle()."""
        row = self.conn.execute(
            "SELECT cycle_id, phase, plan_json FROM cycle "
            "WHERE finished_at IS NULL ORDER BY cycle_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "cycle_id": row["cycle_id"],
            "phase": row["phase"],
            "plan": json.loads(row["plan_json"]),
        }

    def rollback_unfinished_cycle(self, cycle_id: int) -> List[Dict[str, Any]]:
        """Remove partial records of an interrupted, non-atomic cycle.

        Per-object commits already made stay durable on their own, but the
        cycle as a logical unit never completed.  Its partial migration rows
        are detached so replay only sees whole, reproducible cycles.  Physical
        ownership is reconciled separately by recovery digest verification.
        """
        rows = [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM migration WHERE cycle_id=? ORDER BY seq", (cycle_id,)
            ).fetchall()
        ]
        with self.conn:
            self.conn.execute("DELETE FROM migration WHERE cycle_id=?", (cycle_id,))
            self.conn.execute("DELETE FROM cycle WHERE cycle_id=?", (cycle_id,))
            self.conn.execute(
                "UPDATE object_state SET status='committed' WHERE status='in_flight'"
            )
            self.conn.execute(
                "UPDATE progress SET cursor_kind=?, unknown_object=? WHERE id=1",
                ("recovered", None),
            )
        return rows

    def reconcile_tier_usage(self, manifest: Manifest, ownership: Mapping[str, Optional[str]]) -> None:
        """Rebuild per-tier usage/count/cost from reconciled object ownership."""
        sizes = {obj.key: obj.size_bytes for obj in manifest.objects}
        usage = {tier.tier_id: 0 for tier in manifest.tiers}
        counts = {tier.tier_id: 0 for tier in manifest.tiers}
        for key, tier_id in ownership.items():
            if tier_id is None:
                continue
            usage[tier_id] += sizes[key]
            counts[tier_id] += 1
        with self.conn:
            for tier in manifest.tiers:
                tier_id = tier.tier_id
                self.conn.execute(
                    "UPDATE tier_state SET used_bytes=?, object_count=?, free_bytes=?, "
                    "storage_cost=? WHERE tier_id=?",
                    (
                        usage[tier_id],
                        counts[tier_id],
                        tier.capacity_bytes - usage[tier_id],
                        usage[tier_id] * tier.unit_cost,
                        tier_id,
                    ),
                )

    def progress(self) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM progress WHERE id=1").fetchone()
        return dict(row) if row else {}

    def migrations(self) -> List[Dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM migration ORDER BY cycle_id, seq"
            ).fetchall()
        ]

    def cycles(self) -> List[Dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT cycle_id, phase, policy_version, tier_param_version, "
                "plan_json, started_at, finished_at FROM cycle ORDER BY cycle_id"
            ).fetchall()
        ]

    def total_move_cost(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(move_cost),0) AS total FROM migration"
        ).fetchone()
        return int(row["total"])

    def object_cost_breakdown(self) -> List[Dict[str, Any]]:
        """Per-object, per-migration cost trail (fully back-traceable)."""
        return [
            {
                "key": row["key"],
                "cycle_id": row["cycle_id"],
                "seq": row["seq"],
                "action": row["action"],
                "target_tier": row["target_tier"],
                "size_bytes": row["size_bytes"],
                "move_cost": row["move_cost"],
            }
            for row in self.conn.execute(
                "SELECT key, cycle_id, seq, action, target_tier, size_bytes, move_cost "
                "FROM migration ORDER BY key, cycle_id, seq"
            ).fetchall()
        ]

    def recovery_status(self) -> Dict[str, Any]:
        progress = self.progress()
        return {
            "cursor_kind": progress.get("cursor_kind"),
            "last_committed_seq": progress.get("last_committed_seq"),
            "last_committed_key": progress.get("last_committed_key"),
            "unknown_object": progress.get("unknown_object"),
            "in_flight_objects": self.in_flight_objects(),
            "committed_migrations": len(self.migrations()),
        }

    def export_conclusions(self) -> Dict[str, Any]:
        """Deterministic, hashable summary of the whole ledger."""
        versions = self.versions()
        tiers = [
            {
                "tier_id": row["tier_id"],
                "role": row["role"],
                "capacity_bytes": row["capacity_bytes"],
                "used_bytes": row["used_bytes"],
                "free_bytes": row["free_bytes"],
                "object_count": row["object_count"],
                "unit_cost": row["unit_cost"],
                "storage_cost": row["storage_cost"],
            }
            for row in self.tier_rows()
        ]
        objects = [
            {
                "key": row["key"],
                "tier_id": row["tier_id"],
                "size_bytes": row["size_bytes"],
                "digest": row["digest"],
                "heat": row["heat"],
                "status": row["status"],
            }
            for row in self.object_rows()
        ]
        migrations = [
            {
                "cycle_id": row["cycle_id"],
                "seq": row["seq"],
                "key": row["key"],
                "action": row["action"],
                "direction": row["direction"],
                "source_tier": row["source_tier"],
                "target_tier": row["target_tier"],
                "reason": row["reason"],
                "digest": row["digest"],
                "size_bytes": row["size_bytes"],
                "move_cost": row["move_cost"],
                "src_used_before": row["src_used_before"],
                "dst_used_before": row["dst_used_before"],
                "src_used_after": row["src_used_after"],
                "dst_used_after": row["dst_used_after"],
            }
            for row in self.migrations()
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "policy_version": versions["policy_version"],
            "tier_param_version": versions["tier_param_version"],
            "state": versions["state"],
            "tiers": tiers,
            "objects": objects,
            "migrations": migrations,
            "object_cost_breakdown": self.object_cost_breakdown(),
            "total_move_cost": self.total_move_cost(),
            "recovery_status": self.recovery_status(),
        }
