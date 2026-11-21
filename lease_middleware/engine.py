"""Lease state machine, push pipeline and whole-view rebuild."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from .clock import Clock, Instant
from .codec import LeaseRecord, make_lease_id
from .errors import E_LEASE, E_VERSION, MiddlewareError
from .spec import Spec
from .storage import LEGAL_TRANSITIONS, Storage

GRANTED = "granted"
RENEWED = "renewed"
EXPIRED = "expired"
REVOKED = "revoked"
ACTIVE_STATES = (GRANTED, RENEWED)


@dataclass
class PendingRenewal:
    lease_id: str
    client_id: str
    config_key: str


@dataclass
class RebuildDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    upgraded: list[str] = field(default_factory=list)
    cursor_changes: list[dict[str, Any]] = field(default_factory=list)
    unknown_renewals: list[dict[str, str]] = field(default_factory=list)


class LeaseEngine:
    def __init__(self, spec: Spec, storage: Storage, clock: Clock | None = None):
        self.spec = spec
        self.storage = storage
        self.clock = clock or Clock()
        self._lock = threading.RLock()
        # Renewals held only in memory until the next durable commit.
        self._uncommitted: dict[str, PendingRenewal] = {}
        self._sweeper: threading.Thread | None = None
        self._stop = threading.Event()
        self._seq_by_key: dict[tuple[str, str], int] = self._init_seqs()

    def _init_seqs(self) -> dict[tuple[str, str], int]:
        seqs: dict[tuple[str, str], int] = {}
        for row in self.storage.conn.execute(
                "SELECT lease_id FROM lease_records").fetchall():
            parts = row["lease_id"].split(":")
            if len(parts) == 4 and parts[0] == "L":
                try:
                    seq = int(parts[3])
                except ValueError:
                    continue
                key = (parts[1], parts[2])
                seqs[key] = max(seqs.get(key, 0), seq)
        return seqs

    # --------------------------------------------------------------- helpers
    def _deadline_mono(self, now: Instant, duration_ms: int) -> float:
        return now.mono + duration_ms / 1000.0

    def _active_lease(self, lease_id: str) -> LeaseRecord:
        row = self.storage.get_lease_row(lease_id)
        if row is None:
            raise MiddlewareError(E_LEASE, "lease does not exist", "lease_id",
                                  lease_id=lease_id)
        if row["state"] not in ACTIVE_STATES:
            raise MiddlewareError(E_LEASE, f"lease is {row['state']}, not operable",
                                  "state", lease_id=lease_id)
        return LeaseRecord(**_record_kwargs(row))

    def _transition(self, current: str, target: str, lease_id: str) -> None:
        if target not in LEGAL_TRANSITIONS[current]:
            raise MiddlewareError(E_LEASE,
                                  f"illegal transition {current} -> {target}",
                                  "state", lease_id=lease_id)

    # ----------------------------------------------------------------- grant
    def grant(self, client_id: str, config_key: str,
              config_version: int | None = None) -> LeaseRecord:
        with self._lock:
            client = self.spec.clients.get(client_id)
            if client is None:
                raise MiddlewareError(E_LEASE, "unknown client", "client_id",
                                      client_id=client_id)
            if config_key not in client.subscribed_keys:
                raise MiddlewareError(E_LEASE, "client does not subscribe to key",
                                      "subscribed_keys", client_id=client_id)
            config = self.spec.require_config(config_key)
            seq_key = (client_id, config_key)
            self._seq_by_key[seq_key] = self._seq_by_key.get(seq_key, 0) + 1
            lease_id = make_lease_id(client_id, config_key,
                                     self._seq_by_key[seq_key])
            now = self.clock.now()
            version = config.current_version if config_version is None else config_version
            if version not in config._digests:
                raise MiddlewareError(E_LEASE,
                                      f"config version {version} is not declared",
                                      "lease_version", client_id=client_id)
            record = LeaseRecord(
                lease_id=lease_id,
                config_key=config_key,
                expires_at_ms=now.epoch_ms + client.lease_duration_ms,
                lease_version=version,
                client_id=client_id,
                renewal_count=0,
                value_digest=config.digest_at(version),
                parent_lease_id=None,
            )
            deadline = self._deadline_mono(now, client.lease_duration_ms)
            with self.storage.transaction():
                self.storage.upsert_lease(record, GRANTED, deadline)
                self.storage.append_ledger(
                    "lease_granted", created_at_ms=now.epoch_ms, client_id=client_id,
                    lease_id=lease_id, config_key=config_key, lease_version=version,
                    state=GRANTED, renewal_count=0, digest=record.value_digest,
                    payload={"expires_at_ms": record.expires_at_ms})
                self.storage.mark_view_entry(client_id, config_key, GRANTED, lease_id)
            return record

    # ---------------------------------------------------------------- renew
    def renew(self, lease_id: str, expected_version: int, *,
              commit: bool = True) -> LeaseRecord:
        with self._lock:
            row = self.storage.get_lease_row(lease_id)
            if row is None:
                raise MiddlewareError(E_LEASE, "lease does not exist", "lease_id",
                                      lease_id=lease_id)
            self._sweep_one(row)
            row = self.storage.get_lease_row(lease_id)
            record = LeaseRecord(**_record_kwargs(row))
            if row["state"] not in ACTIVE_STATES:
                raise MiddlewareError(E_LEASE, f"lease is {row['state']}, cannot renew",
                                      "state", lease_id=lease_id)
            # Optimistic concurrency on lease version.
            if record.lease_version != expected_version:
                raise MiddlewareError(
                    E_VERSION, "lease version mismatch; renewal rejected",
                    "lease_version", lease_id=lease_id, client_id=record.client_id,
                    context={"expected": expected_version,
                             "actual": record.lease_version})
            client = self.spec.clients[record.client_id]
            now = self.clock.now()
            renewed = LeaseRecord(
                lease_id=record.lease_id,
                config_key=record.config_key,
                expires_at_ms=now.epoch_ms + client.lease_duration_ms,
                lease_version=record.lease_version,
                client_id=record.client_id,
                renewal_count=record.renewal_count + 1,
                value_digest=record.value_digest,
                parent_lease_id=record.parent_lease_id,
            )
            deadline = self._deadline_mono(now, client.lease_duration_ms)
            if not commit:
                self._uncommitted[lease_id] = PendingRenewal(
                    lease_id, record.client_id, record.config_key)
                return renewed
            self._transition(row["state"], RENEWED, lease_id)
            with self.storage.transaction():
                self.storage.upsert_lease(renewed, RENEWED, deadline)
                self.storage.append_ledger(
                    "lease_renewed", created_at_ms=now.epoch_ms,
                    client_id=record.client_id, lease_id=lease_id,
                    config_key=record.config_key, lease_version=renewed.lease_version,
                    state=RENEWED, renewal_count=renewed.renewal_count,
                    digest=renewed.value_digest,
                    payload={"expires_at_ms": renewed.expires_at_ms})
                self.storage.mark_view_entry(record.client_id, record.config_key,
                                             RENEWED, lease_id)
            return renewed

    # --------------------------------------------------------------- expire
    def _sweep_one(self, row: Any) -> bool:
        """Real monotonic-clock expiry check for a single stored row."""
        deadline = row["mono_deadline"]
        if row["state"] in ACTIVE_STATES and deadline is not None \
                and self.clock.monotonic() >= deadline:
            now = self.clock.now()
            self._transition(row["state"], EXPIRED, row["lease_id"])
            with self.storage.transaction():
                self.storage.update_lease_state(row["lease_id"], EXPIRED)
                self.storage.append_ledger(
                    "lease_expired", created_at_ms=now.epoch_ms,
                    client_id=row["client_id"], lease_id=row["lease_id"],
                    config_key=row["config_key"], state=EXPIRED,
                    payload={"mono_deadline": deadline})
                # Reclaim and prominently mark the subscription view entry.
                self.storage.mark_view_entry(row["client_id"], row["config_key"],
                                             EXPIRED, None)
            return True
        return False

    def sweep_expired(self) -> list[str]:
        with self._lock:
            expired: list[str] = []
            for row in self.storage.list_lease_rows():
                if self._sweep_one(row):
                    expired.append(row["lease_id"])
            return expired

    def start_sweeper(self, interval_seconds: float = 0.05) -> None:
        if self._sweeper and self._sweeper.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.wait(interval_seconds):
                try:
                    self.sweep_expired()
                except Exception:  # pragma: no cover - defensive thread guard
                    pass

        self._sweeper = threading.Thread(target=_loop, name="lease-sweeper", daemon=True)
        self._sweeper.start()

    def stop_sweeper(self) -> None:
        self._stop.set()
        if self._sweeper:
            self._sweeper.join(timeout=1.0)

    # --------------------------------------------------------------- revoke
    def revoke(self, lease_id: str) -> None:
        with self._lock:
            row = self.storage.get_lease_row(lease_id)
            if row is None:
                raise MiddlewareError(E_LEASE, "lease does not exist", "lease_id",
                                      lease_id=lease_id)
            if row["state"] in (EXPIRED, REVOKED):
                raise MiddlewareError(E_LEASE, f"lease already {row['state']}",
                                      "state", lease_id=lease_id)
            now = self.clock.now()
            self._transition(row["state"], REVOKED, lease_id)
            with self.storage.transaction():
                self.storage.update_lease_state(lease_id, REVOKED)
                self.storage.append_ledger(
                    "lease_revoked", created_at_ms=now.epoch_ms,
                    client_id=row["client_id"], lease_id=lease_id,
                    config_key=row["config_key"], state=REVOKED)
                self.storage.mark_view_entry(row["client_id"], row["config_key"],
                                             REVOKED, None)

    # -------------------------------------------------------- config publish
    def publish_config(self, config_key: str, new_version: int,
                       new_value: Any) -> dict[str, Any]:
        """Publish a new value and push atomically to valid lease holders."""
        with self._lock:
            from hashlib import sha256
            from .spec import canonical_json

            config = self.spec.require_config(config_key)
            if new_version not in config._digests:
                raise MiddlewareError(E_LEASE,
                                      f"config version {new_version} is not declared",
                                      "config_version", lease_id=None)
            expected_digest = config.digest_at(new_version)
            digest = sha256(canonical_json(new_value).encode("utf-8")).hexdigest()
            if digest != expected_digest:
                raise MiddlewareError(E_LEASE,
                                      "published value disagrees with declared value",
                                      "value", lease_id=None)
            now = self.clock.now()
            self.sweep_expired()
            pushed: list[dict[str, Any]] = []

            # Whole config version advances once per publish.
            self.storage.set_meta(f"config_version:{config_key}", new_version)
            for row in self.storage.list_lease_rows():
                if row["config_key"] != config_key or row["state"] not in ACTIVE_STATES:
                    continue
                client_id = row["client_id"]
                lease_id = row["lease_id"]
                # Ledger row + cursor advance happen in ONE transaction.
                with self.storage.transaction():
                    seq = self.storage.append_ledger(
                        "config_push", created_at_ms=now.epoch_ms, client_id=client_id,
                        lease_id=lease_id, config_key=config_key,
                        config_version=new_version,
                        lease_version=json.loads(row["record_json"])["lease_version"],
                        digest=digest,
                        payload={"value": new_value})
                    cursor = self.storage.get_cursor(client_id)
                    if cursor > new_version:
                        raise MiddlewareError(E_VERSION, "cursor ahead of push",
                                              "seen_config_version", client_id=client_id)
                    new_cursor = max(cursor, new_version)
                    self.storage.set_cursor(client_id, new_cursor)
                    self.storage.assert_cursor_within_ledger(client_id, new_cursor)
                    self.storage.conn.execute(
                        "UPDATE subscription_view SET seen_config_version=? "
                        "WHERE client_id=? AND config_key=?",
                        (new_cursor, client_id, config_key))
                pushed.append({"client_id": client_id, "lease_id": lease_id,
                               "config_version": new_version, "ledger_seq": seq,
                               "seen_config_version": new_cursor})
            return {"config_key": config_key, "config_version": new_version,
                    "digest": digest, "pushed": pushed}

    # -------------------------------------------------------- view rebuild
    def rebuild_subscription_view(self, reason: str,
                                  new_spec: Spec | None = None) -> RebuildDiff:
        """Whole-view replacement on lease-set / config / client-set evolution."""
        with self._lock:
            spec = new_spec or self.spec
            old_cursors = self.storage.get_all_cursors()
            old_view = {(r["client_id"], r["config_key"]): dict(r)
                        for r in self.storage.list_subscription_view()}
            old_clients = set(old_cursors) | {r["client_id"] for r in old_view.values()}
            new_clients = set(spec.clients)

            diff = RebuildDiff()
            diff.added = sorted(new_clients - old_clients)
            diff.removed = sorted(old_clients - new_clients)
            diff.upgraded = sorted(
                cid for cid in new_clients & old_clients
                if spec.clients[cid].format_version != self.spec.clients[cid].format_version
            ) if new_spec else []

            # Renewed-but-not-durable renewals are unknown and must be re-granted.
            for pending in list(self._uncommitted.values()):
                diff.unknown_renewals.append({
                    "lease_id": pending.lease_id,
                    "client_id": pending.client_id,
                    "config_key": pending.config_key,
                    "resolution": "re_grant_required",
                })
            self._uncommitted.clear()

            new_rows: list[tuple[str, str, str | None, str, int]] = []
            new_cursors: dict[str, int] = {}
            active = {(r["client_id"], r["config_key"]): r
                      for r in self.storage.list_lease_rows()}
            for cid in sorted(new_clients):
                client = spec.clients[cid]
                # Rebuild cursor purely from ledger pushes (never invented).
                row = self.storage.conn.execute(
                    "SELECT MAX(config_version) AS m FROM ledger "
                    "WHERE event_type='config_push' AND client_id=?", (cid,)).fetchone()
                cursor = int(row["m"] or 0)
                new_cursors[cid] = cursor
                old_value = old_cursors.get(cid, 0)
                if old_value != cursor:
                    diff.cursor_changes.append(
                        {"client_id": cid, "old_cursor": old_value,
                         "new_cursor": cursor})
                for key in sorted(client.subscribed_keys):
                    lease_row = active.get((cid, key))
                    if lease_row and lease_row["state"] in ACTIVE_STATES:
                        state, lease_id = lease_row["state"], lease_row["lease_id"]
                    elif lease_row:
                        state, lease_id = lease_row["state"], None
                    else:
                        state, lease_id = "ungranted", None
                    new_rows.append((cid, key, lease_id, state, cursor))

            with self.storage.transaction():
                self.storage.replace_subscription_view(new_rows)
                self.storage.replace_all_cursors(new_cursors)
                self.storage.append_ledger(
                    "view_rebuilt", created_at_ms=self.clock.now().epoch_ms,
                    payload={"reason": reason,
                             "added": diff.added, "removed": diff.removed,
                             "upgraded": diff.upgraded,
                             "unknown_renewals": diff.unknown_renewals,
                             "cursor_changes": diff.cursor_changes})
            if new_spec is not None:
                self.spec = new_spec
            return diff

    def uncommitted_renewals(self) -> list[dict[str, str]]:
        with self._lock:
            return [vars(p) for p in self._uncommitted.values()]


def _record_kwargs(row: Any) -> dict[str, Any]:
    import json as _json
    raw = _json.loads(row["record_json"])
    # Drop the persisted format marker so the dataclass default (r3) applies.
    raw.pop("format", None)
    return raw
