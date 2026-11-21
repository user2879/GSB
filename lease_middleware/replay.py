"""Deterministic consistency replay from the append-only ledger."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .errors import E_REPLAY, MiddlewareError
from .spec import Spec
from .storage import LEGAL_TRANSITIONS, Storage

PUSH = "config_push"


@dataclass
class LeaseReplay:
    lease_id: str
    client_id: str
    config_key: str
    state: str
    lease_version: int
    renewal_count: int
    last_digest: str | None = None


@dataclass
class ClientReplay:
    cursor: int = 0
    last_digest: str | None = None
    renewal_versions: list[int] = field(default_factory=list)
    renewal_counts: list[int] = field(default_factory=list)
    pushed_versions: list[int] = field(default_factory=list)
    pushed_by_key: dict[str, list[int]] = field(default_factory=dict)


class Replayer:
    def __init__(self, storage: Storage, spec: Spec):
        self.storage = storage
        self.spec = spec
        self.leases: dict[str, LeaseReplay] = {}
        self.clients: dict[str, ClientReplay] = {}

    def _client(self, cid: str) -> ClientReplay:
        return self.clients.setdefault(cid, ClientReplay())

    def run(self) -> dict[str, Any]:
        for row in self.storage.list_ledger():
            self._apply(row)
        self._cross_check()
        return self.summary()

    # ------------------------------------------------------------ apply one
    def _apply(self, row: Any) -> None:
        etype = row["event_type"]
        cid, lid = row["client_id"], row["lease_id"]
        if etype == "lease_granted":
            if lid in self.leases:
                raise self._fail(row, "lease_id", "duplicate grant of lease")
            self.leases[lid] = LeaseReplay(
                lease_id=lid, client_id=cid, config_key=row["config_key"],
                state="granted", lease_version=row["lease_version"],
                renewal_count=int(row["renewal_count"] or 0),
                last_digest=row["digest"])
        elif etype == "lease_renewed":
            lease = self._require_lease(row)
            self._check_transition(row, lease.state, "renewed")
            client = self._client(cid)
            if client.renewal_counts and row["renewal_count"] <= client.renewal_counts[-1]:
                raise self._fail(row, "renewal_count",
                                 "renewal_count not strictly increasing")
            if client.renewal_versions and row["lease_version"] < client.renewal_versions[-1]:
                raise self._fail(row, "lease_version",
                                 "renewal lease_version regressed")
            client.renewal_versions.append(row["lease_version"])
            client.renewal_counts.append(row["renewal_count"])
            lease.state = "renewed"
            lease.renewal_count = row["renewal_count"]
            lease.last_digest = row["digest"]
        elif etype == "lease_expired":
            lease = self._require_lease(row)
            self._check_transition(row, lease.state, "expired")
            lease.state = "expired"
        elif etype == "lease_revoked":
            lease = self._require_lease(row)
            self._check_transition(row, lease.state, "revoked")
            lease.state = "revoked"
        elif etype == PUSH:
            lease = self._require_lease(row)
            if lease.state not in ("granted", "renewed"):
                raise self._fail(row, "state", "push to non-active lease")
            client = self._client(cid)
            version = row["config_version"]
            key = row["config_key"]
            per_key = client.pushed_by_key.setdefault(key, [])
            if version in per_key:
                raise self._fail(row, "config_version",
                                 "duplicate push of same config version for key")
            per_key.append(version)
            client.pushed_versions.append(version)
            client.cursor = max(client.cursor, version)
            client.last_digest = row["digest"]
        elif etype == "view_rebuilt":
            return  # structural event, no per-lease state effect
        else:
            raise self._fail(row, "event_type", f"unknown event {etype!r}")

    def _require_lease(self, row: Any) -> LeaseReplay:
        lid = row["lease_id"]
        if lid not in self.leases:
            raise self._fail(row, "lease_id", "event for unknown lease")
        return self.leases[lid]

    def _check_transition(self, row: Any, current: str, target: str) -> None:
        if target not in LEGAL_TRANSITIONS[current]:
            raise self._fail(row, "state", f"illegal transition {current}->{target}")

    def _fail(self, row: Any, field_name: str, message: str) -> MiddlewareError:
        return MiddlewareError(
            E_REPLAY, message, field_name, client_id=row["client_id"],
            lease_id=row["lease_id"], context={"ledger_seq": row["seq"],
                                               "event_type": row["event_type"]})

    # ----------------------------------------------------- reconcile storage
    def _cross_check(self) -> None:
        stored = {r["lease_id"]: r for r in self.storage.list_lease_rows()}
        if set(stored) != set(self.leases):
            only_ledger = sorted(set(self.leases) - set(stored))
            only_db = sorted(set(stored) - set(self.leases))
            raise MiddlewareError(
                E_REPLAY, "lease set differs between ledger and lease table", "lease_id",
                context={"only_in_ledger": only_ledger, "only_in_db": only_db})

        for lid in sorted(self.leases):
            replay_lease = self.leases[lid]
            db_row = stored[lid]
            record = json.loads(db_row["record_json"])
            for field_name, replay_val, db_val in (
                ("state", replay_lease.state, db_row["state"]),
                ("lease_version", replay_lease.lease_version, record["lease_version"]),
                ("renewal_count", replay_lease.renewal_count, record["renewal_count"]),
                ("value_digest", replay_lease.last_digest, record["value_digest"]),
            ):
                if replay_val != db_val:
                    raise MiddlewareError(
                        E_REPLAY, f"field {field_name} mismatch", field_name,
                        client_id=replay_lease.client_id, lease_id=lid,
                        context={"replay": replay_val, "stored": db_val})

        cursors = self.storage.get_all_cursors()
        for cid in sorted(set(cursors) | set(self.clients)):
            replay_cursor = self.clients[cid].cursor if cid in self.clients else 0
            if cursors.get(cid, 0) != replay_cursor:
                raise MiddlewareError(
                    E_REPLAY, "seen_config_version cursor mismatch",
                    "seen_config_version", client_id=cid,
                    context={"replay": replay_cursor,
                             "stored": cursors.get(cid, 0)})

    def summary(self) -> dict[str, Any]:
        return {
            "leases": {
                lid: {"client_id": l.client_id, "config_key": l.config_key,
                      "state": l.state, "lease_version": l.lease_version,
                      "renewal_count": l.renewal_count,
                      "last_digest": l.last_digest}
                for lid, l in sorted(self.leases.items())
            },
            "clients": {
                cid: {"seen_config_version": c.cursor,
                      "last_digest": c.last_digest,
                      "renewal_versions": c.renewal_versions,
                      "renewal_counts": c.renewal_counts,
                      "pushed_versions": c.pushed_versions,
                      "pushed_by_key": c.pushed_by_key}
                for cid, c in sorted(self.clients.items())
            },
        }
