"""Session snapshots: versioned, content-addressed whole-state replacement.

A snapshot captures the domain-set version, the fixed-point scale calibre
version, every domain's active leases and ordered queue, the latest per-domain
watermark, and the ledger progress cursor, plus a SHA-256 content digest.

On reopen, any version or digest mismatch against the current declaration causes
a *whole replacement* (never an incremental merge) and the stable list of
differing fields is returned. After replacement, replay compares the effective
lease set, queue order, watermarks and quantisation error bound against the
ledger, event by event; the first mismatch is reported as ``E_REPLAY`` with the
exact event seq, domain id and field.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import ReplayError, SnapshotError
from .models import SCALE_CALIBRE_VERSION

SNAPSHOT_KIND = "distlock-session-snapshot"
SNAPSHOT_VERSION = 1


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def compute_digest(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "digest"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def write_snapshot(path: str | Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["kind"] = SNAPSHOT_KIND
    payload["snapshot_version"] = SNAPSHOT_VERSION
    payload["scale_calibre_version"] = SCALE_CALIBRE_VERSION
    payload["digest"] = compute_digest(payload)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def read_snapshot(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SnapshotError("snapshot file not found",
                            {"path": str(p), "field": "<file>"})
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError("snapshot JSON parse error",
                            {"path": str(p), "line": exc.lineno, "field": exc.msg})
    stored = raw.get("digest")
    actual = compute_digest(raw)
    if not isinstance(stored, str) or stored != actual:
        raise SnapshotError(
            "snapshot content digest mismatch",
            {"path": str(p), "field": "digest",
             "stored": stored, "actual": actual})
    return raw


def diff_declaration(snap: dict[str, Any], *, domain_set_version: int,
                     scale_factor: int,
                     current_domain_ids: list[str]) -> list[str]:
    """Stable, sorted list of differing top-level fields (whole replacement)."""
    diffs: list[str] = []
    if snap.get("domain_set_version") != domain_set_version:
        diffs.append("domain_set_version")
    if snap.get("scale_calibre_version") != SCALE_CALIBRE_VERSION:
        diffs.append("scale_calibre_version")
    if snap.get("scale_factor") != scale_factor:
        diffs.append("scale_factor")
    snap_ids = sorted((snap.get("domains") or {}).keys())
    if snap_ids != sorted(current_domain_ids):
        diffs.append("domains[].domain_id")
    for did in sorted(set(snap_ids) & set(current_domain_ids)):
        snap_dom = (snap.get("domains") or {}).get(did, {})
    return sorted(set(diffs))


def verify_or_replace(path: str | Path, *, domain_set_version: int,
                      scale_factor: int, current_domain_ids: list[str]
                      ) -> tuple[dict[str, Any] | None, list[str]]:
    """Return (snapshot_or_None, diff_fields). Mismatch -> replace (None)."""
    try:
        snap = read_snapshot(path)
    except SnapshotError as exc:
        if exc.code == "E_SNAPSHOT" and exc.context.get("field") == "digest":
            return None, ["digest"]
        raise
    diffs = diff_declaration(
        snap, domain_set_version=domain_set_version,
        scale_factor=scale_factor, current_domain_ids=current_domain_ids)
    if diffs:
        # Whole replacement: discard snapshot instead of merging.
        return None, diffs
    return snap, []


# --------------------------------------------------------------------- replay
def _norm_active(entries: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(
        (e["holder_id"], e["state"], e["started_ms"], e["expires_ms"])
        for e in entries)


def _norm_queue(entries: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    # Queue order is significant, so keep position rather than sorting.
    return [(e["seq"], e["holder_id"], e["priority"]) for e in entries]


def verify_replay(snapshot: dict[str, Any], ledger_rows: dict[str, Any],
                  domains: dict[str, Any], *, up_to_seq: int) -> None:
    """Compare snapshot state against replayed ledger state, first mismatch wins.

    ``ledger_rows`` is produced by :func:`replay_ledger_state` and contains the
    deterministic effective lease set, ordered queue, watermark and error bound
    per domain at seq ``up_to_seq``.
    """
    snap_domains = snapshot.get("domains", {})
    for did in sorted(domains):
        snap_dom = snap_domains.get(did, {})
        led = ledger_rows["domains"].get(did)
        if led is None:
            raise ReplayError("replay state missing domain",
                              {"seq": up_to_seq, "domain_id": did,
                               "field": "<domain>"})

        snap_active = _norm_active(snap_dom.get("active_leases", []))
        led_active = _norm_active(led["active_leases"])
        if snap_active != led_active:
            raise ReplayError(
                "effective lease set diverges from ledger",
                {"seq": up_to_seq, "domain_id": did,
                 "field": "active_leases", "snapshot": snap_active,
                 "ledger": led_active})

        snap_queue = _norm_queue(snap_dom.get("queue", []))
        led_queue = _norm_queue(led["queue"])
        if snap_queue != led_queue:
            raise ReplayError(
                "queue order diverges from ledger",
                {"seq": up_to_seq, "domain_id": did, "field": "queue",
                 "snapshot": snap_queue, "ledger": led_queue})

        snap_wm = snap_dom.get("watermark", {})
        for field in ("active_leases", "free_capacity", "queue_length",
                      "ratio_fixed", "error_bound_fixed"):
            if snap_wm.get(field) != led["watermark"].get(field):
                raise ReplayError(
                    "watermark diverges from ledger",
                    {"seq": up_to_seq, "domain_id": did, "field": field,
                     "snapshot": snap_wm.get(field),
                     "ledger": led["watermark"].get(field)})

    if int(snapshot.get("ledger_cursor", -1)) != int(ledger_rows["cursor"]):
        raise ReplayError(
            "ledger cursor diverges from snapshot",
            {"seq": up_to_seq, "domain_id": None, "field": "ledger_cursor",
             "snapshot": snapshot.get("ledger_cursor"),
             "ledger": ledger_rows["cursor"]})


def replay_ledger_state(receipts: list[dict[str, Any]],
                        transitions: list[dict[str, Any]],
                        watermarks: list[dict[str, Any]],
                        domains: dict[str, Any],
                        holder_priority: dict[str, int], *,
                        up_to_seq: int) -> dict[str, Any]:
    """Deterministically reconstruct state purely from committed ledger rows.

    Active leases follow the explicit transition table. The ordered waiter set
    is derived from receipts whose verdict is QUEUED, minus waiters that were
    subsequently granted, released or expired; it is then ordered with the
    identical priority-desc / seq-asc rule the arbiters use, so the result does
    not depend on dictionary iteration order.
    """
    active: dict[str, dict[str, dict[str, Any]]] = {did: {} for did in domains}
    queued: dict[str, dict[int, dict[str, Any]]] = {did: {} for did in domains}
    removed_waiters: dict[str, set[str]] = {did: set() for did in domains}

    in_scope = lambda tr: tr["seq"] <= up_to_seq  # noqa: E731
    for tr in transitions:
        if not in_scope(tr):
            break
        did = tr["domain_id"]
        hid = tr["holder_id"]
        to_state = tr["state_to"]
        if to_state in ("granted", "renewed"):
            active[did][hid] = {
                "holder_id": hid, "state": to_state,
                "started_ms": tr["started_ms"], "expires_ms": tr["expires_ms"]}
            removed_waiters[did].add(hid)
            queued[did].pop(next((s for s, w in queued[did].items()
                                  if w["holder_id"] == hid), -1), None)
        elif to_state in ("expired", "released"):
            active[did].pop(hid, None)
            removed_waiters[did].add(hid)
        elif to_state == "pending":
            removed_waiters[did].discard(hid)

    # Derive authoritative queue membership from committed receipts.
    for rc in receipts:
        if rc["seq"] > up_to_seq:
            break
        did = rc["domain_id"]
        hid = rc["holder_id"]
        if rc["verdict"] == "QUEUED" and hid not in removed_waiters[did] \
                and hid not in active[did]:
            queued[did][rc["seq"]] = {
                "seq": rc["seq"], "holder_id": hid,
                "priority": holder_priority.get(hid, 0)}
        if rc["verdict"] in ("GRANTED", "RENEWED", "RELEASED",
                             "RENEW_EXPIRED"):
            queued[did].pop(rc["seq"], None)

    ordered_queues = {
        did: sorted(queued_by.values(),
                    key=lambda w: (-w["priority"], w["seq"]))
        for did, queued_by in queued.items()}

    latest_wm: dict[str, dict[str, Any]] = {}
    for wm in watermarks:
        if wm["seq"] <= up_to_seq:
            latest_wm[wm["domain_id"]] = wm

    state: dict[str, dict[str, Any]] = {}
    for did, dom in sorted(domains.items()):
        wm = latest_wm.get(did)
        state[did] = {
            "active_leases": sorted(active[did].values(),
                                    key=lambda e: e["holder_id"]),
            "queue": ordered_queues[did],
            "watermark": {
                "active_leases": wm["active_leases"],
                "free_capacity": wm["free_capacity"],
                "queue_length": wm["queue_length"],
                "ratio_fixed": wm["ratio_fixed"],
                "error_bound_fixed": wm["error_bound_fixed"]} if wm else {
                "active_leases": 0, "free_capacity": dom.capacity,
                "queue_length": 0, "ratio_fixed": 0,
                "error_bound_fixed": 1},
        }
    return {"cursor": up_to_seq, "domains": state}


def _norm_active(entries: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(
        (e["holder_id"], e["state"], e["started_ms"], e["expires_ms"])
        for e in entries)


def _norm_queue(entries: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    # Queue order is significant, so keep position rather than sorting.
    return [(e["seq"], e["holder_id"], e["priority"]) for e in entries]


def verify_replay(snapshot: dict[str, Any], ledger_rows: dict[str, Any],
                  domains: dict[str, Any], *, up_to_seq: int) -> None:
    """Compare snapshot state against replayed ledger state, first mismatch wins.

    ``ledger_rows`` is produced by :func:`replay_ledger_state` and contains the
    deterministic effective lease set, ordered queue, watermark and error bound
    per domain at seq ``up_to_seq``.
    """
    snap_domains = snapshot.get("domains", {})
    for did in sorted(domains):
        snap_dom = snap_domains.get(did, {})
        led = ledger_rows["domains"].get(did)
        if led is None:
            raise ReplayError("replay state missing domain",
                              {"seq": up_to_seq, "domain_id": did,
                               "field": "<domain>"})

        snap_active = _norm_active(snap_dom.get("active_leases", []))
        led_active = _norm_active(led["active_leases"])
        if snap_active != led_active:
            raise ReplayError(
                "effective lease set diverges from ledger",
                {"seq": up_to_seq, "domain_id": did,
                 "field": "active_leases", "snapshot": snap_active,
                 "ledger": led_active})

        snap_queue = _norm_queue(snap_dom.get("queue", []))
        led_queue = _norm_queue(led["queue"])
        if snap_queue != led_queue:
            raise ReplayError(
                "queue order diverges from ledger",
                {"seq": up_to_seq, "domain_id": did, "field": "queue",
                 "snapshot": snap_queue, "ledger": led_queue})

        snap_wm = snap_dom.get("watermark", {})
        for field in ("active_leases", "free_capacity", "queue_length",
                      "ratio_fixed", "error_bound_fixed"):
            if snap_wm.get(field) != led["watermark"].get(field):
                raise ReplayError(
                    "watermark diverges from ledger",
                    {"seq": up_to_seq, "domain_id": did, "field": field,
                     "snapshot": snap_wm.get(field),
                     "ledger": led["watermark"].get(field)})

    if int(snapshot.get("ledger_cursor", -1)) != int(ledger_rows["cursor"]):
        raise ReplayError(
            "ledger cursor diverges from snapshot",
            {"seq": up_to_seq, "domain_id": None, "field": "ledger_cursor",
             "snapshot": snapshot.get("ledger_cursor"),
             "ledger": ledger_rows["cursor"]})


def replay_ledger_state(receipts: list[dict[str, Any]],
                        transitions: list[dict[str, Any]],
                        watermarks: list[dict[str, Any]],
                        domains: dict[str, Any],
                        holder_priority: dict[str, int], *,
                        up_to_seq: int) -> dict[str, Any]:
    """Deterministically reconstruct state purely from committed ledger rows.

    The reconstruction follows the explicit transition table, so it does not
    depend on dictionary iteration order; queue ordering follows the same
    priority-desc / seq-asc rule used by the arbiters.
    """
    state: dict[str, dict[str, Any]] = {}
    for did, dom in sorted(domains.items()):
        state[did] = {
            "active_leases": [],
            "queue": [],
            "watermark": {"active_leases": 0, "free_capacity": dom.capacity,
                          "queue_length": 0, "ratio_fixed": 0,
                          "error_bound_fixed": 1},
        }

    active: dict[str, dict[str, dict[str, Any]]] = {did: {} for did in domains}
    queues: dict[str, list[dict[str, Any]]] = {did: [] for did in domains}

    for tr in transitions:
        if tr["seq"] > up_to_seq:
            break
        did = tr["domain_id"]
        hid = tr["holder_id"]
        to_state = tr["state_to"]
        if to_state in ("granted", "renewed"):
            active[did][hid] = {
                "holder_id": hid, "state": to_state,
                "started_ms": tr["started_ms"], "expires_ms": tr["expires_ms"]}
            queues[did] = [w for w in queues[did] if w["holder_id"] != hid]
        elif to_state in ("expired", "released"):
            active[did].pop(hid, None)
            if to_state == "released":
                queues[did] = [w for w in queues[did] if w["holder_id"] != hid]
        elif to_state == "pending":
            queues[did].append({
                "seq": tr["seq"], "holder_id": hid,
                "priority": holder_priority.get(hid, 0)})
            queues[did].sort(key=lambda w: (-w["priority"], w["seq"]))

    # Latest committed watermark per domain at or below the cursor.
    latest_wm: dict[str, dict[str, Any]] = {}
    for wm in watermarks:
        if wm["seq"] <= up_to_seq:
            latest_wm[wm["domain_id"]] = wm
    for did, dom in sorted(domains.items()):
        wm = latest_wm.get(did)
        state[did]["active_leases"] = sorted(
            active[did].values(), key=lambda e: e["holder_id"])
        state[did]["queue"] = list(queues[did])
        if wm:
            state[did]["watermark"] = {
                "active_leases": wm["active_leases"],
                "free_capacity": wm["free_capacity"],
                "queue_length": wm["queue_length"],
                "ratio_fixed": wm["ratio_fixed"],
                "error_bound_fixed": wm["error_bound_fixed"]}

    return {"cursor": up_to_seq, "domains": state}
