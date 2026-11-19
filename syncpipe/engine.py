"""Online authoritative synthesizer.

The engine consumes deterministically ordered tick groups, advances one
authority version per handled event, suspends events whose
``base_snapshot_version`` does not match the current authority version
(E_BASE_VERSION) for later catch-up, converges field-level conflicts by
commit-order LWW and persists an append-only snapshot/log chain.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .errors import (
    E_BASE_VERSION,
    E_CONFLICT_UNRESOLVED,
    make_error,
)
from .state import (
    adjudicate_tick,
    apply_event,
    build_snapshot,
    event_ref,
    initial_state,
    state_digest,
)
from .storage import KEEP_ANCHOR_EVERY, Storage

GENESIS_VERSION = 0
SNAPSHOT_EVERY = 2


class Engine:
    def __init__(self, storage: Storage, seed: int):
        self.storage = storage
        self.seed = seed
        self.version = GENESIS_VERSION
        self.tick = 0
        self.state: Dict[str, Any] = initial_state()
        self.manifest: Dict[str, Any] = {}
        self.hash_chain: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.conflicts: List[Dict[str, Any]] = []
        self.suspended: List[Dict[str, Any]] = []
        self._suspended_keys = set()

    # ---------- lifecycle ----------

    def initialize(self) -> None:
        self.storage.ensure_dirs()
        self.storage.remove_shutdown_marker()
        self.manifest = self.storage.init_manifest(self.seed)
        genesis = build_snapshot(
            version=GENESIS_VERSION,
            tick=0,
            parent_version=None,
            state=initial_state(),
            source_event=None,
        )
        self.storage.write_snapshot(genesis)
        self._record_snapshot(genesis)
        self.storage.write_manifest(self.manifest)

    def _record_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.manifest["snapshot_files"].append(
            {"version": snapshot["version"], "file": Storage.snapshot_filename(snapshot["version"])}
        )
        self.manifest["versions"].append(
            {
                "version": snapshot["version"],
                "tick": snapshot["tick"],
                "parent_version": snapshot["parent_version"],
                "state_hash": snapshot["state_hash"],
                "snapshot_hash": snapshot["snapshot_hash"],
            }
        )
        self.manifest["head_version"] = snapshot["version"]

    def _record_version(
        self,
        event: Optional[Dict[str, Any]],
        status: str,
        new_state: Dict[str, Any],
        new_tick: int,
        error_code: Optional[str] = None,
        conflict_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.version += 1
        self.tick = new_tick
        self.state = new_state
        parent = self.version - 1
        snapshot = build_snapshot(
            version=self.version,
            tick=new_tick,
            parent_version=parent,
            state=copy.deepcopy(new_state),
            source_event=event_ref(event) if event else None,
        )
        log_entry = {
            "version": self.version,
            "tick": new_tick,
            "parent_version": parent,
            "event": event,
            "status": status,
            "state_hash": snapshot["state_hash"],
        }
        if error_code:
            log_entry["error_code"] = error_code
        if conflict_records:
            log_entry["conflicts"] = conflict_records
        self.storage.append_log_entry(log_entry)
        if self.version % SNAPSHOT_EVERY == 0:
            self.storage.write_snapshot(snapshot)
            self._record_snapshot(snapshot)
        else:
            self.manifest["versions"].append(
                {
                    "version": self.version,
                    "tick": new_tick,
                    "parent_version": parent,
                    "state_hash": snapshot["state_hash"],
                    "snapshot_hash": snapshot["snapshot_hash"],
                }
            )
            self.manifest["head_version"] = self.version
        self.hash_chain.append(
            {
                "version": self.version,
                "tick": new_tick,
                "state_hash": snapshot["state_hash"],
                "snapshot_hash": snapshot["snapshot_hash"],
                "status": status,
                "event": event_ref(event) if event else None,
            }
        )
        return snapshot

    # ---------- main loop ----------

    def run(self, tick_groups: List[List[Dict[str, Any]]], parser_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.initialize()
        for record in parser_errors:
            self.storage.append_error(record)
            self.errors.append(record)

        # One cross-tick worklist. Each tick merges its events in, then we run
        # to a fixed point: repeatedly apply the commit-order first event whose
        # base matches the current authority until no event is ready. Events
        # that remain are revisited when the next tick adds more events, which
        # is what lets a lagging peer catch up instead of deadlocking.
        worklist: Dict[Tuple[str, int], Dict[str, Any]] = {}
        conflict_by_tick: Dict[int, List[Dict[str, Any]]] = {}
        for group in tick_groups:
            conflict_records = adjudicate_tick(group)
            self.conflicts.extend(conflict_records)
            conflict_by_tick[group[0]["tick"]] = conflict_records
            for event in group:
                worklist[(event["peer_id"], event["local_seq"])] = event
            self._drain_worklist(worklist, conflict_by_tick)
            self.storage.write_manifest(self.manifest)

        self.suspended = list(worklist.values())
        self._suspended_keys = set(worklist)
        for event in self.suspended:
            record = make_error(
                E_BASE_VERSION,
                f"event never caught up: base={event['base_snapshot_version']} head={self.version}",
                version=self.version,
                tick=event["tick"],
                event_loc=(event["peer_id"], event["local_seq"]),
            )
            self.errors.append(record)
            self.storage.append_error(record)

        head_snapshot = self._ensure_head_snapshot()
        self.storage.write_manifest(self.manifest)
        self.storage.close_log()
        return self._summary(head_snapshot)

    def _drain_worklist(
        self,
        worklist: Dict[Tuple[str, int], Dict[str, Any]],
        conflict_by_tick: Dict[int, List[Dict[str, Any]]],
    ) -> None:
        commit_order = lambda e: (e["tick"], e["peer_id"], e["local_seq"])
        while worklist:
            ready = [
                event for event in worklist.values()
                if event["base_snapshot_version"] == self.version
            ]
            if ready:
                event = min(ready, key=commit_order)
            else:
                # No event matches the current authority. Apply the
                # commit-order first lagging event as the version driver so
                # the authority keeps advancing; record the base drift as
                # E_BASE_VERSION (never silently reorder or drop). Ahead-base
                # events are preferred only when nothing is lagging.
                lagging = [
                    e for e in worklist.values()
                    if e["base_snapshot_version"] < self.version
                ]
                pool = lagging or list(worklist.values())
                event = min(pool, key=commit_order)
                record = make_error(
                    E_BASE_VERSION,
                    f"base drift driver: base={event['base_snapshot_version']} "
                    f"head={self.version}",
                    version=self.version,
                    tick=event["tick"],
                    event_loc=(event["peer_id"], event["local_seq"]),
                )
                self.errors.append(record)
                self.storage.append_error(record)
            del worklist[(event["peer_id"], event["local_seq"])]
            self._handle_event(event, conflict_by_tick.get(event["tick"], []))

    def _handle_event(self, event: Dict[str, Any], tick_conflicts: List[Dict[str, Any]]) -> None:
        related = [
            record
            for record in tick_conflicts
            if record["tick"] == event["tick"]
            and record["winner"]["peer_id"] == event["peer_id"]
            and record["winner"]["local_seq"] == event["local_seq"]
        ]
        try:
            new_state = apply_event(self.state, event)
        except Exception as exc:  # PipelineError by contract
            error_code = getattr(exc, "code", E_CONFLICT_UNRESOLVED)
            record = make_error(
                error_code,
                str(exc),
                version=self.version + 1,
                tick=event["tick"],
                event_loc=(event["peer_id"], event["local_seq"]),
            )
            self.errors.append(record)
            self.storage.append_error(record)
            self._record_version(
                event,
                status="rejected",
                new_state=copy.deepcopy(self.state),
                new_tick=event["tick"],
                error_code=error_code,
                conflict_records=related,
            )
            return
        self._record_version(
            event,
            status="applied",
            new_state=new_state,
            new_tick=event["tick"],
            conflict_records=related,
        )

    def _ensure_head_snapshot(self) -> Dict[str, Any]:
        available = set(self.storage.available_snapshot_versions())
        if self.version not in available:
            snapshot = build_snapshot(
                version=self.version,
                tick=self.tick,
                parent_version=self.version - 1,
                state=copy.deepcopy(self.state),
                source_event=self.hash_chain[-1]["event"] if self.hash_chain else None,
            )
            self.storage.write_snapshot(snapshot)
            self.manifest["snapshot_files"].append(
                {"version": snapshot["version"], "file": Storage.snapshot_filename(snapshot["version"])}
            )
            for item in self.manifest["versions"]:
                if item["version"] == snapshot["version"]:
                    item["snapshot_hash"] = snapshot["snapshot_hash"]
                    item["state_hash"] = snapshot["state_hash"]
            self.storage.write_manifest(self.manifest)
            return snapshot
        return self.storage.read_snapshot_file(self.version)

    # ---------- compaction ----------

    def compact(self) -> Dict[str, Any]:
        """Collapse intermediate snapshots while keeping reconstructibility.

        Retained anchors: genesis, every KEEP_ANCHOR_EVERY versions and the
        current head. Every removed version remains fully reconstructible
        from an ancestor anchor plus its op-log entries; version numbers and
        hashes in the manifest are untouched (continuous semantics preserved).
        """
        available = self.storage.available_snapshot_versions()
        head = self.manifest["head_version"]
        kept = {GENESIS_VERSION, head}
        kept.update(v for v in available if v % KEEP_ANCHOR_EVERY == 0)
        removed = []
        for version in available:
            if version in kept:
                continue
            self.storage.remove_snapshot_file(version)
            removed.append(version)
        compaction = {
            "kept_snapshots": sorted(kept & set(available)),
            "removed_snapshots": sorted(removed),
            "head_version": head,
        }
        self.manifest["compactions"].append(compaction)
        self.manifest["snapshot_files"] = [
            item
            for item in self.manifest["snapshot_files"]
            if item["version"] not in set(removed)
        ]
        self.storage.write_manifest(self.manifest)
        return compaction

    # ---------- report ----------

    def _summary(self, head_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "head_version": self.version,
            "head_tick": self.tick,
            "head_state_hash": head_snapshot["state_hash"],
            "head_snapshot_hash": head_snapshot["snapshot_hash"],
            "version_count": len(self.hash_chain),
            "applied": sum(1 for item in self.hash_chain if item["status"] == "applied"),
            "rejected": sum(1 for item in self.hash_chain if item["status"] == "rejected"),
            "suspended_remaining": len(self.suspended),
            "conflict_count": len(self.conflicts),
            "conflicts": self.conflicts,
            "errors": self.errors,
            "hash_chain": self.hash_chain,
        }

    def write_shutdown(self) -> None:
        self.storage.write_shutdown_marker(self.manifest)
        self.manifest["shutdown"] = True
        self.storage.write_manifest(self.manifest)


def rebuild_from_anchor(
    storage: Storage, target_version: int, anchor_version: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Rebuild any target version from an ancestor snapshot + op logs.

    When ``anchor_version`` is omitted the nearest retained ancestor is used.
    Returns ``(snapshot, state, log_entries_replayed)`` and is used both by
    incremental catch-up and by the independent validator.
    """
    entries, truncation_errors = storage.read_all_log_entries()
    entries = [entry for entry in entries if entry["version"] <= target_version]
    anchors = [v for v in storage.available_snapshot_versions() if v <= target_version]
    if not anchors:
        raise FileNotFoundError(f"no ancestor snapshot for target v{target_version}")
    if anchor_version is None:
        anchor_version = max(anchors)
    elif anchor_version not in anchors:
        raise FileNotFoundError(f"anchor v{anchor_version} not retained")
    snapshot = storage.read_snapshot_file(anchor_version)
    state = copy.deepcopy(snapshot["state"])
    replayed: List[Dict[str, Any]] = []
    for entry in entries:
        if entry["version"] <= anchor_version:
            continue
        if entry["status"] == "applied" and entry.get("event") is not None:
            state = apply_event(state, entry["event"])
        if state_digest(state) != entry["state_hash"]:
            raise RuntimeError(f"replay hash mismatch at v{entry['version']}")
        replayed.append(entry)
    rebuilt_snapshot = build_snapshot(
        version=target_version,
        tick=entries[-1]["tick"] if entries else snapshot["tick"],
        parent_version=target_version - 1,
        state=state,
        source_event=entries[-1]["event"] if entries else snapshot.get("source_event"),
    )
    return rebuilt_snapshot, state, replayed
