"""Coordination engine: persistence, leases, elections, propagation and snapshots."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .errors import (
    CoordError,
    dependency_error,
    protocol_error,
    replay_error,
    snapshot_error,
)
from .model import (
    graph_from_members,
    normalize_member,
    normalize_ref,
    normalize_state,
    stable_downstream,
    state_from_definition,
    validate_graph,
)
from .util import (
    JsonlLogger,
    atomic_write_json,
    is_int,
    monotonic_ms,
    read_json,
    stable_hash,
    wall_ms,
)


class CoordinatorEngine:
    """A single-process coordination server with an in-memory atomic view."""

    def __init__(
        self,
        root: str | Path,
        *,
        definition: Any | None = None,
        mark_unknown_on_load: bool = False,
        monotonic_clock: Callable[[], int] = monotonic_ms,
        wall_clock: Callable[[], int] = wall_ms,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.logger = JsonlLogger(self.root / "logs" / "engine.jsonl")
        self._lock = threading.RLock()
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._deadlines: dict[str, int] = {}

        loaded_existing = self.state_path.exists()
        try:
            if loaded_existing:
                raw = read_json(self.state_path)
                self._state = normalize_state(raw)
            elif definition is not None:
                self._state = state_from_definition(definition)
            else:
                self._state = state_from_definition({"term": 1, "members": []})
        except CoordError as exc:
            self.logger.write(
                "load_rejected",
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
            raise
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.logger.write("load_rejected", code="E_STATE", message=str(exc), details={})
            raise CoordError("E_STATE", f"cannot load state: {exc}") from exc

        self._rebuild_deadlines_locked()
        if not loaded_existing:
            self._refresh_election_locked(force_unknown=True)
            self._persist_locked()
            self.logger.write(
                "state_initialized",
                state_version=self._state["state_version"],
                term=self._state["term"],
                members=sorted(self._state["members"]),
            )
        elif mark_unknown_on_load:
            self._state["votes"] = {}
            self._state["election_requires_rerun"] = True
            self._state["election"] = {
                "status": "unknown",
                "leader": None,
                "term": self._state["term"],
                "eligible": [],
                "vote_counts": {},
                "view_version": self._state["election"].get("view_version", 0) + 1,
                "requires_rerun": True,
            }
            self._touch_locked()
            self._persist_locked()
            self.logger.write(
                "election_unknown_after_restart",
                state_version=self._state["state_version"],
                term=self._state["term"],
                message="votes cleared; nodes must rerun the election",
            )

    # ------------------------------------------------------------------
    # Persistence and basic accessors
    # ------------------------------------------------------------------
    @property
    def state_version(self) -> int:
        with self._lock:
            return int(self._state["state_version"])

    @property
    def term(self) -> int:
        with self._lock:
            return int(self._state["term"])

    def state_copy(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def _touch_locked(self) -> None:
        self._state["state_version"] += 1

    def _persist_locked(self) -> None:
        atomic_write_json(self.state_path, self._state)

    def _now_pair(self) -> tuple[int, int]:
        return int(self._monotonic_clock()), int(self._wall_clock())

    def _rebuild_deadlines_locked(self) -> None:
        now_mono, now_wall = self._now_pair()
        self._deadlines = {}
        for member_id, member in self._state["members"].items():
            remaining = int(member["lease_expiry_ms"]) - now_wall
            if member["lease_valid"] and remaining > 0:
                self._deadlines[member_id] = now_mono + remaining
            else:
                self._deadlines[member_id] = now_mono

    def _graph_snapshot_locked(self) -> dict[str, list[str]]:
        return graph_from_members(self._state["members"])

    def _token_health_locked(self) -> dict[str, bool]:
        return {
            token_id: bool(token["healthy"])
            for token_id, token in sorted(self._state["tokens"].items())
        }

    def _election_result_locked(self) -> dict[str, Any]:
        return deepcopy(self._state["election"])

    # ------------------------------------------------------------------
    # Eligibility, election and votes
    # ------------------------------------------------------------------
    def _dependencies_healthy_locked(self, member: dict[str, Any]) -> bool:
        for reference in member["dependencies"]:
            kind, _, identifier = reference.partition(":")
            if kind == "token":
                token = self._state["tokens"].get(identifier)
                if token is None or not token["healthy"]:
                    return False
            elif kind == "member":
                upstream = self._state["members"].get(identifier)
                if upstream is None:
                    return False
                if not (
                    upstream["status"] == "active"
                    and upstream["lease_valid"]
                    and upstream["candidate_eligible"]
                ):
                    return False
            else:
                return False
        return True

    def _refresh_election_locked(self, *, force_unknown: bool = False) -> None:
        previous_version = int(self._state.get("election", {}).get("view_version", 0))
        eligible = sorted(
            member_id
            for member_id, member in self._state["members"].items()
            if member["candidate_eligible"]
            and member["lease_valid"]
            and member["status"] == "active"
        )
        term_key = str(self._state["term"])
        counts = {member_id: 0 for member_id in eligible}
        for voter in sorted(self._state["votes"].get(term_key, {})):
            record = self._state["votes"][term_key][voter]
            candidate = record.get("candidate")
            if record.get("granted") and candidate in counts:
                counts[candidate] += 1
        requires_rerun = bool(self._state.get("election_requires_rerun", False))
        unknown = force_unknown or requires_rerun or not eligible
        leader = None
        if not unknown:
            leader = sorted(eligible, key=lambda item: (-counts[item], item))[0]
        if requires_rerun or force_unknown:
            counts = {}
        self._state["election"] = {
            "status": "unknown" if unknown else "complete",
            "leader": leader,
            "term": self._state["term"],
            "eligible": eligible,
            "vote_counts": dict(sorted(counts.items())),
            "view_version": previous_version + 1,
            "requires_rerun": bool(unknown and (requires_rerun or force_unknown or not eligible)),
        }
        if unknown and not eligible:
            self._state["election_requires_rerun"] = True

    def _campaign_records_locked(self, term: int) -> dict[str, dict[str, Any]]:
        key = str(term)
        if key not in self._state["votes"]:
            self._state["votes"][key] = {}
        return self._state["votes"][key]

    def vote(self, term: int, member_id: str, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            self._reap_expired_locked(self._now_pair()[0], persist=True)
            if term != self._state["term"]:
                raise CoordError(
                    "E_TERM",
                    "vote term does not match the coordination term",
                    expected=self._state["term"],
                    received=term,
                )
            voter = self._state["members"].get(member_id)
            candidate = self._state["members"].get(candidate_id)
            if voter is None:
                raise CoordError("E_MEMBER", "voter member does not exist", member=member_id)
            if candidate is None:
                raise CoordError("E_MEMBER", "candidate member does not exist", member=candidate_id)

            records = self._campaign_records_locked(term)
            recorded = records.get(member_id)
            if recorded is not None:
                return {
                    "term": term,
                    "member": member_id,
                    "candidate": recorded["candidate"],
                    "granted": bool(recorded["granted"]),
                    "idempotent": True,
                    "reason": recorded.get("reason", "already_voted"),
                    "election": self._election_result_locked(),
                    "state_version": self._state["state_version"],
                }

            voter_eligible = (
                voter["status"] == "active"
                and voter["lease_valid"]
                and voter["candidate_eligible"]
            )
            candidate_eligible = (
                candidate["status"] == "active"
                and candidate["lease_valid"]
                and candidate["candidate_eligible"]
            )
            granted = voter_eligible and candidate_eligible
            reason = None
            if not voter_eligible:
                reason = "voter_ineligible"
            elif not candidate_eligible:
                reason = "candidate_ineligible"
            if granted:
                records[member_id] = {
                    "candidate": candidate_id,
                    "granted": True,
                    "reason": None,
                }
                self._state["election_requires_rerun"] = False
                self._refresh_election_locked()
                self._touch_locked()
                self._persist_locked()
                self.logger.write(
                    "vote_recorded",
                    term=term,
                    member=member_id,
                    candidate=candidate_id,
                    granted=True,
                    reason=None,
                    state_version=self._state["state_version"],
                )
            return {
                "term": term,
                "member": member_id,
                "candidate": candidate_id,
                "granted": granted,
                "idempotent": False,
                "reason": reason,
                "election": self._election_result_locked(),
                "state_version": self._state["state_version"],
            }

    # ------------------------------------------------------------------
    # Lease heartbeat and deterministic expiration
    # ------------------------------------------------------------------
    def heartbeat(self, term: int, member_id: str, lease_version: int) -> dict[str, Any]:
        with self._lock:
            now_mono, now_wall = self._now_pair()
            self._reap_expired_locked(now_mono, persist=True)
            if term != self._state["term"]:
                raise CoordError(
                    "E_TERM",
                    "heartbeat term does not match the coordination term",
                    expected=self._state["term"],
                    received=term,
                )
            member = self._state["members"].get(member_id)
            if member is None:
                raise CoordError("E_MEMBER", "member does not exist", member=member_id)
            if member["term"] != term:
                raise CoordError(
                    "E_TERM",
                    "member term does not match heartbeat term",
                    expected=member["term"],
                    received=term,
                    member=member_id,
                )
            if lease_version != member["lease_version"]:
                raise CoordError(
                    "E_VERSION",
                    "lease version is stale",
                    member=member_id,
                    expected=member["lease_version"],
                    received=lease_version,
                )
            if not self._dependencies_healthy_locked(member):
                if member["candidate_eligible"] or member["lease_valid"]:
                    member["candidate_eligible"] = False
                    member["lease_valid"] = False
                    member["status"] = "invalidated"
                    member["invalidation_version"] += 1
                    self._deadlines[member_id] = now_mono
                    self._refresh_election_locked()
                    self._touch_locked()
                    self._persist_locked()
                raise dependency_error(
                    "one or more dependencies are unhealthy; lease renewal rejected",
                    member=member_id,
                    dependencies=list(member["dependencies"]),
                )

            member["lease_expiry_ms"] = now_wall + int(member["lease_ms"])
            member["lease_version"] += 1
            member["lease_valid"] = True
            member["candidate_eligible"] = True
            member["status"] = "active"
            self._deadlines[member_id] = now_mono + int(member["lease_ms"])
            self._refresh_election_locked()
            self._touch_locked()
            self._persist_locked()
            result = {
                "term": term,
                "member": member_id,
                "lease_version": member["lease_version"],
                "lease_expiry_ms": member["lease_expiry_ms"],
                "lease_ms": member["lease_ms"],
                "election": self._election_result_locked(),
                "state_version": self._state["state_version"],
            }
            self.logger.write("heartbeat_renewed", **result)
            return result

    def expired_members(self) -> list[str]:
        with self._lock:
            now_mono = self._now_pair()[0]
            expired = []
            for member_id in sorted(self._state["members"]):
                member = self._state["members"][member_id]
                if member["lease_valid"] and self._deadlines.get(member_id, now_mono) <= now_mono:
                    expired.append(member_id)
            return expired

    def reap_expired(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._reap_expired_locked(self._now_pair()[0], persist=True)

    def _reap_expired_locked(self, now_mono: int, *, persist: bool) -> list[dict[str, Any]]:
        expired = [
            member_id
            for member_id in sorted(self._state["members"])
            if self._state["members"][member_id]["lease_valid"]
            and self._deadlines.get(member_id, now_mono) <= now_mono
        ]
        if not expired:
            return []
        events: list[dict[str, Any]] = []
        for member_id in expired:
            member = self._state["members"][member_id]
            if not member["lease_valid"]:
                continue
            graph_before = self._graph_snapshot_locked()
            token_before = self._token_health_locked()
            member["lease_valid"] = False
            member["candidate_eligible"] = False
            member["status"] = "expired"
            self._deadlines[member_id] = now_mono
            event = self._invalidate_downstream_locked(
                trigger=f"member:{member_id}",
                operation="lease_expired",
                cause="lease_expired",
                graph_before=graph_before,
                token_before=token_before,
                include_trigger=True,
                now_mono=now_mono,
            )
            events.append(event)
        self._refresh_election_locked()
        self._touch_locked()
        if persist:
            self._persist_locked()
            for event in events:
                self.logger.write("lease_reclaimed", trace=event, state_version=self._state["state_version"])
        return events

    # ------------------------------------------------------------------
    # Dependency propagation
    # ------------------------------------------------------------------
    def _invalidate_downstream_locked(
        self,
        *,
        trigger: str,
        operation: str,
        cause: str,
        graph_before: dict[str, list[str]],
        token_before: dict[str, bool],
        include_trigger: bool,
        now_mono: int,
    ) -> dict[str, Any]:
        synthetic_members = {
            member_id: {"dependencies": list(dependencies)}
            for member_id, dependencies in graph_before.items()
        }
        synthetic_tokens = {token_id: {} for token_id in token_before}
        affected = stable_downstream(
            synthetic_members,
            synthetic_tokens,
            trigger,
            include_trigger=include_trigger,
        )
        steps: list[dict[str, Any]] = []
        for index, member_id in enumerate(affected, start=1):
            member = self._state["members"].get(member_id)
            if member is None:
                continue
            before_version = int(member["invalidation_version"])
            member["candidate_eligible"] = False
            member["lease_valid"] = False
            member["status"] = "expired" if cause == "lease_expired" and member_id == trigger.partition(":")[2] else "invalidated"
            member["invalidation_version"] = before_version + 1
            member["lease_version"] += 1
            self._deadlines[member_id] = now_mono
            steps.append(
                {
                    "step": index,
                    "trigger": trigger,
                    "affected": member_id,
                    "action": "candidate_and_lease_invalidated",
                    "invalidation_version_before": before_version,
                    "invalidation_version_after": member["invalidation_version"],
                    "candidate_eligible_after": False,
                    "lease_valid_after": False,
                }
            )
        event: dict[str, Any] = {
            "trace_id": int(self._state["next_trace_id"]),
            "operation": operation,
            "cause": cause,
            "trigger": trigger,
            "include_trigger": include_trigger,
            "graph_before": deepcopy(graph_before),
            "token_health_before": deepcopy(token_before),
            "affected_order": affected,
            "steps": steps,
        }
        event["checksum"] = stable_hash(event)
        self._state["next_trace_id"] += 1
        self._state["propagation_traces"].append(event)
        self._refresh_election_locked()
        return event

    def invalidate_token(self, token_id: str) -> dict[str, Any]:
        with self._lock:
            token = self._state["tokens"].get(token_id)
            if token is None:
                raise CoordError("E_MEMBER", "health token does not exist", token=token_id)
            graph_before = self._graph_snapshot_locked()
            token_before = self._token_health_locked()
            token["healthy"] = False
            token["version"] += 1
            now_mono = self._now_pair()[0]
            event = self._invalidate_downstream_locked(
                trigger=f"token:{token_id}",
                operation="token_invalidated",
                cause="health_token_invalid",
                graph_before=graph_before,
                token_before=token_before,
                include_trigger=False,
                now_mono=now_mono,
            )
            self._refresh_election_locked()
            self._touch_locked()
            self._persist_locked()
            self.logger.write(
                "token_invalidated",
                token=token_id,
                version=token["version"],
                trace=event,
                state_version=self._state["state_version"],
            )
            return {
                "token": token_id,
                "token_version": token["version"],
                "affected": event["affected_order"],
                "trace": event,
                "election": self._election_result_locked(),
                "state_version": self._state["state_version"],
            }

    # ------------------------------------------------------------------
    # Incremental membership changes and term advancement
    # ------------------------------------------------------------------
    @staticmethod
    def _list_field(changes: dict[str, Any], name: str) -> list[Any]:
        value = changes.get(name, [])
        if value is None:
            return []
        if not isinstance(value, list):
            raise protocol_error(f"{name} must be a list", field=name)
        return value

    def apply_changeset(self, changes: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(changes, dict):
            raise protocol_error("changeset must be an object")
        with self._lock:
            old_state = deepcopy(self._state)
            old_deadlines = dict(self._deadlines)
            try:
                return self._apply_changeset_locked(changes)
            except BaseException:
                self._state = old_state
                self._deadlines = old_deadlines
                raise

    def _apply_changeset_locked(self, changes: dict[str, Any]) -> dict[str, Any]:
        base_version = changes.get("base_version")
        if base_version is not None:
            if not is_int(base_version):
                raise protocol_error("base_version must be an integer")
            if base_version != self._state["state_version"]:
                raise CoordError(
                    "E_VERSION",
                    "changeset base version is stale",
                    expected=self._state["state_version"],
                    received=base_version,
                )

        additions = self._list_field(changes, "add")
        removals = self._list_field(changes, "remove")
        modifications = self._list_field(changes, "modify")
        lease_updates = self._list_field(changes, "lease")
        if not all(isinstance(item, str) and item for item in removals):
            raise protocol_error("remove entries must be non-empty member IDs")
        if len(removals) != len(set(removals)):
            raise protocol_error("remove entries contain duplicates")

        graph_before = self._graph_snapshot_locked()
        token_before = self._token_health_locked()
        now_mono, _ = self._now_pair()
        added_ids: list[str] = []
        modified_ids: list[str] = []
        lease_ids: list[str] = []

        for index, raw in enumerate(additions):
            if not isinstance(raw, dict):
                raise protocol_error("add entries must be objects", index=index)
            member_id = raw.get("id")
            if not isinstance(member_id, str) or not member_id:
                raise protocol_error("added member requires a non-empty id", index=index)
            if member_id in self._state["members"]:
                raise protocol_error("member already exists", member=member_id)
            if member_id in removals:
                raise protocol_error("cannot add and remove the same member", member=member_id)
            member = normalize_member(raw, where=f"add[{index}]")
            member["term"] = self._state["term"]
            member["lease_version"] = 0
            member["lease_expiry_ms"] = 0
            member["lease_valid"] = False
            member["candidate_eligible"] = False
            member["status"] = "active"
            member["invalidation_version"] = 0
            self._state["members"][member_id] = member
            self._deadlines[member_id] = now_mono
            added_ids.append(member_id)

        for member_id in removals:
            if member_id not in self._state["members"]:
                raise CoordError("E_MEMBER", "member to remove does not exist", member=member_id)
            del self._state["members"][member_id]
            self._deadlines.pop(member_id, None)
        removed_refs = {f"member:{member_id}" for member_id in removals}
        if removed_refs:
            for member in self._state["members"].values():
                member["dependencies"] = [
                    dependency
                    for dependency in member["dependencies"]
                    if dependency not in removed_refs
                ]

        for index, raw in enumerate(modifications):
            if not isinstance(raw, dict):
                raise protocol_error("modify entries must be objects", index=index)
            member_id = raw.get("id")
            if not isinstance(member_id, str) or member_id not in self._state["members"]:
                raise CoordError("E_MEMBER", "member to modify does not exist", member=member_id)
            member = self._state["members"][member_id]
            dependency_changed = "dependencies" in raw
            if dependency_changed:
                dependencies = raw["dependencies"]
                if not isinstance(dependencies, list) or any(
                    not isinstance(item, str) for item in dependencies
                ):
                    raise protocol_error(
                        "dependencies must be a list of strings", member=member_id
                    )
                member["dependencies"] = list(dependencies)
            if "lease_ms" in raw:
                lease_ms = raw["lease_ms"]
                if not is_int(lease_ms) or lease_ms <= 0:
                    raise protocol_error(
                        "lease_ms must be a positive integer", member=member_id
                    )
                member["lease_ms"] = lease_ms
            if dependency_changed:
                modified_ids.append(member_id)

        for index, raw in enumerate(lease_updates):
            if not isinstance(raw, dict):
                raise protocol_error("lease entries must be objects", index=index)
            member_id = raw.get("id")
            if not isinstance(member_id, str) or member_id not in self._state["members"]:
                raise CoordError("E_MEMBER", "member for lease update does not exist", member=member_id)
            lease_ms = raw.get("lease_ms")
            if not is_int(lease_ms) or lease_ms <= 0:
                raise protocol_error("lease_ms must be a positive integer", member=member_id)
            self._state["members"][member_id]["lease_ms"] = lease_ms
            lease_ids.append(member_id)

        for member in self._state["members"].values():
            if member["dependencies"]:
                normalized_dependencies = []
                for dependency in member["dependencies"]:
                    try:
                        normalized_dependencies.append(
                            normalize_ref(
                                dependency,
                                set(self._state["members"]),
                                set(self._state["tokens"]),
                            )
                        )
                    except ValueError as exc:
                        raise dependency_error(
                            f"invalid dependency {dependency!r}",
                            member=member["id"],
                            dependency=dependency,
                            chain=[f"member:{member['id']}", dependency],
                        ) from exc
                member["dependencies"] = normalized_dependencies
        validate_graph(self._state["members"], self._state["tokens"])

        events: list[dict[str, Any]] = []
        affected_union: set[str] = set()
        for removed_id in sorted(removals):
            event = self._invalidate_downstream_locked(
                trigger=f"member:{removed_id}",
                operation="remove_member",
                cause="member_deleted",
                graph_before=graph_before,
                token_before=token_before,
                include_trigger=False,
                now_mono=now_mono,
            )
            affected_union.update(event["affected_order"])
            events.append(event)
        for modified_id in sorted(modified_ids):
            event = self._invalidate_downstream_locked(
                trigger=f"member:{modified_id}",
                operation="modify_dependencies",
                cause="dependency_changed",
                graph_before=graph_before,
                token_before=token_before,
                include_trigger=True,
                now_mono=now_mono,
            )
            affected_union.update(event["affected_order"])
            events.append(event)

        if not removals and not modifications and not additions and not lease_updates:
            raise protocol_error("changeset is empty")

        self._refresh_election_locked()
        self._touch_locked()
        self._persist_locked()
        removed_or_affected = set(removals) | affected_union | set(added_ids)
        unaffected = sorted(
            member_id
            for member_id in self._state["members"]
            if member_id not in removed_or_affected
        )
        affected_order: list[str] = []
        seen_affected: set[str] = set()
        for event in events:
            for member_id in event["affected_order"]:
                if member_id not in seen_affected:
                    seen_affected.add(member_id)
                    affected_order.append(member_id)
        result = {
            "state_version": self._state["state_version"],
            "term": self._state["term"],
            "added": sorted(added_ids),
            "removed": sorted(removals),
            "modified": sorted(modified_ids),
            "lease_updated": sorted(lease_ids),
            "affected_order": affected_order,
            "unaffected": [
                {
                    "id": member_id,
                    "term": self._state["members"][member_id]["term"],
                    "candidate_eligible": self._state["members"][member_id]["candidate_eligible"],
                    "invalidation_version": self._state["members"][member_id][
                        "invalidation_version"
                    ],
                }
                for member_id in unaffected
            ],
            "traces": events,
            "election": self._election_result_locked(),
        }
        self.logger.write("changeset_applied", **result)
        return result

    def advance_term(self, trigger_member: str | None = None) -> dict[str, Any]:
        with self._lock:
            old_state = deepcopy(self._state)
            old_deadlines = dict(self._deadlines)
            try:
                graph_before = self._graph_snapshot_locked()
                token_before = self._token_health_locked()
                self._state["term"] += 1
                for member in self._state["members"].values():
                    member["term"] = self._state["term"]
                self._state["election_requires_rerun"] = True
                event = None
                if trigger_member is not None:
                    if trigger_member not in self._state["members"]:
                        raise CoordError(
                            "E_MEMBER", "term trigger member does not exist", member=trigger_member
                        )
                    event = self._invalidate_downstream_locked(
                        trigger=f"member:{trigger_member}",
                        operation="advance_term",
                        cause="term_advanced",
                        graph_before=graph_before,
                        token_before=token_before,
                        include_trigger=True,
                        now_mono=self._now_pair()[0],
                    )
                self._refresh_election_locked(force_unknown=True)
                self._touch_locked()
                self._persist_locked()
                result = {
                    "term": self._state["term"],
                    "affected": event["affected_order"] if event else [],
                    "trace": event,
                    "election": self._election_result_locked(),
                    "state_version": self._state["state_version"],
                }
                self.logger.write("term_advanced", **result)
                return result
            except BaseException:
                self._state = old_state
                self._deadlines = old_deadlines
                raise

    # ------------------------------------------------------------------
    # Queries and snapshots
    # ------------------------------------------------------------------
    def query_leader(self) -> dict[str, Any]:
        with self._lock:
            self._reap_expired_locked(self._now_pair()[0], persist=True)
            return {
                "term": self._state["term"],
                "leader": self._state["election"].get("leader"),
                "election": self._election_result_locked(),
                "requires_rerun": bool(
                    self._state.get("election_requires_rerun", False)
                    or self._state["election"].get("requires_rerun", False)
                ),
                "state_version": self._state["state_version"],
            }

    def query_state(self) -> dict[str, Any]:
        with self._lock:
            now_mono, _ = self._now_pair()
            self._reap_expired_locked(now_mono, persist=True)
            members = {}
            for member_id in sorted(self._state["members"]):
                member = deepcopy(self._state["members"][member_id])
                remaining = 0
                if member["lease_valid"]:
                    remaining = max(0, self._deadlines.get(member_id, now_mono) - now_mono)
                member["lease_remaining_ms"] = remaining
                members[member_id] = member
            return {
                "state_version": self._state["state_version"],
                "term": self._state["term"],
                "members": members,
                "tokens": deepcopy(self._state["tokens"]),
                "election": self._election_result_locked(),
                "state_path": str(self.state_path),
            }

    def create_snapshot(self, path: str | Path | None = None) -> dict[str, Any]:
        with self._lock:
            self._reap_expired_locked(self._now_pair()[0], persist=True)
            payload = deepcopy(self._state)
            document = {
                "format": "coord-snapshot-v1",
                "snapshot_version": int(self._state["state_version"]),
                "payload": payload,
                "payload_checksum": stable_hash(payload),
            }
            target = Path(path) if path is not None else self.root / "snapshots" / "snapshot.json"
            if not target.is_absolute():
                target = self.root / target
            atomic_write_json(target, document)
            result = {
                "path": str(target),
                "snapshot_version": document["snapshot_version"],
                "payload_checksum": document["payload_checksum"],
                "current_state_version": int(self._state["state_version"]),
                "replaced": False,
            }
            self.logger.write("snapshot_created", **result)
            return result

    @staticmethod
    def validate_replay(
        payload: dict[str, Any], events: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        traces = payload.get("propagation_traces", []) if events is None else events
        if not isinstance(traces, list):
            raise replay_error("propagation traces must be a list", step=1)
        for event_index, event in enumerate(traces):
            trace_id = event.get("trace_id", event_index + 1) if isinstance(event, dict) else event_index + 1
            try:
                if not isinstance(event, dict):
                    raise ValueError("trace event is not an object")
                checksum = event.get("checksum")
                unsigned = {key: value for key, value in event.items() if key != "checksum"}
                if checksum != stable_hash(unsigned):
                    raise replay_error(
                        "trace checksum does not reproduce",
                        trace_id=trace_id,
                        step=1,
                    )
                graph_before = event.get("graph_before")
                token_before = event.get("token_health_before")
                trigger = event.get("trigger")
                affected_order = event.get("affected_order")
                steps = event.get("steps")
                if not isinstance(graph_before, dict) or not isinstance(token_before, dict):
                    raise ValueError("trace lacks graph_before/token_health_before")
                if not isinstance(trigger, str) or not isinstance(affected_order, list):
                    raise ValueError("trace lacks trigger/affected_order")
                if not isinstance(steps, list):
                    raise ValueError("trace lacks steps")
                synthetic_members = {
                    member_id: {"dependencies": list(dependencies)}
                    for member_id, dependencies in graph_before.items()
                }
                synthetic_tokens = {token_id: {} for token_id in token_before}
                expected = stable_downstream(
                    synthetic_members,
                    synthetic_tokens,
                    trigger,
                    include_trigger=bool(event.get("include_trigger", False)),
                )
                if expected != affected_order:
                    mismatch = next(
                        (
                            index
                            for index, (left, right) in enumerate(
                                zip(expected, affected_order), start=1
                            )
                            if left != right
                        ),
                        min(len(expected), len(affected_order)) + 1,
                    )
                    raise replay_error(
                        "trace propagation order is not reproducible",
                        trace_id=trace_id,
                        step=mismatch,
                        expected=expected,
                        recorded=affected_order,
                    )
                if len(steps) != len(affected_order):
                    raise replay_error(
                        "trace step count differs from affected order",
                        trace_id=trace_id,
                        step=min(len(steps), len(affected_order)) + 1,
                    )
                for step_index, (step, expected_member) in enumerate(
                    zip(steps, affected_order), start=1
                ):
                    if not isinstance(step, dict):
                        raise replay_error(
                            "trace step is not an object", trace_id=trace_id, step=step_index
                        )
                    if step.get("step") != step_index or step.get("affected") != expected_member:
                        raise replay_error(
                            "trace step does not reproduce",
                            trace_id=trace_id,
                            step=step_index,
                            expected_affected=expected_member,
                            recorded_affected=step.get("affected"),
                        )
                    before = step.get("invalidation_version_before")
                    after = step.get("invalidation_version_after")
                    if not is_int(before) or not is_int(after) or after != before + 1:
                        raise replay_error(
                            "trace invalidation version transition is invalid",
                            trace_id=trace_id,
                            step=step_index,
                        )
                for index, step in enumerate(steps, start=1):
                    if step.get("trigger") != trigger:
                        raise replay_error(
                            "trace step trigger does not match event",
                            trace_id=trace_id,
                            step=index,
                        )
            except CoordError:
                raise
            except Exception as exc:
                raise replay_error(
                    f"trace cannot be replayed: {exc}",
                    trace_id=trace_id,
                    step=max(1, int(event.get("step", 1)) if isinstance(event, dict) else 1),
                ) from exc
        return {"ok": True, "events": len(traces)}

    def restore_snapshot(self, path: str | Path) -> dict[str, Any]:
        target = Path(path)
        if not target.is_absolute():
            target = self.root / target
        try:
            document = read_json(target)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.logger.write(
                "snapshot_rejected", code="E_SNAPSHOT", path=str(target), message=str(exc)
            )
            raise snapshot_error("snapshot cannot be read", path=str(target), cause=str(exc)) from exc
        if not isinstance(document, dict) or document.get("format") != "coord-snapshot-v1":
            self.logger.write(
                "snapshot_rejected", code="E_SNAPSHOT", path=str(target), message="bad format"
            )
            raise snapshot_error("snapshot format is invalid", path=str(target))
        payload = document.get("payload")
        if not isinstance(payload, dict):
            raise snapshot_error("snapshot payload is missing", path=str(target))
        checksum = document.get("payload_checksum")
        if checksum != stable_hash(payload):
            self.logger.write(
                "snapshot_rejected", code="E_SNAPSHOT", path=str(target), message="checksum mismatch"
            )
            raise snapshot_error("snapshot checksum does not match payload", path=str(target))
        try:
            restored = normalize_state(payload)
        except CoordError as exc:
            self.logger.write(
                "snapshot_rejected",
                code="E_SNAPSHOT",
                path=str(target),
                message=exc.message,
                details=exc.details,
            )
            raise snapshot_error(
                "snapshot state is invalid", path=str(target), cause=exc.as_dict()
            ) from exc
        try:
            replay = self.validate_replay(restored)
        except CoordError as exc:
            self.logger.write(
                "snapshot_replay_rejected",
                code=exc.code,
                path=str(target),
                message=exc.message,
                details=exc.details,
            )
            raise

        with self._lock:
            previous_version = int(self._state["state_version"])
            snapshot_version = int(document.get("snapshot_version", restored["state_version"]))
            version_mismatch = previous_version != snapshot_version
            restored["state_version"] = max(previous_version, snapshot_version) + 1
            self._state = restored
            self._rebuild_deadlines_locked()
            self._reap_expired_locked(self._now_pair()[0], persist=False)
            self._persist_locked()
            result = {
                "path": str(target),
                "previous_state_version": previous_version,
                "snapshot_version": snapshot_version,
                "version_mismatch": version_mismatch,
                "replaced": True,
                "replacement_mode": "whole_state",
                "restored_state_version": self._state["state_version"],
                "replay": replay,
                "election": self._election_result_locked(),
            }
            self.logger.write("snapshot_restored", **result)
            return result
