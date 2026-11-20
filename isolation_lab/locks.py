from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .models import canonical_json


LockEventCallback = Callable[[str, str | None, int | None, dict[str, Any], int], None]


@dataclass
class LockOwner:
    session_id: str
    mode: str
    grant_seq: int
    granted_virtual_ms: int
    isolation_level: str


@dataclass
class WaitRequest:
    request_seq: int
    session_id: str
    row_key: str
    mode: str
    isolation_level: str
    requested_virtual_ms: int
    op_no: int
    op_started_virtual_ms: int
    wait_limit_ms: int
    deadline_virtual_ms: int
    blocked_by: tuple[str, ...] = ()
    queue_position: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_seq": self.request_seq,
            "session_id": self.session_id,
            "row_key": self.row_key,
            "mode": self.mode,
            "isolation_level": self.isolation_level,
            "requested_virtual_ms": self.requested_virtual_ms,
            "op_no": self.op_no,
            "op_started_virtual_ms": self.op_started_virtual_ms,
            "wait_limit_ms": self.wait_limit_ms,
            "deadline_virtual_ms": self.deadline_virtual_ms,
            "blocked_by": list(self.blocked_by),
            "queue_position": self.queue_position,
        }


@dataclass
class AcquireResult:
    status: str
    grant_seq: int | None
    request: WaitRequest | None
    reason: str
    blocked_by: tuple[str, ...] = ()
    queue_position: int = 0


@dataclass
class GrantedWait:
    request: WaitRequest
    grant_seq: int
    granted_virtual_ms: int
    reason: str


@dataclass
class RowLockState:
    owners: dict[str, LockOwner] = field(default_factory=dict)
    queue: list[WaitRequest] = field(default_factory=list)

    def owner_snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": owner.session_id,
                "mode": owner.mode,
                "grant_seq": owner.grant_seq,
                "granted_virtual_ms": owner.granted_virtual_ms,
                "isolation_level": owner.isolation_level,
            }
            for owner in sorted(self.owners.values(), key=lambda item: item.session_id)
        ]

    def queue_snapshot(self) -> list[dict[str, Any]]:
        return [request.to_dict() for request in sorted(self.queue, key=lambda item: item.request_seq)]


class LockManager:
    """Deterministic row lock table with shared/exclusive grants and FIFO queues."""

    def __init__(self, event_callback: LockEventCallback | None = None) -> None:
        self._rows: dict[str, RowLockState] = {}
        self._next_request_seq = 1
        self._next_grant_seq = 1
        self._notifications: list[GrantedWait] = []
        self.event_callback = event_callback

    def _row(self, row_key: str) -> RowLockState:
        state = self._rows.get(row_key)
        if state is None:
            state = RowLockState()
            self._rows[row_key] = state
        return state

    def _emit(
        self,
        event_type: str,
        session_id: str | None,
        op_no: int | None,
        payload: dict[str, Any],
        now: int,
    ) -> None:
        if self.event_callback is not None:
            self.event_callback(event_type, session_id, op_no, payload, now)

    def _next_grant(self) -> int:
        grant_seq = self._next_grant_seq
        self._next_grant_seq += 1
        return grant_seq

    @staticmethod
    def _other_owners(state: RowLockState, session_id: str) -> list[LockOwner]:
        return [owner for sid, owner in state.owners.items() if sid != session_id]

    def _compatible(self, state: RowLockState, session_id: str, mode: str) -> bool:
        others = self._other_owners(state, session_id)
        if mode == "S":
            return all(owner.mode == "S" for owner in others)
        return not others

    def _blocking_owners(self, state: RowLockState, session_id: str, mode: str) -> tuple[str, ...]:
        others = self._other_owners(state, session_id)
        if mode == "S":
            return tuple(sorted(owner.session_id for owner in others if owner.mode == "X"))
        return tuple(sorted(owner.session_id for owner in others))

    def _grant(
        self,
        state: RowLockState,
        request_session: str,
        mode: str,
        isolation: str,
        now: int,
        reason: str,
    ) -> int:
        grant_seq = self._next_grant()
        state.owners[request_session] = LockOwner(
            session_id=request_session,
            mode=mode,
            grant_seq=grant_seq,
            granted_virtual_ms=now,
            isolation_level=isolation,
        )
        return grant_seq

    def acquire(
        self,
        *,
        session_id: str,
        row_key: str,
        mode: str,
        isolation_level: str,
        now: int,
        op_no: int,
        op_started_virtual_ms: int,
        remaining_budget_ms: int,
        operation_timeout_ms: int,
    ) -> AcquireResult:
        if mode not in ("S", "X"):
            raise ValueError(f"invalid lock mode: {mode}")
        state = self._row(row_key)
        owner = state.owners.get(session_id)

        if owner is not None:
            if owner.mode == "X" or mode == "S":
                self._emit(
                    "lock.granted",
                    session_id,
                    op_no,
                    {
                        "row_key": row_key,
                        "mode": owner.mode,
                        "requested_mode": mode,
                        "grant_seq": owner.grant_seq,
                        "reason": "already_held",
                        "isolation_level": isolation_level,
                        "owners": state.owner_snapshot(),
                        "queue": state.queue_snapshot(),
                    },
                    now,
                )
                return AcquireResult("granted", owner.grant_seq, None, "already_held")

        # A shared holder asking for exclusive mode is an upgrade. It can succeed
        # immediately only when no other session holds the row and no waiter is queued.
        if owner is not None and owner.mode == "S" and mode == "X":
            if not state.queue and self._compatible(state, session_id, "X"):
                grant_seq = self._grant(state, session_id, "X", isolation_level, now, "upgrade")
                self._emit(
                    "lock.granted",
                    session_id,
                    op_no,
                    {
                        "row_key": row_key,
                        "mode": "X",
                        "requested_mode": "X",
                        "grant_seq": grant_seq,
                        "reason": "upgrade",
                        "isolation_level": isolation_level,
                        "owners": state.owner_snapshot(),
                        "queue": state.queue_snapshot(),
                    },
                    now,
                )
                return AcquireResult("granted", grant_seq, None, "upgrade")

        existing = next(
            (
                request
                for request in state.queue
                if request.session_id == session_id and request.row_key == row_key
            ),
            None,
        )
        if existing is not None:
            return AcquireResult(
                "waiting",
                None,
                existing,
                "already_queued",
                blocked_by=existing.blocked_by,
                queue_position=existing.queue_position,
            )

        if not state.queue and self._compatible(state, session_id, mode):
            grant_seq = self._grant(state, session_id, mode, isolation_level, now, "compatible")
            self._emit(
                "lock.granted",
                session_id,
                op_no,
                {
                    "row_key": row_key,
                    "mode": mode,
                    "requested_mode": mode,
                    "grant_seq": grant_seq,
                    "reason": "compatible",
                    "isolation_level": isolation_level,
                    "owners": state.owner_snapshot(),
                    "queue": state.queue_snapshot(),
                },
                now,
            )
            return AcquireResult("granted", grant_seq, None, "compatible")

        wait_limit_ms = min(max(0, remaining_budget_ms), max(0, operation_timeout_ms))
        request_seq = self._next_request_seq
        self._next_request_seq += 1
        blocked_by = self._blocking_owners(state, session_id, mode)
        request = WaitRequest(
            request_seq=request_seq,
            session_id=session_id,
            row_key=row_key,
            mode=mode,
            isolation_level=isolation_level,
            requested_virtual_ms=now,
            op_no=op_no,
            op_started_virtual_ms=op_started_virtual_ms,
            wait_limit_ms=wait_limit_ms,
            deadline_virtual_ms=now + wait_limit_ms,
            blocked_by=blocked_by,
        )
        state.queue.append(request)
        state.queue.sort(key=lambda item: item.request_seq)
        for index, queued in enumerate(state.queue):
            queued.queue_position = index
        request.queue_position = state.queue.index(request)
        had_earlier_waiters = len(state.queue) > 1
        self._emit(
            "lock.waiting",
            session_id,
            op_no,
            {
                "row_key": row_key,
                "mode": mode,
                "request_seq": request_seq,
                "queue_position": request.queue_position,
                "blocked_by": list(blocked_by),
                "wait_limit_ms": wait_limit_ms,
                "deadline_virtual_ms": request.deadline_virtual_ms,
                "isolation_level": isolation_level,
                "compatibility_basis": "fifo_queue" if had_earlier_waiters else "mode_conflict",
                "owners": state.owner_snapshot(),
                "queue": state.queue_snapshot(),
            },
            now,
        )
        return AcquireResult(
            "waiting",
            None,
            request,
            "fifo_queue" if had_earlier_waiters else "mode_conflict",
            blocked_by=blocked_by,
            queue_position=request.queue_position,
        )

    def drain_notifications(self) -> list[GrantedWait]:
        notifications = self._notifications
        self._notifications = []
        return notifications

    def pump(self, now: int) -> list[GrantedWait]:
        """Grant at most the FIFO-compatible prefix for every row, deterministically."""
        changed = True
        while changed:
            changed = False
            for row_key in sorted(self._rows):
                state = self._rows[row_key]
                while state.queue:
                    head = state.queue[0]
                    if not self._compatible(state, head.session_id, head.mode):
                        break
                    state.queue.pop(0)
                    grant_seq = self._grant(
                        state,
                        head.session_id,
                        head.mode,
                        head.isolation_level,
                        now,
                        "queue_head",
                    )
                    for index, queued in enumerate(state.queue):
                        queued.queue_position = index
                    notification = GrantedWait(head, grant_seq, now, "queue_head")
                    self._notifications.append(notification)
                    self._emit(
                        "lock.granted",
                        head.session_id,
                        head.op_no,
                        {
                            "row_key": row_key,
                            "mode": head.mode,
                            "requested_mode": head.mode,
                            "grant_seq": grant_seq,
                            "reason": "queue_head",
                            "request_seq": head.request_seq,
                            "isolation_level": head.isolation_level,
                            "owners": state.owner_snapshot(),
                            "queue": state.queue_snapshot(),
                        },
                        now,
                    )
                    changed = True
                # Empty row states remain internally for stable row identity; they
                # are omitted from exported lock snapshots and entry counts.
        return self._notifications

    def _release_owner(
        self,
        state: RowLockState,
        row_key: str,
        session_id: str,
        now: int,
        reason: str,
        op_no: int | None,
    ) -> bool:
        owner = state.owners.get(session_id)
        if owner is None:
            return False
        del state.owners[session_id]
        self._emit(
            "lock.released",
            session_id,
            op_no,
            {
                "row_key": row_key,
                "mode": owner.mode,
                "grant_seq": owner.grant_seq,
                "reason": reason,
                "owners": state.owner_snapshot(),
                "queue": state.queue_snapshot(),
            },
            now,
        )
        return True

    def release_shared(self, session_id: str, row_key: str, now: int, reason: str = "read_committed_transient") -> bool:
        state = self._rows.get(row_key)
        if state is None:
            return False
        owner = state.owners.get(session_id)
        if owner is None or owner.mode != "S":
            return False
        self._release_owner(state, row_key, session_id, now, reason, None)
        self.pump(now)
        return True

    def release_session(self, session_id: str, now: int, reason: str) -> int:
        released = 0
        for row_key in sorted(self._rows):
            state = self._rows[row_key]
            if session_id in state.owners:
                owner = state.owners[session_id]
                del state.owners[session_id]
                released += 1
                self._emit(
                    "lock.released",
                    session_id,
                    None,
                    {
                        "row_key": row_key,
                        "mode": owner.mode,
                        "grant_seq": owner.grant_seq,
                        "reason": reason,
                        "owners": state.owner_snapshot(),
                        "queue": state.queue_snapshot(),
                    },
                    now,
                )
            removed_queue = [request for request in state.queue if request.session_id == session_id]
            if removed_queue:
                state.queue = [request for request in state.queue if request.session_id != session_id]
                released += len(removed_queue)
                for request in removed_queue:
                    self._emit(
                        "lock.wait_cancelled",
                        session_id,
                        request.op_no,
                        {
                            "row_key": row_key,
                            "mode": request.mode,
                            "request_seq": request.request_seq,
                            "reason": reason,
                            "owners": state.owner_snapshot(),
                            "queue": state.queue_snapshot(),
                        },
                        now,
                    )
        self.pump(now)
        return released

    def cancel_wait(self, session_id: str, row_key: str, now: int, reason: str) -> WaitRequest | None:
        state = self._rows.get(row_key)
        if state is None:
            return None
        match = next((request for request in state.queue if request.session_id == session_id), None)
        if match is None:
            return None
        state.queue = [request for request in state.queue if request is not match]
        for index, queued in enumerate(state.queue):
            queued.queue_position = index
        self._emit(
            "lock.wait_cancelled",
            session_id,
            match.op_no,
            {
                "row_key": row_key,
                "mode": match.mode,
                "request_seq": match.request_seq,
                "reason": reason,
                "owners": state.owner_snapshot(),
                "queue": state.queue_snapshot(),
            },
            now,
        )
        return match

    def waiting_dependencies(self) -> dict[str, set[str]]:
        dependencies: dict[str, set[str]] = {}
        for state in self._rows.values():
            for request in state.queue:
                blockers = set(self._blocking_owners(state, request.session_id, request.mode))
                dependencies.setdefault(request.session_id, set()).update(blockers)
        return dependencies

    def waiting_request(self, session_id: str) -> WaitRequest | None:
        for state in self._rows.values():
            for request in state.queue:
                if request.session_id == session_id:
                    return request
        return None

    def held_by(self, session_id: str) -> list[dict[str, Any]]:
        held: list[dict[str, Any]] = []
        for row_key in sorted(self._rows):
            owner = self._rows[row_key].owners.get(session_id)
            if owner is not None:
                held.append(
                    {
                        "row_key": row_key,
                        "mode": owner.mode,
                        "grant_seq": owner.grant_seq,
                        "granted_virtual_ms": owner.granted_virtual_ms,
                    }
                )
        return held

    def all_waiters(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row_key in sorted(self._rows):
            for request in sorted(self._rows[row_key].queue, key=lambda item: item.request_seq):
                item = request.to_dict()
                item["row_key"] = row_key
                result.append(item)
        return result

    def state_dict(self) -> dict[str, Any]:
        rows = []
        for row_key in sorted(self._rows):
            state = self._rows[row_key]
            if not state.owners and not state.queue:
                continue
            rows.append(
                {
                    "row_key": row_key,
                    "owners": state.owner_snapshot(),
                    "queue": state.queue_snapshot(),
                }
            )
        return {
            "rows": rows,
            "next_request_seq": self._next_request_seq,
            "next_grant_seq": self._next_grant_seq,
        }

    def owner_count(self) -> int:
        return sum(len(state.owners) for state in self._rows.values())

    def waiter_count(self) -> int:
        return sum(len(state.queue) for state in self._rows.values())

    def entry_count(self) -> int:
        return self.owner_count() + self.waiter_count()

    def clear(self) -> int:
        cleared = self.entry_count()
        self._rows.clear()
        self._notifications.clear()
        self._next_request_seq = 1
        self._next_grant_seq = 1
        return cleared
