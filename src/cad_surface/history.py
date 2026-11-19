"""Undo/redo stack with explicit parent versions and branch truncation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import content_hash
from .errors import DomainError, ErrorCode
from .state import GeometryState


@dataclass(frozen=True)
class EditOperation:
    kind: str
    op_version: int
    parent_version: int
    patch_key: str
    index_u: int
    index_v: int
    before: tuple[float, float, float]
    after: tuple[float, float, float]
    result_state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "op_version": self.op_version,
            "parent_version": self.parent_version,
            "patch_key": self.patch_key,
            "control_point_index": [self.index_u, self.index_v],
            "before": list(self.before),
            "after": list(self.after),
            "result_state_hash": self.result_state_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EditOperation":
        try:
            index = value["control_point_index"]
            before = value["before"]
            after = value["after"]
            if len(before) != 3 or len(after) != 3 or len(index) != 2:
                raise ValueError("invalid vector length")
            return cls(
                kind=str(value["kind"]),
                op_version=int(value["op_version"]),
                parent_version=int(value["parent_version"]),
                patch_key=str(value["patch_key"]),
                index_u=int(index[0]),
                index_v=int(index[1]),
                before=(float(before[0]), float(before[1]), float(before[2])),
                after=(float(after[0]), float(after[1]), float(after[2])),
                result_state_hash=str(value["result_state_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError(ErrorCode.SCHEMA_VERSION, "invalid edit operation payload", error=str(exc)) from exc

    def content_hash(self) -> str:
        return content_hash(self.to_dict())


StateLoader = Callable[[int], GeometryState]


class HistoryManager:
    def __init__(
        self,
        current: GeometryState,
        *,
        max_depth: int = 32,
        state_loader: StateLoader | None = None,
        states: Mapping[int, GeometryState] | None = None,
        operations: Iterable[EditOperation] | None = None,
        undo_versions: Sequence[int] = (),
        redo_versions: Sequence[int] = (),
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least one")
        self.max_depth = int(max_depth)
        self.current = current
        self._state_loader = state_loader
        self.states: dict[int, GeometryState] = dict(states or {})
        self.states[current.version] = current
        self.operations: dict[int, EditOperation] = {operation.op_version: operation for operation in operations or ()}
        self.undo_stack: list[int] = list(undo_versions)
        self.redo_stack: list[int] = list(redo_versions)
        self.truncated_versions: list[int] = []

    def _load_state(self, version: int) -> GeometryState:
        if version not in self.states:
            if self._state_loader is None:
                raise DomainError(
                    ErrorCode.GEOM_STALE,
                    "requested state is not available in memory",
                    version=version,
                )
            self.states[version] = self._state_loader(version)
        return self.states[version]

    def register_state(self, state: GeometryState) -> None:
        self.states[state.version] = state

    def register_operation(self, operation: EditOperation) -> None:
        self.operations[operation.op_version] = operation

    def record_commit(self, state: GeometryState, operation: EditOperation) -> None:
        if operation.parent_version != self.current.version:
            raise DomainError(
                ErrorCode.GEOM_STALE,
                "new edit parent does not match current state",
                current=self.current.version,
                parent=operation.parent_version,
            )
        if state.version != operation.op_version or state.parent_version != operation.parent_version:
            raise DomainError(ErrorCode.GEOM_STALE, "new state and operation version links disagree")
        previous_version = self.current.version
        self.states[state.version] = state
        self.operations[operation.op_version] = operation
        self.undo_stack.append(previous_version)
        self._trim_undo()
        self.redo_stack.clear()
        self.current = state

    def _trim_undo(self) -> None:
        while len(self.undo_stack) > self.max_depth:
            self.truncated_versions.append(self.undo_stack.pop(0))

    def undo(self, *, persist_callback: Callable[[int, list[int], list[int]], None] | None = None) -> GeometryState:
        if not self.undo_stack:
            raise DomainError(
                ErrorCode.UNDO_EMPTY,
                "undo stack is empty",
                current_version=self.current.version,
            )
        target_version = self.undo_stack[-1]
        target = self._load_state(target_version)
        old_version = self.current.version
        new_undo = self.undo_stack[:-1]
        new_redo = [old_version, *self.redo_stack]
        if persist_callback is not None:
            persist_callback(target_version, new_undo, new_redo)
        self.undo_stack = new_undo
        self.redo_stack = new_redo
        self.current = target
        return target

    def redo(self, *, persist_callback: Callable[[int, list[int], list[int]], None] | None = None) -> GeometryState:
        if not self.redo_stack:
            raise DomainError(ErrorCode.UNDO_EMPTY, "redo stack is empty", current_version=self.current.version)
        target_version = self.redo_stack[0]
        target = self._load_state(target_version)
        old_version = self.current.version
        new_redo = self.redo_stack[1:]
        new_undo = [*self.undo_stack, old_version]
        while len(new_undo) > self.max_depth:
            self.truncated_versions.append(new_undo.pop(0))
        if persist_callback is not None:
            persist_callback(target_version, new_undo, new_redo)
        self.undo_stack = new_undo
        self.redo_stack = new_redo
        self.current = target
        return target

    def switch_to(self, version: int, *, persist_callback: Callable[[int, list[int], list[int]], None] | None = None) -> GeometryState:
        """Switch explicitly to a known historical branch and discard redo."""

        target = self._load_state(int(version))
        lineage: list[int] = []
        cursor = target
        seen: set[int] = set()
        while cursor.version != 0:
            if cursor.version in seen:
                raise DomainError(ErrorCode.GEOM_STALE, "cycle in state parent chain", version=cursor.version)
            seen.add(cursor.version)
            parent = cursor.parent_version
            if parent is None:
                raise DomainError(ErrorCode.GEOM_STALE, "state chain ended without version zero", version=cursor.version)
            lineage.append(parent)
            if parent == 0 and 0 not in self.states and self._state_loader is not None:
                self.states[0] = self._state_loader(0)
            cursor = self._load_state(parent)
            if len(lineage) > self.max_depth + 1:
                break
        lineage = list(reversed(lineage))
        while len(lineage) > self.max_depth:
            self.truncated_versions.append(lineage.pop(0))
        new_redo: list[int] = []
        if persist_callback is not None:
            persist_callback(target.version, lineage, new_redo)
        self.undo_stack = lineage
        self.redo_stack = new_redo
        self.current = target
        return target

    def next_version(self, known_versions: Iterable[int] = ()) -> int:
        versions = set(self.states)
        versions.update(operation.op_version for operation in self.operations.values())
        versions.update(int(item) for item in known_versions)
        versions.add(self.current.version)
        return max(versions) + 1

    def lineage_to(self, version: int) -> list[int]:
        lineage: list[int] = [int(version)]
        cursor = int(version)
        seen = {cursor}
        while cursor != 0:
            operation = self.operations.get(cursor)
            if operation is None:
                raise DomainError(ErrorCode.GEOM_STALE, "operation log is missing a parent link", version=cursor)
            cursor = operation.parent_version
            if cursor in seen:
                raise DomainError(ErrorCode.GEOM_STALE, "cycle in operation parent links", version=cursor)
            seen.add(cursor)
            lineage.append(cursor)
        return list(reversed(lineage))
