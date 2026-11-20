from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .canonical import canonical_bytes, sha256_bytes, write_json, write_jsonl
from .errors import ErrorCode, fail
from .quadtree import QuadtreeNode, QuadtreeManifest


@dataclass(frozen=True)
class StreamingBudgets:
    memory_bytes: int
    vram_bytes: int
    prefetch_radius: int = 1
    max_resident_blocks: int = 0

    def __post_init__(self) -> None:
        if int(self.memory_bytes) <= 0 or int(self.vram_bytes) <= 0:
            fail(
                ErrorCode.E_BUDGET_DENIED,
                "memory and VRAM budgets must both be positive",
                memory_bytes=int(self.memory_bytes),
                vram_bytes=int(self.vram_bytes),
            )
        if int(self.prefetch_radius) < 0:
            raise ValueError("prefetch_radius must be non-negative")
        if int(self.max_resident_blocks) < 0:
            raise ValueError("max_resident_blocks must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_bytes": int(self.memory_bytes),
            "vram_bytes": int(self.vram_bytes),
            "prefetch_radius": int(self.prefetch_radius),
            "max_resident_blocks": int(self.max_resident_blocks),
        }


@dataclass
class ResidentEntry:
    block_id: str
    memory_bytes: int
    vram_bytes: int
    content_sha256: str
    loaded_frame: int
    last_used_frame: int
    load_count: int
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_report(self, current_frame: int) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "memory_bytes": self.memory_bytes,
            "vram_bytes": self.vram_bytes,
            "content_sha256": self.content_sha256,
            "loaded_frame": self.loaded_frame,
            "last_used_frame": self.last_used_frame,
            "age_frames": current_frame - self.last_used_frame,
            "load_count": self.load_count,
        }


@dataclass(frozen=True)
class StreamingStepResult:
    frame: int
    visible_block_ids: tuple[str, ...]
    load_requests: tuple[str, ...]
    completed_loads: tuple[str, ...]
    retained_blocks: tuple[str, ...]
    unloads: tuple[str, ...]
    prefetch_requested: tuple[str, ...]
    prefetch_skipped: tuple[str, ...]
    memory_used_bytes: int
    vram_used_bytes: int
    memory_budget_bytes: int
    vram_budget_bytes: int
    memory_utilization: float
    vram_utilization: float
    resident_blocks: int
    pressure_reduced_residency: bool
    visible_missing: tuple[str, ...]
    draw_digest: str

    def to_report(self) -> dict[str, Any]:
        return {
            "load_requests": list(self.load_requests),
            "completed_loads": list(self.completed_loads),
            "retained_blocks": list(self.retained_blocks),
            "unloads": list(self.unloads),
            "prefetch_requested": list(self.prefetch_requested),
            "prefetch_skipped": list(self.prefetch_skipped),
            "load_request_count": len(self.load_requests),
            "completed_load_count": len(self.completed_loads),
            "retain_count": len(self.retained_blocks),
            "unload_count": len(self.unloads),
            "prefetch_request_count": len(self.prefetch_requested),
            "prefetch_skip_count": len(self.prefetch_skipped),
            "memory_used_bytes": self.memory_used_bytes,
            "vram_used_bytes": self.vram_used_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "vram_budget_bytes": self.vram_budget_bytes,
            "memory_utilization": self.memory_utilization,
            "vram_utilization": self.vram_utilization,
            "resident_blocks": self.resident_blocks,
            "pressure_reduced_residency": self.pressure_reduced_residency,
            "visible_missing": list(self.visible_missing),
            "draw_digest": self.draw_digest,
        }


class StreamingManager:
    def __init__(
        self,
        heights: np.ndarray,
        manifest: QuadtreeManifest,
        budgets: StreamingBudgets,
        memory_target_blocks: int = 0,
    ) -> None:
        self.heights = heights
        self.manifest = manifest
        self.budgets = budgets
        self.memory_target_blocks = int(memory_target_blocks)
        self.resident: dict[str, ResidentEntry] = {}
        self.ledger_events: list[dict[str, Any]] = []
        self._ledger_state: dict[str, str] = {}
        self._memory_used = 0
        self._vram_used = 0
        self._load_counts: dict[str, int] = {}

    @property
    def memory_used(self) -> int:
        return int(self._memory_used)

    @property
    def vram_used(self) -> int:
        return int(self._vram_used)

    def _event(
        self,
        frame: int,
        block_id: str,
        action: str,
        reason: str,
        parameter_stamp: str,
        completed: bool = True,
        bytes_delta: tuple[int, int] = (0, 0),
    ) -> dict[str, Any]:
        event = {
            "frame": int(frame),
            "block_id": block_id,
            "action": action,
            "reason": reason,
            "completed": bool(completed),
            "memory_delta": int(bytes_delta[0]),
            "vram_delta": int(bytes_delta[1]),
            "memory_used_bytes": int(self._memory_used),
            "vram_used_bytes": int(self._vram_used),
            "memory_budget_bytes": int(self.budgets.memory_bytes),
            "vram_budget_bytes": int(self.budgets.vram_bytes),
            "parameter_stamp": parameter_stamp,
        }
        self.ledger_events.append(event)
        return event

    def _payload(self, node: QuadtreeNode) -> tuple[bytes, str]:
        if node.block_id is None or node.sample_x1 >= self.heights.shape[1] or node.sample_z1 >= self.heights.shape[0]:
            fail(ErrorCode.E_BLOCK_MISSING, "block sample range is outside terrain", block_id=node.block_id)
        payload = np.ascontiguousarray(
            self.heights[node.sample_z0 : node.sample_z1 + 1, node.sample_x0 : node.sample_x1 + 1],
            dtype="<u2",
        ).tobytes(order="C")
        digest = sha256_bytes(payload)
        if digest != node.payload_sha256:
            fail(
                ErrorCode.E_BLOCK_MISSING,
                "block payload hash does not match quadtree manifest",
                block_id=node.block_id,
                expected_sha256=node.payload_sha256,
                actual_sha256=digest,
            )
        return payload, digest

    def _load(self, block_id: str, frame: int, reason: str, parameter_stamp: str) -> None:
        node = self.manifest.get_block(block_id)
        payload, digest = self._payload(node)
        if block_id in self.resident:
            fail(ErrorCode.E_RESIDENT_LEAK, "block was loaded twice without unload", block_id=block_id)
        self._load_counts[block_id] = self._load_counts.get(block_id, 0) + 1
        entry = ResidentEntry(
            block_id=block_id,
            memory_bytes=int(node.memory_bytes),
            vram_bytes=int(node.vram_bytes),
            content_sha256=digest,
            loaded_frame=int(frame),
            last_used_frame=int(frame),
            load_count=self._load_counts[block_id],
        )
        self.resident[block_id] = entry
        self._memory_used += entry.memory_bytes
        self._vram_used += entry.vram_bytes
        self._ledger_state[block_id] = "resident"
        self._event(
            frame,
            block_id,
            "LOAD_COMPLETE",
            reason,
            parameter_stamp,
            completed=True,
            bytes_delta=(entry.memory_bytes, entry.vram_bytes),
        )

    def _unload(self, block_id: str, frame: int, reason: str, parameter_stamp: str) -> None:
        entry = self.resident.get(block_id)
        if entry is None:
            fail(ErrorCode.E_RESIDENT_LEAK, "attempted to unload a non-resident block", block_id=block_id)
        del self.resident[block_id]
        self._memory_used -= entry.memory_bytes
        self._vram_used -= entry.vram_bytes
        self._ledger_state[block_id] = "absent"
        self._event(
            frame,
            block_id,
            "UNLOAD",
            reason,
            parameter_stamp,
            completed=True,
            bytes_delta=(-entry.memory_bytes, -entry.vram_bytes),
        )

    def _can_fit(self, additional_memory: int, additional_vram: int) -> bool:
        return (
            self._memory_used + additional_memory <= self.budgets.memory_bytes
            and self._vram_used + additional_vram <= self.budgets.vram_bytes
        )

    def _required_for(self, block_ids: Iterable[str]) -> tuple[int, int]:
        memory = 0
        vram = 0
        for block_id in block_ids:
            if block_id not in self.resident:
                node = self.manifest.get_block(block_id)
                memory += node.memory_bytes
                vram += node.vram_bytes
        return memory, vram

    def _evict_lru(
        self,
        frame: int,
        protected: set[str],
        required_memory: int,
        required_vram: int,
        parameter_stamp: str,
        target_blocks: int = 0,
    ) -> list[str]:
        evicted: list[str] = []
        candidates = sorted(
            (entry for block_id, entry in self.resident.items() if block_id not in protected),
            key=lambda entry: (entry.last_used_frame, entry.block_id),
        )
        for entry in candidates:
            needs_capacity = not self._can_fit(required_memory, required_vram)
            needs_count = target_blocks > 0 and len(self.resident) > target_blocks
            if not needs_capacity and not needs_count:
                break
            self._unload(entry.block_id, frame, "pressure_lru_eviction", parameter_stamp)
            evicted.append(entry.block_id)
        return evicted

    def step(
        self,
        frame: int,
        visible_block_ids: Iterable[str],
        prefetch_block_ids: Iterable[str],
        parameter_stamp: str,
    ) -> StreamingStepResult:
        visible = tuple(sorted(set(visible_block_ids)))
        prefetch = tuple(sorted(set(prefetch_block_ids) - set(visible)))
        protected = set(visible)
        desired = protected | set(prefetch)
        for block_id in visible:
            self.manifest.get_block(block_id)

        required_memory, required_vram = self._required_for(visible)
        missing_visible = [block_id for block_id in visible if block_id not in self.resident]
        if self._memory_used + required_memory > self.budgets.memory_bytes or self._vram_used + required_vram > self.budgets.vram_bytes:
            max_evictable_memory = sum(
                entry.memory_bytes for block_id, entry in self.resident.items() if block_id not in protected
            )
            max_evictable_vram = sum(
                entry.vram_bytes for block_id, entry in self.resident.items() if block_id not in protected
            )
            if (
                self._memory_used - max_evictable_memory + required_memory > self.budgets.memory_bytes
                or self._vram_used - max_evictable_vram + required_vram > self.budgets.vram_bytes
            ):
                fail(
                    ErrorCode.E_BUDGET_DENIED,
                    "mandatory visible working set cannot fit in declared budgets",
                    frame=int(frame),
                    visible_blocks=len(visible),
                    required_memory_bytes=required_memory,
                    required_vram_bytes=required_vram,
                    memory_budget_bytes=self.budgets.memory_bytes,
                    vram_budget_bytes=self.budgets.vram_bytes,
                )

        all_evicted = self._evict_lru(
            frame,
            protected,
            required_memory,
            required_vram,
            parameter_stamp,
        )
        for block_id in missing_visible:
            self._load(block_id, frame, "mandatory_visible", parameter_stamp)

        for block_id in visible:
            entry = self.resident[block_id]
            entry.last_used_frame = int(frame)
            self._event(frame, block_id, "RETAIN", "visible_resident", parameter_stamp)

        prefetch_requested: list[str] = []
        prefetch_skipped: list[str] = []
        pressure_reduced = bool(all_evicted)
        for block_id in prefetch:
            node = self.manifest.get_block(block_id)
            if block_id in self.resident:
                prefetch_requested.append(block_id)
                self.resident[block_id].last_used_frame = int(frame)
                continue
            prefetch_requested.append(block_id)
            if not self._can_fit(node.memory_bytes, node.vram_bytes):
                prefetch_skipped.append(block_id)
                continue
            if self.memory_target_blocks > 0 and len(self.resident) >= self.memory_target_blocks:
                prefetch_skipped.append(block_id)
                continue
            self._load(block_id, frame, "speculative_prefetch", parameter_stamp)

        if prefetch_skipped:
            pressure_reduced = True

        post_load_unloads: list[str] = []
        for block_id in sorted(set(self.resident) - desired):
            self._unload(block_id, frame, "not_in_desired_set", parameter_stamp)
            post_load_unloads.append(block_id)

        if self.memory_target_blocks > 0:
            post_load_unloads.extend(
                self._evict_lru(
                    frame,
                    protected,
                    0,
                    0,
                    parameter_stamp,
                    target_blocks=self.memory_target_blocks,
                )
            )

        visible_missing = tuple(block_id for block_id in visible if block_id not in self.resident)
        if visible_missing:
            fail(
                ErrorCode.E_BUDGET_DENIED,
                "visible block is missing after load/eviction ordering",
                frame=int(frame),
                missing_blocks=list(visible_missing),
            )
        if self._memory_used > self.budgets.memory_bytes or self._vram_used > self.budgets.vram_bytes:
            fail(
                ErrorCode.E_BUDGET_DENIED,
                "streaming resident bytes exceed declared budget",
                memory_used_bytes=self._memory_used,
                vram_used_bytes=self._vram_used,
            )

        digest_input = {
            "visible": [
                [block_id, self.resident[block_id].content_sha256]
                for block_id in visible
            ],
        }
        draw_digest = sha256_bytes(canonical_bytes(digest_input, newline=False))
        all_unloads = tuple(sorted((set(all_evicted) | set(post_load_unloads)) - protected))
        return StreamingStepResult(
            frame=int(frame),
            visible_block_ids=visible,
            load_requests=tuple(missing_visible),
            completed_loads=tuple(missing_visible),
            retained_blocks=visible,
            unloads=all_unloads,
            prefetch_requested=tuple(prefetch_requested),
            prefetch_skipped=tuple(prefetch_skipped),
            memory_used_bytes=self._memory_used,
            vram_used_bytes=self._vram_used,
            memory_budget_bytes=self.budgets.memory_bytes,
            vram_budget_bytes=self.budgets.vram_bytes,
            memory_utilization=self._memory_used / float(self.budgets.memory_bytes),
            vram_utilization=self._vram_used / float(self.budgets.vram_bytes),
            resident_blocks=len(self.resident),
            pressure_reduced_residency=pressure_reduced,
            visible_missing=visible_missing,
            draw_digest=draw_digest,
        )

    def audit(self) -> dict[str, Any]:
        inconsistent = sorted(
            block_id
            for block_id in set(self.resident) | {key for key, state in self._ledger_state.items() if state == "resident"}
            if (block_id in self.resident) != (self._ledger_state.get(block_id) == "resident")
        )
        memory_sum = sum(entry.memory_bytes for entry in self.resident.values())
        vram_sum = sum(entry.vram_bytes for entry in self.resident.values())
        if inconsistent or memory_sum != self._memory_used or vram_sum != self._vram_used:
            fail(
                ErrorCode.E_RESIDENT_LEAK,
                "resident accounting does not match ledger state",
                inconsistent_blocks=inconsistent,
                resident_memory_sum=memory_sum,
                accounted_memory=self._memory_used,
                resident_vram_sum=vram_sum,
                accounted_vram=self._vram_used,
            )
        if self._memory_used > self.budgets.memory_bytes or self._vram_used > self.budgets.vram_bytes:
            fail(
                ErrorCode.E_BUDGET_DENIED,
                "resident accounting exceeds budget during audit",
                memory_used_bytes=self._memory_used,
                vram_used_bytes=self._vram_used,
            )
        return {
            "resident_blocks": len(self.resident),
            "memory_used_bytes": self._memory_used,
            "vram_used_bytes": self._vram_used,
            "ledger_event_count": len(self.ledger_events),
            "state": "OK",
        }

    def inject_resident_leak(self) -> None:
        self._memory_used += 1

    def close(self, frame: int, parameter_stamp: str) -> tuple[str, ...]:
        self.memory_target_blocks = min(self.memory_target_blocks, 1) if self.memory_target_blocks else 0
        closed = tuple(sorted(self.resident))
        for block_id in closed:
            self._unload(block_id, frame, "shutdown", parameter_stamp)
        return closed

    def resident_report(self, frame: int) -> list[dict[str, Any]]:
        return [self.resident[block_id].to_report(frame) for block_id in sorted(self.resident)]

    def history_report(self) -> list[dict[str, Any]]:
        events_by_block: dict[str, list[dict[str, Any]]] = {}
        for event in self.ledger_events:
            events_by_block.setdefault(event["block_id"], []).append(
                {
                    "frame": event["frame"],
                    "action": event["action"],
                    "reason": event["reason"],
                    "completed": event["completed"],
                    "memory_delta": event["memory_delta"],
                    "vram_delta": event["vram_delta"],
                }
            )
        rows: list[dict[str, Any]] = []
        for block_id in sorted(events_by_block):
            actions = events_by_block[block_id]
            node = self.manifest.get_block(block_id)
            rows.append(
                {
                    "block_id": block_id,
                    "memory_bytes": node.memory_bytes,
                    "vram_bytes": node.vram_bytes,
                    "payload_sha256": node.payload_sha256,
                    "load_count": sum(1 for action in actions if action["action"] == "LOAD_COMPLETE"),
                    "unload_count": sum(1 for action in actions if action["action"] == "UNLOAD"),
                    "final_resident": block_id in self.resident,
                    "history": actions,
                }
            )
        return rows

    def write_ledger(self, root: str | Path, filename: str = "residency_ledger.jsonl") -> tuple[Path, str]:
        path = Path(root) / "ledgers" / filename
        digest = write_jsonl(path, self.ledger_events)
        return path, digest

    def write_history(self, root: str | Path, filename: str = "residency_history.json") -> tuple[Path, str]:
        path = Path(root) / "ledgers" / filename
        digest = write_json(path, {"blocks": self.history_report(), "event_count": len(self.ledger_events)})
        return path, digest
