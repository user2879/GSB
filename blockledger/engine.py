from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .blocks import (
    ContentStore,
    MaterializedBlocks,
    initial_block_bytes,
    write_block_bytes,
)
from .canonical import canonical_bytes, content_digest, sha256_hex
from .errors import SpecError
from .spec import expand_intervals


@dataclass
class BlockState:
    block_index: int
    version: int
    baseline_version: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "version": self.version,
            "baseline_version": self.baseline_version,
            "digest": self.digest,
        }


@dataclass
class SnapshotCapture:
    snapshot_id: str
    volume_id: str
    version: int
    captured_seq: int
    intervals: list[list[int]]
    digests: dict[int, str]
    baselines: dict[int, int]
    block_versions: dict[int, int]
    status: str = "active"
    deleted_seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "volume_id": self.volume_id,
            "version": self.version,
            "captured_seq": self.captured_seq,
            "intervals": self.intervals,
            "digests": {str(key): value for key, value in sorted(self.digests.items())},
            "baselines": {str(key): value for key, value in sorted(self.baselines.items())},
            "block_versions": {str(key): value for key, value in sorted(self.block_versions.items())},
            "status": self.status,
            "deleted_seq": self.deleted_seq,
        }


@dataclass
class ReplayResult:
    spec: dict[str, Any]
    volumes: dict[str, dict[str, Any]]
    snapshots: dict[str, SnapshotCapture]
    effects: list[dict[str, Any]] = field(default_factory=list)
    rollback_plans: list[dict[str, Any]] = field(default_factory=list)
    store: ContentStore | None = None

    def fingerprint(self) -> str:
        return replay_fingerprint(self)


def _new_volume_state(volume: dict[str, Any]) -> dict[str, Any]:
    blocks: dict[int, BlockState] = {}
    contents: dict[int, bytes] = {}
    for block_index in range(int(volume["block_count"])):
        content = initial_block_bytes(volume, block_index)
        digest = content_digest(
            volume["id"], block_index, int(volume["baseline_version"]), content
        )
        blocks[block_index] = BlockState(
            block_index=block_index,
            version=1,
            baseline_version=int(volume["baseline_version"]),
            digest=digest,
        )
        contents[block_index] = content
    return {
        "config": volume,
        "baseline_version": int(volume["baseline_version"]),
        "blocks": blocks,
        "contents": contents,
    }


def replay_spec(
    spec: dict[str, Any],
    *,
    content_store: str | Path | ContentStore | None = None,
    materialized: MaterializedBlocks | str | Path | None = None,
    verify_declared_digests: bool = True,
) -> ReplayResult:
    store: ContentStore | None
    if isinstance(content_store, ContentStore):
        store = content_store
    elif content_store is not None:
        store = ContentStore(content_store)
    else:
        store = ContentStore(None)

    blocks_io: MaterializedBlocks | None
    if isinstance(materialized, MaterializedBlocks):
        blocks_io = materialized
    elif materialized is not None:
        blocks_io = MaterializedBlocks(materialized)
    else:
        blocks_io = None

    volumes: dict[str, dict[str, Any]] = {}
    for volume in sorted(spec["volumes"], key=lambda item: item["id"]):
        state = _new_volume_state(volume)
        volumes[volume["id"]] = state
        if store is not None or blocks_io is not None:
            for index, block_state in state["blocks"].items():
                content = state["contents"][index]
                if store is not None:
                    store.put(block_state.digest, content)
                if blocks_io is not None:
                    blocks_io.write(volume["id"], index, content)

    result = ReplayResult(spec=spec, volumes=volumes, snapshots={}, store=store)
    for event in spec["events"]:
        apply_event(
            result,
            event,
            store=store,
            blocks_io=blocks_io,
            verify_declared_digests=verify_declared_digests,
        )
    return result


def apply_events(
    result: ReplayResult,
    events: list[dict[str, Any]],
    *,
    store: ContentStore | None = None,
    blocks_io: MaterializedBlocks | None = None,
    verify_declared_digests: bool = True,
) -> None:
    if store is None:
        store = result.store
    if store is None:
        store = ContentStore(None)
        result.store = store
    for event in events:
        apply_event(
            result,
            event,
            store=store,
            blocks_io=blocks_io,
            verify_declared_digests=verify_declared_digests,
        )


def apply_event(
    result: ReplayResult,
    event: dict[str, Any],
    *,
    store: ContentStore | None = None,
    blocks_io: MaterializedBlocks | None = None,
    verify_declared_digests: bool = True,
) -> dict[str, Any]:
    event_type = event["type"]
    seq = int(event["seq"])

    if event_type == "write":
        effect = _apply_write(result, event, seq, store, blocks_io)
    elif event_type == "snapshot_create":
        effect = _apply_snapshot_create(
            result, event, seq, verify_declared_digests=verify_declared_digests
        )
    elif event_type == "rollback":
        effect = _apply_rollback(result, event, seq, store, blocks_io)
    elif event_type == "snapshot_delete":
        effect = _apply_delete(result, event, seq)
    else:  # pragma: no cover - loader prevents this
        raise SpecError(f"unknown event type {event_type!r}", entry_id=f"event:{seq}", field="type")
    result.effects.append(effect)
    return effect


def _persist_block(
    volume_id: str,
    index: int,
    content: bytes,
    digest: str,
    store: ContentStore | None,
    blocks_io: MaterializedBlocks | None,
) -> None:
    if store is not None:
        store.put(digest, content)
    if blocks_io is not None:
        blocks_io.write(volume_id, index, content)


def _apply_write(
    result: ReplayResult,
    event: dict[str, Any],
    seq: int,
    store: ContentStore | None,
    blocks_io: MaterializedBlocks | None,
) -> dict[str, Any]:
    volume_id = event["volume_id"]
    volume_state = result.volumes[volume_id]
    volume = volume_state["config"]
    indices = expand_intervals(event["intervals"])
    changed: list[int] = []
    for index in indices:
        content = write_block_bytes(volume, seq, int(event["seed"]), index)
        block_state: BlockState = volume_state["blocks"][index]
        block_state.version += 1
        block_state.baseline_version = int(volume_state["baseline_version"])
        block_state.digest = content_digest(
            volume_id, index, block_state.baseline_version, content
        )
        volume_state["contents"][index] = content
        _persist_block(volume_id, index, content, block_state.digest, store, blocks_io)
        changed.append(index)

    return {
        "seq": seq,
        "type": "write",
        "entry_id": event.get("event_id", f"event:{seq}"),
        "volume_id": volume_id,
        "effective_blocks": changed,
        "basis": {
            "rule": "numpy-pcg64-write-expansion",
            "event_seed": event["seed"],
            "block_version_increment": 1,
            "baseline_version": volume_state["baseline_version"],
        },
    }


def _apply_snapshot_create(
    result: ReplayResult,
    event: dict[str, Any],
    seq: int,
    *,
    verify_declared_digests: bool,
) -> dict[str, Any]:
    snapshot_id = event["snapshot_id"]
    declared = next(item for item in result.spec["snapshots"] if item["id"] == snapshot_id)
    volume_id = declared["volume_id"]
    volume_state = result.volumes[volume_id]
    indices = list(declared["blocks"])

    digests: dict[int, str] = {}
    baselines: dict[int, int] = {}
    versions: dict[int, int] = {}
    mismatches: list[dict[str, Any]] = []
    for index in indices:
        block_state: BlockState = volume_state["blocks"][index]
        digests[index] = block_state.digest
        baselines[index] = block_state.baseline_version
        versions[index] = block_state.version
        expected_declared = declared["digests"].get(str(index))
        if verify_declared_digests and expected_declared != block_state.digest:
            mismatches.append(
                {
                    "block_index": index,
                    "declared": expected_declared,
                    "actual": block_state.digest,
                }
            )

    if mismatches:
        first = mismatches[0]
        raise SpecError(
            "declared snapshot digest does not match deterministic event replay",
            entry_id=snapshot_id,
            field=f"digests.{first['block_index']}",
            expected=first["declared"],
            actual=first["actual"],
        )

    result.snapshots[snapshot_id] = SnapshotCapture(
        snapshot_id=snapshot_id,
        volume_id=volume_id,
        version=int(declared["version"]),
        captured_seq=seq,
        intervals=declared["intervals"],
        digests=digests,
        baselines=baselines,
        block_versions=versions,
    )
    return {
        "seq": seq,
        "type": "snapshot_create",
        "entry_id": snapshot_id,
        "snapshot_id": snapshot_id,
        "volume_id": volume_id,
        "effective_blocks": indices,
        "basis": {
            "rule": "freeze-current-metadata-bound-digests",
            "copied_block_bytes": False,
            "digest_count": len(digests),
        },
    }


def _apply_rollback(
    result: ReplayResult,
    event: dict[str, Any],
    seq: int,
    store: ContentStore | None,
    blocks_io: MaterializedBlocks | None,
) -> dict[str, Any]:
    snapshot_id = event["snapshot_id"]
    snapshot = result.snapshots[snapshot_id]
    volume_id = snapshot.volume_id
    volume_state = result.volumes[volume_id]
    indices = sorted(snapshot.digests)
    restored: list[int] = []

    for index in indices:
        expected_digest = snapshot.digests[index]
        content: bytes | None = None
        if store is not None:
            content = store.get(expected_digest)
        if content is None:
            content = _regenerate_snapshot_bytes(result, snapshot, index)
            if store is not None:
                store.put(expected_digest, content)
        actual_digest = content_digest(
            volume_id, index, snapshot.baselines[index], content
        )
        if actual_digest != expected_digest:
            raise SpecError(
                "internal replay failed to reproduce snapshot digest",
                entry_id=snapshot_id,
                field=f"digests.{index}",
                expected=expected_digest,
                actual=actual_digest,
            )

        block_state: BlockState = volume_state["blocks"][index]
        block_state.version += 1
        block_state.baseline_version = snapshot.baselines[index]
        block_state.digest = expected_digest
        volume_state["contents"][index] = content
        _persist_block(volume_id, index, content, expected_digest, store, blocks_io)
        restored.append(index)

    previous_baseline = int(volume_state["baseline_version"])
    volume_state["baseline_version"] = previous_baseline + 1
    plan = {
        "rollback_id": event["rollback_id"],
        "event_seq": seq,
        "volume_id": volume_id,
        "snapshot_id": snapshot_id,
        "blocks": [
            {
                "block_index": index,
                "expected_digest": snapshot.digests[index],
                "baseline_version": snapshot.baselines[index],
                "snapshot_block_version": snapshot.block_versions[index],
                "replay_block_version": volume_state["blocks"][index].version,
            }
            for index in restored
        ],
        "baseline_before": previous_baseline,
        "baseline_after": volume_state["baseline_version"],
        "max_blocks": event.get("max_blocks"),
    }
    result.rollback_plans.append(plan)
    return {
        "seq": seq,
        "type": "rollback",
        "entry_id": event["rollback_id"],
        "rollback_id": event["rollback_id"],
        "snapshot_id": snapshot_id,
        "volume_id": volume_id,
        "effective_blocks": restored,
        "basis": {
            "rule": "restore-cas-bytes-and-advance-volume-baseline",
            "baseline_before": previous_baseline,
            "baseline_after": volume_state["baseline_version"],
            "block_version_increment": 1,
        },
    }


def _regenerate_snapshot_bytes(result: ReplayResult, snapshot: SnapshotCapture, index: int) -> bytes:
    # A full replay has the bytes in memory. This fallback keeps the engine
    # usable if a future caller supplies a digest-only persisted snapshot.
    current = result.volumes[snapshot.volume_id]["contents"].get(index)
    if current is not None:
        return current
    volume = result.volumes[snapshot.volume_id]["config"]
    return initial_block_bytes(volume, index)


def _apply_delete(result: ReplayResult, event: dict[str, Any], seq: int) -> dict[str, Any]:
    snapshot_id = event["snapshot_id"]
    snapshot = result.snapshots[snapshot_id]
    snapshot.status = "deleted"
    snapshot.deleted_seq = seq
    dependent_rollbacks = [
        plan["rollback_id"]
        for plan in result.rollback_plans
        if plan["snapshot_id"] == snapshot_id
    ]
    return {
        "seq": seq,
        "type": "snapshot_delete",
        "entry_id": snapshot_id,
        "snapshot_id": snapshot_id,
        "volume_id": snapshot.volume_id,
        "effective_blocks": sorted(snapshot.digests),
        "dependent_rollbacks": dependent_rollbacks,
        "basis": {
            "rule": "tombstone-snapshot-and-propagate-dependencies",
            "removed_digest_references": len(snapshot.digests),
        },
    }


def replay_canonical(result: ReplayResult) -> dict[str, Any]:
    volumes: dict[str, Any] = {}
    for volume_id in sorted(result.volumes):
        state = result.volumes[volume_id]
        volumes[volume_id] = {
            "baseline_version": state["baseline_version"],
            "blocks": [state["blocks"][index].to_dict() for index in sorted(state["blocks"])],
        }
    snapshots = {
        snapshot_id: result.snapshots[snapshot_id].to_dict()
        for snapshot_id in sorted(result.snapshots)
    }
    return {"volumes": volumes, "snapshots": snapshots, "effects": result.effects}


def replay_fingerprint(result: ReplayResult) -> str:
    return sha256_hex(canonical_bytes(replay_canonical(result)))


def chunk_plan_blocks(plan: dict[str, Any], max_blocks: int) -> list[list[dict[str, Any]]]:
    blocks = plan["blocks"]
    limit = max(1, int(max_blocks))
    return [blocks[pos : pos + limit] for pos in range(0, len(blocks), limit)]


def spec_fingerprint(spec: dict[str, Any]) -> str:
    return sha256_hex(canonical_bytes(spec))


__all__ = [
    "BlockState",
    "SnapshotCapture",
    "ReplayResult",
    "replay_spec",
    "apply_events",
    "apply_event",
    "replay_fingerprint",
    "chunk_plan_blocks",
    "spec_fingerprint",
]
