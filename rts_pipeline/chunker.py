"""Deterministic tick-range chunking with boundary snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_json,
    canonical_bytes,
    hash_value,
    hash_without,
    read_json,
)
from .constants import CHUNK_VERSION, SCHEMA_VERSION
from .engine import synthesize_game
from .errors import PipelineError, require
from .events import events_digest, normalize_events
from .rules import initial_state, rules_hash, validate_rules

FIXED_BLOCK_MEMORY = 4096
ESTIMATED_COPY_FACTOR = 4


def _single_chunk(
    chunk_id: int,
    start_tick: int,
    end_tick: int,
    events: list[dict[str, Any]],
    entry_snapshot: dict[str, Any],
    exit_snapshot: dict[str, Any],
    game: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_version": CHUNK_VERSION,
        "chunk_id": chunk_id,
        "start_tick": start_tick,
        "end_tick": end_tick,
        "game_seed": game["seed"],
        "rules_hash": rules_hash(rules),
        "entry_snapshot": entry_snapshot,
        "exit_snapshot": exit_snapshot,
        "events": events,
    }
    content["content_hash"] = hash_without(content, "content_hash")
    return content


def estimate_block_memory(chunk: dict[str, Any]) -> int:
    payload_bytes = len(canonical_bytes(chunk["events"])) + len(
        canonical_bytes(chunk["entry_snapshot"])
    ) + len(canonical_bytes(chunk["exit_snapshot"]))
    return FIXED_BLOCK_MEMORY + ESTIMATED_COPY_FACTOR * payload_bytes


def prepare_chunks(
    root: Path,
    events: Iterable[dict[str, Any]],
    rules: dict[str, Any],
    game: dict[str, Any],
    chunk_ticks: int,
) -> dict[str, Any]:
    normalized_rules = validate_rules(rules)
    normalized_events = normalize_events(events, normalized_rules)
    require(chunk_ticks > 0, "E_SCHEMA", "chunk_ticks must be positive")
    tick_start = game["tick_start"]
    tick_end = game["tick_end"]
    require(
        isinstance(tick_start, int)
        and isinstance(tick_end, int)
        and tick_end >= tick_start,
        "E_SCHEMA",
        "game tick range is invalid",
    )
    base_state = initial_state(normalized_rules)
    boundary_result = synthesize_game(
        events=normalized_events,
        rules=normalized_rules,
        seed=game["seed"],
        tick_start=tick_start,
        tick_end=tick_end,
        initial_snapshot=base_state,
    )
    ticks_by_number = {record["tick"]: record for record in boundary_result["tick_records"]}
    events_by_tick: dict[int, list[dict[str, Any]]] = {}
    for event in normalized_events:
        events_by_tick.setdefault(event["tick"], []).append(event)

    chunks_dir = Path(root) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_chunks: list[dict[str, Any]] = []
    chunk_id = 0
    start = tick_start
    while start <= tick_end:
        end = min(start + chunk_ticks - 1, tick_end)
        entry_snapshot = (
            base_state
            if start == tick_start
            else ticks_by_number[start - 1]["snapshot"]
        )
        exit_snapshot = ticks_by_number[end]["snapshot"]
        chunk_events: list[dict[str, Any]] = []
        for tick in range(start, end + 1):
            chunk_events.extend(events_by_tick.get(tick, []))
        chunk = _single_chunk(
            chunk_id=chunk_id,
            start_tick=start,
            end_tick=end,
            events=chunk_events,
            entry_snapshot=entry_snapshot,
            exit_snapshot=exit_snapshot,
            game=game,
            rules=normalized_rules,
        )
        filename = f"chunk-{chunk_id:06d}.json"
        atomic_write_json(chunks_dir / filename, chunk, pretty=False)
        manifest_chunks.append(
            {
                "chunk_id": chunk_id,
                "start_tick": start,
                "end_tick": end,
                "tick_count": end - start + 1,
                "event_count": len(chunk_events),
                "file": f"chunks/{filename}",
                "content_hash": chunk["content_hash"],
                "entry_snapshot_hash": hash_value(entry_snapshot),
                "exit_snapshot_hash": hash_value(exit_snapshot),
                "estimated_memory_bytes": estimate_block_memory(chunk),
            }
        )
        chunk_id += 1
        start = end + 1

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_version": CHUNK_VERSION,
        "chunk_ticks": chunk_ticks,
        "tick_start": tick_start,
        "tick_end": tick_end,
        "chunk_count": len(manifest_chunks),
        "event_count": len(normalized_events),
        "events_hash": events_digest(normalized_events),
        "rules_hash": rules_hash(normalized_rules),
        "game_seed": game["seed"],
        "initial_snapshot": base_state,
        "initial_snapshot_hash": hash_value(base_state),
        "chunks": manifest_chunks,
    }
    manifest["manifest_hash"] = hash_without(manifest, "manifest_hash")
    atomic_write_json(chunks_dir / "manifest.json", manifest)
    return manifest


def load_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(
        Path(root) / "chunks" / "manifest.json", error_code="E_CHUNK_BOUNDARY"
    )
    if not isinstance(manifest, dict):
        raise PipelineError("E_CHUNK_BOUNDARY", "chunk manifest must be an object")
    actual = manifest.get("manifest_hash")
    expected = hash_without(manifest, "manifest_hash")
    if actual != expected:
        raise PipelineError(
            "E_CHUNK_BOUNDARY",
            "chunk manifest hash mismatch",
            expected=expected,
            actual=actual,
        )
    return manifest


def load_chunk(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    chunk = read_json(Path(root) / entry["file"], error_code="E_CHUNK_BOUNDARY")
    if not isinstance(chunk, dict):
        raise PipelineError("E_CHUNK_BOUNDARY", "chunk content must be an object", chunk_id=entry["chunk_id"])
    actual = chunk.get("content_hash")
    expected = hash_without(chunk, "content_hash")
    if actual != expected:
        raise PipelineError(
            "E_CHUNK_BOUNDARY",
            "chunk content hash mismatch",
            chunk_id=entry["chunk_id"],
            expected=expected,
            actual=actual,
        )
    for field in ("chunk_id", "start_tick", "end_tick"):
        if chunk.get(field) != entry.get(field):
            raise PipelineError(
                "E_CHUNK_BOUNDARY",
                "chunk metadata disagrees with manifest",
                chunk_id=entry["chunk_id"],
                field=field,
                manifest=entry.get(field),
                chunk=chunk.get(field),
            )
    if hash_value(chunk.get("entry_snapshot")) != entry["entry_snapshot_hash"]:
        raise PipelineError("E_CHUNK_BOUNDARY", "chunk entry snapshot hash mismatch", chunk_id=entry["chunk_id"])
    if hash_value(chunk.get("exit_snapshot")) != entry["exit_snapshot_hash"]:
        raise PipelineError("E_CHUNK_BOUNDARY", "chunk exit snapshot hash mismatch", chunk_id=entry["chunk_id"])
    return chunk
