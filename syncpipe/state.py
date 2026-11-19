"""Deterministic state primitives.

This module is deliberately free of any I/O, storage or in-memory cache
concerns so that the online synthesizer, the crash recovery path and the
independent validator all share exactly the same apply/hash semantics.

All hashing goes through :func:`canonical_bytes`, which pins JSON
serialization (sorted keys, fixed separators, UTF-8); two identical logical
objects therefore always hash to the same digest regardless of process,
machine or arrival scheduling.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from .errors import E_CONFLICT_UNRESOLVED, PipelineError

SNAPSHOT_SCHEMA = "syncpipe.snapshot/v1"
CONFLICT_BASIS = "field_lww:commit_order(peer_id,local_seq)"


def to_native(value: Any) -> Any:
    """Convert numpy scalars/arrays and other exotic types to plain JSON types."""
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if hasattr(value, "item"):
        try:
            return to_native(value.item())
        except (ValueError, TypeError):
            pass
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        to_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_hex(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def initial_state() -> Dict[str, Any]:
    return {"entities": {}, "tombstones": []}


def state_digest(state: Dict[str, Any]) -> str:
    """Hash of the logical state content only (version independent)."""
    payload = {
        "entities": state["entities"],
        "tombstones": sorted(state["tombstones"]),
    }
    return digest_hex(payload)


def snapshot_payload(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "snapshot_hash"}


def snapshot_digest(snapshot: Dict[str, Any]) -> str:
    return digest_hex(snapshot_payload(snapshot))


def _field_writes(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload") or {}
    fields = payload.get("fields")
    return dict(fields) if isinstance(fields, dict) else {}


def adjudicate_tick(tick_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Produce deterministic field-level LWW adjudication records for a tick.

    ``tick_events`` must already be in commit order (peer_id, local_seq
    ascending). For every entity field written by more than one peer in the
    tick, the last writer in commit order wins and every earlier writer is
    recorded as overridden together with the adjudication basis.
    """
    records: List[Dict[str, Any]] = []
    per_entity: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for event in tick_events:
        payload = event.get("payload") or {}
        entity_id = payload.get("entity_id")
        if entity_id is None:
            continue
        field_map = per_entity.setdefault(entity_id, {})
        for field, value in _field_writes(event).items():
            field_map.setdefault(field, []).append(
                {
                    "peer_id": event["peer_id"],
                    "local_seq": event["local_seq"],
                    "value": to_native(value),
                }
            )
    for entity_id in sorted(per_entity):
        for field in sorted(per_entity[entity_id]):
            writers = per_entity[entity_id][field]
            if len(writers) <= 1:
                continue
            winner = writers[-1]
            records.append(
                {
                    "tick": tick_events[0]["tick"],
                    "entity_id": entity_id,
                    "field": field,
                    "winner": {"peer_id": winner["peer_id"], "local_seq": winner["local_seq"]},
                    "overridden": [
                        {"peer_id": w["peer_id"], "local_seq": w["local_seq"], "value": w["value"]}
                        for w in writers[:-1]
                    ],
                    "basis": CONFLICT_BASIS,
                }
            )
    return records


def apply_event(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one event to ``state`` purely (returns a new state dict).

    Structural conflicts (duplicate spawn, write/destroy on a missing
    entity, respawn of a tombstoned entity) raise E_CONFLICT_UNRESOLVED;
    field-level concurrent writes are converged by commit-order LWW and
    therefore never raise.
    """
    payload = event.get("payload") or {}
    entity_id = payload.get("entity_id")
    op_type = event["op_type"]
    entities = state["entities"]
    tombstones = list(state["tombstones"])

    if op_type == "spawn":
        if entity_id in entities:
            raise PipelineError(
                E_CONFLICT_UNRESOLVED,
                f"spawn target {entity_id!r} already exists",
                tick=event["tick"],
                event_loc=(event["peer_id"], event["local_seq"]),
            )
        if entity_id in tombstones:
            raise PipelineError(
                E_CONFLICT_UNRESOLVED,
                f"spawn target {entity_id!r} was already destroyed",
                tick=event["tick"],
                event_loc=(event["peer_id"], event["local_seq"]),
            )
        new_entities = dict(entities)
        new_entities[entity_id] = {
            "entity_id": entity_id,
            "spawned_by": event["peer_id"],
            "fields": to_native(_field_writes(event)),
        }
        return {"entities": new_entities, "tombstones": tombstones}

    if entity_id not in entities:
        raise PipelineError(
            E_CONFLICT_UNRESOLVED,
            f"{op_type} targets missing entity {entity_id!r}",
            tick=event["tick"],
            event_loc=(event["peer_id"], event["local_seq"]),
        )

    if op_type == "destroy":
        new_entities = dict(entities)
        del new_entities[entity_id]
        if entity_id not in tombstones:
            tombstones.append(entity_id)
        tombstones.sort()
        return {"entities": new_entities, "tombstones": tombstones}

    if op_type in ("move", "update"):
        entity = entities[entity_id]
        merged_fields = dict(entity["fields"])
        merged_fields.update(to_native(_field_writes(event)))
        new_entity = dict(entity)
        new_entity["fields"] = merged_fields
        new_entities = dict(entities)
        new_entities[entity_id] = new_entity
        return {"entities": new_entities, "tombstones": tombstones}

    raise PipelineError(
        E_CONFLICT_UNRESOLVED,
        f"unknown op_type {op_type!r}",
        tick=event["tick"],
        event_loc=(event["peer_id"], event["local_seq"]),
    )


def build_snapshot(
    version: int,
    tick: int,
    parent_version: Any,
    state: Dict[str, Any],
    source_event: Any,
) -> Dict[str, Any]:
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "version": version,
        "tick": tick,
        "parent_version": parent_version,
        "state": state,
        "state_hash": state_digest(state),
        "source_event": source_event,
    }
    snapshot["snapshot_hash"] = snapshot_digest(snapshot)
    return snapshot


def event_ref(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tick": event["tick"],
        "peer_id": event["peer_id"],
        "local_seq": event["local_seq"],
        "op_type": event["op_type"],
    }
