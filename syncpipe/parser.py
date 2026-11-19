"""Event stream parsing, continuity validation and deterministic ordering."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .errors import E_PEER_GAP, make_error

VALID_OP_TYPES = ("spawn", "move", "update", "destroy")
REQUIRED_FIELDS = ("tick", "peer_id", "op_type", "payload", "base_snapshot_version", "local_seq")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_stream(lines) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse JSONL event lines.

    Returns ``(tick_groups, errors, quarantined)`` where each tick group is a
    list of events sorted by ``(peer_id, local_seq)``. Illegal ticks and
    broken per-peer sequences are reported as E_PEER_GAP (with a precise
    ``reason``) and quarantined rather than silently dropped.
    """
    raw_events: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []

    for line_no, line in enumerate(lines, start=1):
        text = line.strip() if isinstance(line, str) else line
        if not text:
            continue
        try:
            event = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(
                make_error(
                    E_PEER_GAP,
                    f"malformed_event_line line={line_no}: {exc}",
                    version=None,
                )
            )
            quarantined.append({"line_no": line_no, "raw": text, "reason": "malformed_json"})
            continue
        if not isinstance(event, dict) or any(field not in event for field in REQUIRED_FIELDS):
            errors.append(
                make_error(
                    E_PEER_GAP,
                    f"missing_required_field line={line_no} required={list(REQUIRED_FIELDS)}",
                    version=None,
                )
            )
            quarantined.append({"line_no": line_no, "raw": event, "reason": "missing_field"})
            continue
        if not _is_int(event["tick"]) or event["tick"] < 0:
            errors.append(
                make_error(
                    E_PEER_GAP,
                    f"illegal_tick tick={event['tick']!r} line={line_no}",
                    version=None,
                    tick=event["tick"] if _is_int(event["tick"]) else None,
                    event_loc=(str(event["peer_id"]), event["local_seq"])
                    if _is_int(event.get("local_seq"))
                    else None,
                )
            )
            quarantined.append({"line_no": line_no, "raw": event, "reason": "illegal_tick"})
            continue
        if not isinstance(event["peer_id"], str) or not _is_int(event["local_seq"]) or event["local_seq"] < 0:
            errors.append(
                make_error(
                    E_PEER_GAP,
                    f"illegal_peer_or_seq line={line_no} peer_id={event.get('peer_id')!r} "
                    f"local_seq={event.get('local_seq')!r}",
                    version=None,
                    tick=event["tick"],
                )
            )
            quarantined.append({"line_no": line_no, "raw": event, "reason": "illegal_peer_or_seq"})
            continue
        if event["op_type"] not in VALID_OP_TYPES:
            errors.append(
                make_error(
                    E_PEER_GAP,
                    f"unknown_op_type {event['op_type']!r} line={line_no}",
                    version=None,
                    tick=event["tick"],
                    event_loc=(event["peer_id"], event["local_seq"]),
                )
            )
            quarantined.append({"line_no": line_no, "raw": event, "reason": "unknown_op_type"})
            continue
        if not isinstance(event["payload"], dict) or not _is_int(event["base_snapshot_version"]):
            errors.append(
                make_error(
                    E_PEER_GAP,
                    f"illegal_payload_or_base_version line={line_no}",
                    version=None,
                    tick=event["tick"],
                    event_loc=(event["peer_id"], event["local_seq"]),
                )
            )
            quarantined.append({"line_no": line_no, "raw": event, "reason": "illegal_payload_or_base"})
            continue
        raw_events.append(event)

    accepted: List[Dict[str, Any]] = []
    expected_seq: Dict[str, int] = {}
    broken_peers = set()
    for event in sorted(raw_events, key=lambda e: (e["tick"], e["peer_id"], e["local_seq"])):
        peer = event["peer_id"]
        if peer in broken_peers:
            errors.append(
                make_error(
                    E_PEER_GAP,
                    f"peer_sequence_unrecoverable peer={peer} local_seq={event['local_seq']}",
                    version=None,
                    tick=event["tick"],
                    event_loc=(peer, event["local_seq"]),
                )
            )
            quarantined.append({"line_no": None, "raw": event, "reason": "peer_already_broken"})
            continue
        expected = expected_seq.get(peer, 0)
        if event["local_seq"] != expected:
            errors.append(
                make_error(
                    E_PEER_GAP,
                    f"peer_sequence_gap peer={peer} expected={expected} actual={event['local_seq']}",
                    version=None,
                    tick=event["tick"],
                    event_loc=(peer, event["local_seq"]),
                )
            )
            quarantined.append({"line_no": None, "raw": event, "reason": "sequence_gap"})
            broken_peers.add(peer)
            continue
        expected_seq[peer] = expected + 1
        accepted.append(event)

    by_tick: Dict[int, List[Dict[str, Any]]] = {}
    for event in accepted:
        by_tick.setdefault(event["tick"], []).append(event)
    tick_groups = [by_tick[tick] for tick in sorted(by_tick)]
    for group in tick_groups:
        group.sort(key=lambda e: (e["peer_id"], e["local_seq"]))
    return tick_groups, errors, quarantined
