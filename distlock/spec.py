"""Loading and strong validation of domain/holder declarations and event stream.

Validation failures abort the *whole* load (reject-as-a-unit) with a structured
error carrying file, 1-based line number and field name. General declaration
problems raise :class:`SpecError` (``E_SPEC``); the two injectable timing and
reference violations use their specialised codes ``E_TIMING`` / ``E_REF`` while
remaining specification rejections.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ReferenceError_, SpecError, TimingSpecError
from .models import ACTIONS, ACTION_CN, Domain, Holder, LeaseEvent


def _is_int(v: Any) -> bool:
    # bool is a subclass of int but is not an acceptable integer here.
    return isinstance(v, int) and not isinstance(v, bool)


def _require_int(obj: dict[str, Any], key: str, where: str, line: int | None) -> int:
    if key not in obj:
        raise SpecError(
            f"missing required integer field '{key}'",
            {"file": where, "line": line, "field": key},
        )
    if not _is_int(obj[key]):
        raise SpecError(
            f"field '{key}' must be an integer",
            {"file": where, "line": line, "field": key, "actual": type(obj[key]).__name__},
        )
    return obj[key]


def load_spec(spec_path: str | Path) -> tuple[dict[str, Domain], dict[str, Holder], dict[str, Any]]:
    """Load and validate the JSON declaration file."""
    p = Path(spec_path)
    if not p.exists():
        raise SpecError("spec file not found", {"file": str(p), "field": "<file>"})
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(
            "spec JSON parse error",
            {"file": str(p), "line": exc.lineno, "field": exc.msg},
        ) from exc

    if not isinstance(raw, dict):
        raise SpecError("spec root must be a JSON object", {"file": str(p), "field": "<root>"})

    session = raw.get("session", {})
    if not isinstance(session, dict):
        raise SpecError("session must be an object", {"file": str(p), "field": "session"})
    domain_set_version = _require_int(session, "domain_set_version", str(p), None)
    scale = _require_int(session, "scale_factor", str(p), None)
    if scale <= 0:
        raise SpecError("scale_factor must be positive", {"file": str(p), "field": "session.scale_factor"})
    global_concurrency = _require_int(session, "global_concurrency", str(p), None)
    if global_concurrency < 1:
        raise SpecError("global_concurrency must be >= 1",
                        {"file": str(p), "field": "session.global_concurrency"})
    per_holder_concurrency = _require_int(session, "per_holder_concurrency", str(p), None)
    if per_holder_concurrency < 1:
        raise SpecError("per_holder_concurrency must be >= 1",
                        {"file": str(p), "field": "session.per_holder_concurrency"})
    backoff = session.get("retry_backoff_ms", [0])
    if not isinstance(backoff, list) or not backoff or not all(_is_int(x) and x >= 0 for x in backoff):
        raise SpecError("retry_backoff_ms must be a non-empty list of non-negative integers",
                        {"file": str(p), "field": "session.retry_backoff_ms"})

    domains: dict[str, Domain] = {}
    for i, d in enumerate(raw.get("domains", [])):
        if not isinstance(d, dict):
            raise SpecError("domain entry must be an object",
                            {"file": str(p), "line": i + 1, "field": "domains"})
        did = d.get("domain_id")
        if not isinstance(did, str) or not did:
            raise SpecError("domain_id must be a non-empty string",
                            {"file": str(p), "line": i + 1, "field": "domain_id"})
        if did in domains:
            raise SpecError("duplicate domain_id",
                            {"file": str(p), "line": i + 1, "field": "domain_id", "value": did})
        capacity = _require_int(d, "capacity", str(p), i + 1)
        lease_ms = _require_int(d, "lease_duration_ms", str(p), i + 1)
        renew_ms = _require_int(d, "renew_interval_ms", str(p), i + 1)
        queue_limit = _require_int(d, "queue_limit", str(p), i + 1)
        if capacity < 1:
            raise SpecError("capacity must be >= 1",
                            {"file": str(p), "line": i + 1, "field": "capacity", "domain_id": did})
        if lease_ms < 1:
            raise SpecError("lease_duration_ms must be >= 1",
                            {"file": str(p), "line": i + 1, "field": "lease_duration_ms", "domain_id": did})
        if queue_limit < 0:
            raise SpecError("queue_limit must be >= 0",
                            {"file": str(p), "line": i + 1, "field": "queue_limit", "domain_id": did})
        # Hard timing rule: renewal interval must be strictly smaller than lease.
        if renew_ms >= lease_ms:
            raise TimingSpecError(
                "renew_interval_ms must be strictly less than lease_duration_ms",
                {"file": str(p), "line": i + 1, "field": "renew_interval_ms",
                 "domain_id": did, "renew_interval_ms": renew_ms,
                 "lease_duration_ms": lease_ms},
            )
        version = _require_int(d, "version", str(p), i + 1) if "version" in d else 1
        domains[did] = Domain(did, capacity, lease_ms, renew_ms, queue_limit, version)

    if not domains:
        raise SpecError("at least one domain is required",
                        {"file": str(p), "field": "domains"})

    holders: dict[str, Holder] = {}
    for i, h in enumerate(raw.get("holders", [])):
        if not isinstance(h, dict):
            raise SpecError("holder entry must be an object",
                            {"file": str(p), "line": i + 1, "field": "holders"})
        hid = h.get("holder_id")
        if not isinstance(hid, str) or not hid:
            raise SpecError("holder_id must be a non-empty string",
                            {"file": str(p), "line": i + 1, "field": "holder_id"})
        if hid in holders:
            raise SpecError("duplicate holder_id",
                            {"file": str(p), "line": i + 1, "field": "holder_id", "value": hid})
        priority = _require_int(h, "priority", str(p), i + 1)
        targets = h.get("target_domains")
        if not isinstance(targets, list) or not targets or not all(isinstance(x, str) for x in targets):
            raise SpecError("target_domains must be a non-empty list of strings",
                            {"file": str(p), "line": i + 1, "field": "target_domains", "holder_id": hid})
        for ref in targets:
            if ref not in domains:
                raise ReferenceError_(
                    "holder target domain does not exist",
                    {"file": str(p), "line": i + 1, "field": "target_domains",
                     "holder_id": hid, "unknown_domain": ref},
                )
        holders[hid] = Holder(hid, priority, tuple(targets))

    if not holders:
        raise SpecError("at least one holder is required",
                        {"file": str(p), "field": "holders"})

    session_meta = {
        "domain_set_version": domain_set_version,
        "scale_factor": scale,
        "global_concurrency": global_concurrency,
        "per_holder_concurrency": per_holder_concurrency,
        "retry_backoff_ms": list(backoff),
    }
    return domains, holders, session_meta


def load_events(events_path: str | Path,
                domains: dict[str, Domain],
                holders: dict[str, Holder]) -> list[LeaseEvent]:
    """Load and validate the NDJSON lease event stream."""
    p = Path(events_path)
    if not p.exists():
        raise SpecError("event stream not found", {"file": str(p), "field": "<file>"})

    events: list[LeaseEvent] = []
    last_seq: int | None = None
    last_ts: int | None = None
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SpecError("NDJSON parse error",
                                {"file": str(p), "line": line_no, "field": exc.msg}) from exc
            if not isinstance(obj, dict):
                raise SpecError("event must be a JSON object",
                                {"file": str(p), "line": line_no, "field": "<row>"})

            seq = _require_int(obj, "seq", str(p), line_no)
            ts = _require_int(obj, "timestamp_ms", str(p), line_no)
            expected = 1 if last_seq is None else last_seq + 1
            if last_seq is not None and seq != expected:
                raise SpecError(
                    "event seq must be consecutive and unique",
                    {"file": str(p), "line": line_no, "field": "seq",
                     "expected": expected, "actual": seq},
                )
            if last_seq is None and seq != 1:
                raise SpecError("first event seq must be 1",
                                {"file": str(p), "line": line_no, "field": "seq", "actual": seq})
            if last_ts is not None and ts < last_ts:
                raise SpecError(
                    "timestamp_ms must be monotonically non-decreasing",
                    {"file": str(p), "line": line_no, "field": "timestamp_ms",
                     "previous": last_ts, "actual": ts},
                )

            hid = obj.get("holder_id")
            if not isinstance(hid, str) or hid not in holders:
                raise SpecError("unknown holder_id",
                                {"file": str(p), "line": line_no, "field": "holder_id", "value": hid})
            did = obj.get("domain_id")
            if not isinstance(did, str) or did not in domains:
                raise SpecError("unknown domain_id",
                                {"file": str(p), "line": line_no, "field": "domain_id", "value": did})
            if did not in holders[hid].target_domains:
                raise ReferenceError_(
                    "event domain is not in holder target_domains",
                    {"file": str(p), "line": line_no, "field": "domain_id",
                     "holder_id": hid, "domain_id": did},
                )
            action_raw = obj.get("action")
            action = ACTION_CN.get(action_raw, action_raw)
            if action not in ACTIONS:
                raise SpecError(
                    "action must be one of 申请/续约/释放 (ACQUIRE/RENEW/RELEASE)",
                    {"file": str(p), "line": line_no, "field": "action", "value": action_raw},
                )

            events.append(LeaseEvent(seq, ts, hid, did, action, line_no))
            last_seq, last_ts = seq, ts

    if not events:
        raise SpecError("event stream is empty", {"file": str(p), "field": "<file>"})
    return events
