"""实例、故障域与故障事件流的加载及强校验(业务码 E_SPEC)。

任一不合法整体拒绝, 并在 details 中指出首个出错行与字段。
JSON 数组声明的"行"定义为元素序号(从 1 开始), NDJSON 为物理行号。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import DiagError
from .util import digest_obj

SUPPORTED_ISOLATION_LEVELS = (0, 1, 2, 3)
FAULT_TYPES = ("degrade", "crash", "flap", "recover")


@dataclass(frozen=True)
class InstanceSpec:
    id: str
    domain: str
    weight: int
    replica: int


@dataclass(frozen=True)
class DomainSpec:
    id: str
    isolation_level: int
    recovery_cooldown_ms: int


@dataclass(frozen=True)
class Event:
    seq: int
    ts_ms: int
    instance: str
    fault: str


def _fail(line: int, field: str, reason: str) -> None:
    raise DiagError("E_SPEC", f"line {line} field {field}: {reason}",
                    {"line": line, "field": field, "reason": reason})


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load_json_list(path, kind: str) -> list:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DiagError("E_SPEC", f"{kind} file not found: {p}",
                        {"line": 0, "field": "<file>", "reason": "not found"})
    except json.JSONDecodeError as exc:
        raise DiagError("E_SPEC", f"{kind} invalid JSON: {exc}",
                        {"line": exc.lineno, "field": "<json>", "reason": str(exc)})
    if not isinstance(data, list):
        _fail(1, "<root>", f"{kind} root must be a JSON array")
    return data


def load_domains(path) -> dict[str, DomainSpec]:
    domains: dict[str, DomainSpec] = {}
    for idx, item in enumerate(_load_json_list(path, "domains"), start=1):
        if not isinstance(item, dict):
            _fail(idx, "<item>", "domain entry must be an object")
        did = item.get("id")
        if not isinstance(did, str) or not did:
            _fail(idx, "id", "domain id must be a non-empty string")
        if did in domains:
            _fail(idx, "id", f"duplicate domain id {did!r}")
        level = item.get("isolation_level")
        if not _is_int(level) or level not in SUPPORTED_ISOLATION_LEVELS:
            _fail(idx, "isolation_level",
                  f"must be a supported integer in {SUPPORTED_ISOLATION_LEVELS}")
        cooldown = item.get("recovery_cooldown_ms")
        if not _is_int(cooldown) or cooldown < 0:
            _fail(idx, "recovery_cooldown_ms", "must be a non-negative integer")
        domains[did] = DomainSpec(did, level, cooldown)
    if not domains:
        _fail(1, "<root>", "at least one domain is required")
    return domains


def load_instances(path, domains: dict[str, DomainSpec]) -> dict[str, InstanceSpec]:
    instances: dict[str, InstanceSpec] = {}
    for idx, item in enumerate(_load_json_list(path, "instances"), start=1):
        if not isinstance(item, dict):
            _fail(idx, "<item>", "instance entry must be an object")
        iid = item.get("id")
        if not isinstance(iid, str) or not iid:
            _fail(idx, "id", "instance id must be a non-empty string")
        if iid in instances:
            _fail(idx, "id", f"duplicate instance id {iid!r}")
        domain = item.get("domain")
        if not isinstance(domain, str) or domain not in domains:
            _fail(idx, "domain", f"references unknown domain {domain!r}")
        weight = item.get("weight")
        if not _is_int(weight) or weight < 0:
            _fail(idx, "weight", "must be a non-negative integer")
        replica = item.get("replica")
        if not _is_int(replica) or replica < 1:
            _fail(idx, "replica", "must be a positive integer")
        instances[iid] = InstanceSpec(iid, domain, weight, replica)
    if not instances:
        _fail(1, "<root>", "at least one instance is required")
    return instances


def load_events(path, instances: dict[str, InstanceSpec]) -> list[Event]:
    events: list[Event] = []
    p = Path(path)
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise DiagError("E_SPEC", f"events file not found: {p}",
                        {"line": 0, "field": "<file>", "reason": "not found"})
    prev_ts = 0
    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            _fail(lineno, "<json>", f"invalid JSON: {exc}")
        if not isinstance(item, dict):
            _fail(lineno, "<item>", "event must be an object")
        seq = item.get("seq")
        expected_seq = len(events) + 1
        if not _is_int(seq) or seq != expected_seq:
            _fail(lineno, "seq", f"must be the continuous unique integer {expected_seq}")
        ts_ms = item.get("ts_ms")
        if not _is_int(ts_ms) or ts_ms < 0:
            _fail(lineno, "ts_ms", "must be a non-negative integer")
        if ts_ms < prev_ts:
            _fail(lineno, "ts_ms", "must be monotonically non-decreasing")
        prev_ts = ts_ms
        iid = item.get("instance")
        if not isinstance(iid, str) or iid not in instances:
            _fail(lineno, "instance", f"references unknown instance {iid!r}")
        fault = item.get("fault")
        if fault not in FAULT_TYPES:
            _fail(lineno, "fault", f"must be one of {FAULT_TYPES}")
        events.append(Event(seq, ts_ms, iid, fault))
    if not events:
        _fail(1, "<root>", "at least one event is required")
    return events


def instances_version(instances: dict[str, InstanceSpec]) -> str:
    return digest_obj({iid: vars(spec) for iid, spec in sorted(instances.items())})


def domains_version(domains: dict[str, DomainSpec]) -> str:
    return digest_obj({did: vars(spec) for did, spec in sorted(domains.items())})


def events_digest(events: list[Event]) -> str:
    return digest_obj([vars(e) for e in events])
