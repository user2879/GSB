"""Loading and strict validation of local JSON / NDJSON declarations.

Four declaration kinds are supported:

* ``workers``       : worker process declarations
* ``requests``      : the deterministic request stream
* ``lease_policy``  : lease duration and global active-lease ceiling
* ``merge_rules``   : receipt normalization filling basis

Every malformed record aborts the load with :class:`SpecError` (``E_SPEC``)
naming the first offending field and its source line number.
"""

import json
import os
from dataclasses import dataclass, field

from .errors import SpecError

LOOPBACK_HOST = "127.0.0.1"


def _line_of(text, offset):
    return text.count("\n", 0, offset) + 1


def _iter_json_records(text, source):
    """Yield ``(record, line)`` from a .json document.

    Top-level arrays are decoded element by element so each element keeps its
    real source line; a single object yields one record pinned to line 1.
    """
    decoder = json.JSONDecoder()
    stripped = text.lstrip()
    base = len(text) - len(stripped)
    if not stripped.startswith("["):
        yield json.loads(text), 1
        return
    pos = base + 1
    length = len(text)
    while pos < length:
        while pos < length and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= length or text[pos] == "]":
            return
        line = _line_of(text, pos)
        obj, end = decoder.raw_decode(text, pos)
        yield obj, line
        pos = end


def _iter_ndjson_records(text, source):
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            yield json.loads(raw), line_no
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"invalid JSON in {source}: {exc.msg}",
                line=line_no,
            ) from exc


def load_records(path):
    """Load a declaration file as ``(record, source_line)`` pairs."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    source = os.path.basename(path)
    if path.endswith(".ndjson"):
        return list(_iter_ndjson_records(text, source))
    if path.endswith(".json"):
        return list(_iter_json_records(text, source))
    raise SpecError(f"unsupported declaration extension: {source}")


def _is_int(value):
    # bool is a subclass of int but is not a valid integer declaration.
    return isinstance(value, int) and not isinstance(value, bool)


def _require(cond, message, field, line):
    if not cond:
        raise SpecError(message, field=field, line=line)


@dataclass(frozen=True)
class Worker:
    worker_id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    slot_cap: int

    def to_dict(self):
        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "port": self.port,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "slot_cap": self.slot_cap,
        }


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival: int
    worker_id: str

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "arrival": self.arrival,
            "worker_id": self.worker_id,
        }


@dataclass(frozen=True)
class LeasePolicy:
    lease_duration_ms: int
    global_max_active: int

    def to_dict(self):
        return {
            "lease_duration_ms": self.lease_duration_ms,
            "global_max_active": self.global_max_active,
        }


@dataclass(frozen=True)
class MergeRules:
    default_param_version: int

    def to_dict(self):
        return {"default_param_version": self.default_param_version}


@dataclass
class Spec:
    workers: list
    requests: list
    lease_policy: LeasePolicy
    merge_rules: MergeRules
    worker_by_id: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "workers": [w.to_dict() for w in self.workers],
            "requests": [r.to_dict() for r in self.requests],
            "lease_policy": self.lease_policy.to_dict(),
            "merge_rules": self.merge_rules.to_dict(),
        }


def _validate_workers(records):
    workers = []
    seen_ids = set()
    seen_ports = set()
    for raw, line in records:
        _require(isinstance(raw, dict), "worker record must be an object",
                 "<record>", line)
        worker_id = raw.get("worker_id")
        _require(isinstance(worker_id, str) and worker_id,
                 "worker_id must be a non-empty string", "worker_id", line)
        _require(worker_id not in seen_ids,
                 f"duplicate worker_id {worker_id!r}", "worker_id", line)
        host = raw.get("host")
        _require(host == LOOPBACK_HOST,
                 f"host must be the literal {LOOPBACK_HOST!r}", "host", line)
        port = raw.get("port")
        _require(_is_int(port) and 1 <= port <= 65535,
                 "port must be an integer in [1, 65535]", "port", line)
        _require(port not in seen_ports,
                 f"duplicate loopback port {port}", "port", line)
        model_dim = raw.get("model_dim")
        _require(_is_int(model_dim) and model_dim > 0,
                 "model_dim must be a positive integer", "model_dim", line)
        hidden_dim = raw.get("hidden_dim")
        _require(_is_int(hidden_dim) and hidden_dim > 0,
                 "hidden_dim must be a positive integer", "hidden_dim", line)
        slot_cap = raw.get("slot_cap")
        _require(_is_int(slot_cap) and slot_cap >= 0,
                 "slot_cap must be a non-negative integer", "slot_cap", line)
        seen_ids.add(worker_id)
        seen_ports.add(port)
        workers.append(
            Worker(worker_id, host, port, model_dim, hidden_dim, slot_cap)
        )
    _require(workers, "at least one worker is required", "workers", None)
    return workers


def _validate_requests(records, known_workers):
    requests = []
    seen_ids = set()
    seen_arrivals = set()
    for raw, line in records:
        _require(isinstance(raw, dict), "request record must be an object",
                 "<record>", line)
        request_id = raw.get("request_id")
        _require(isinstance(request_id, str) and request_id,
                 "request_id must be a non-empty string", "request_id", line)
        _require(request_id not in seen_ids,
                 f"duplicate request_id {request_id!r}", "request_id", line)
        arrival = raw.get("arrival")
        _require(_is_int(arrival) and arrival >= 0,
                 "arrival must be a non-negative integer", "arrival", line)
        _require(arrival not in seen_arrivals,
                 f"duplicate arrival {arrival}", "arrival", line)
        worker_id = raw.get("worker_id")
        _require(isinstance(worker_id, str) and worker_id,
                 "worker_id must be a non-empty string", "worker_id", line)
        _require(worker_id in known_workers,
                 f"unknown worker_id {worker_id!r}", "worker_id", line)
        seen_ids.add(request_id)
        seen_arrivals.add(arrival)
        requests.append(Request(request_id, arrival, worker_id))
    _require(requests, "at least one request is required", "requests", None)
    return requests


def _validate_policy(records):
    _require(len(records) == 1, "exactly one lease policy is required",
             "lease_policy", records[0][1] if records else None)
    raw, line = records[0]
    duration = raw.get("lease_duration_ms")
    _require(_is_int(duration) and duration >= 0,
             "lease_duration_ms must be a non-negative integer",
             "lease_duration_ms", line)
    global_max = raw.get("global_max_active")
    _require(_is_int(global_max) and global_max >= 0,
             "global_max_active must be a non-negative integer",
             "global_max_active", line)
    return LeasePolicy(duration, global_max)


def _validate_merge(records):
    _require(len(records) == 1, "exactly one merge rule set is required",
             "merge_rules", records[0][1] if records else None)
    raw, line = records[0]
    default_version = raw.get("default_param_version")
    _require(_is_int(default_version) and default_version >= 0,
             "default_param_version must be a non-negative integer",
             "default_param_version", line)
    return MergeRules(default_version)


def load_spec(workers_path, requests_path, policy_path, merge_path):
    """Load and cross-validate the four declaration files into a Spec."""
    workers = _validate_workers(load_records(workers_path))
    worker_by_id = {w.worker_id: w for w in workers}
    requests = _validate_requests(load_records(requests_path), worker_by_id)
    policy = _validate_policy(load_records(policy_path))
    merge = _validate_merge(load_records(merge_path))
    return Spec(workers, requests, policy, merge, worker_by_id)


def load_spec_dir(directory):
    base = os.path.abspath(directory)
    return load_spec(
        os.path.join(base, "workers.json"),
        os.path.join(base, "requests.ndjson"),
        os.path.join(base, "lease_policy.json"),
        os.path.join(base, "merge_rules.json"),
    )
