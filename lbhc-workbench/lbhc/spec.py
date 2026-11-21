"""Cluster declaration loading and strong validation.

Any violation rejects the whole declaration with E_SPEC and a JSON-ish
field path such as ``pools[0].backends[2].port``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from .errors import SpecError

PROTOCOLS = ("tcp", "http")
ROLES = ("active", "draining")
BEHAVIORS = ("ok", "mismatch", "delay", "refused")  # simulator-only knob
LOOPBACK = "127.0.0.1"


@dataclass(frozen=True)
class Backend:
    id: str
    host: str
    port: int
    weight: int
    role: str
    behavior: str  # simulator injection knob: ok|mismatch|delay|refused


@dataclass(frozen=True)
class Pool:
    id: str
    protocol: str
    timeout_ms: int
    interval_ms: int
    success_threshold: int
    failure_threshold: int
    concurrency_quota: int
    expect_body: str
    backends: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class Spec:
    version: int
    concurrency_budget: int
    rate_limit_per_sec: int
    pools: tuple

    @property
    def instance_count(self):
        return sum(len(p.backends) for p in self.pools)

    @property
    def pool_count(self):
        return len(self.pools)

    def backend(self, instance_id):
        for p in self.pools:
            for b in p.backends:
                if b.id == instance_id:
                    return p, b
        raise KeyError(instance_id)


def _is_int(v):
    return type(v) is int  # bool is a subclass of int; reject it


def _need_int(node, key, path, minimum):
    if key not in node:
        raise SpecError(f"missing required field {key!r}", path=path)
    v = node[key]
    if not _is_int(v) or v < minimum:
        raise SpecError(f"must be an integer >= {minimum}, got {v!r}",
                        path=path)
    return v


def _need_str(node, key, path, allowed=None):
    if key not in node:
        raise SpecError(f"missing required field {key!r}", path=path)
    v = node[key]
    if not isinstance(v, str) or not v:
        raise SpecError(f"must be a non-empty string, got {v!r}", path=path)
    if allowed is not None and v not in allowed:
        raise SpecError(f"must be one of {allowed}, got {v!r}", path=path)
    return v


def load_spec(path):
    """Load and strongly validate a cluster declaration JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        raise SpecError(f"cannot read spec file: {exc}", path="$")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc}", path="$")
    if not isinstance(doc, dict):
        raise SpecError("top level must be a JSON object", path="$")

    version = _need_int(doc, "version", "version", 1)

    g = doc.get("global")
    if not isinstance(g, dict):
        raise SpecError("missing required object 'global'", path="global")
    budget = _need_int(g, "concurrency_budget",
                       "global.concurrency_budget", 1)
    rate = _need_int(g, "rate_limit_per_sec",
                     "global.rate_limit_per_sec", 1)

    pools_raw = doc.get("pools")
    if not isinstance(pools_raw, list) or not pools_raw:
        raise SpecError("must be a non-empty array", path="pools")

    seen_pools = set()
    seen_instances = set()
    seen_ports = {}
    pools = []
    for pi, praw in enumerate(pools_raw):
        ppath = f"pools[{pi}]"
        if not isinstance(praw, dict):
            raise SpecError("pool must be an object", path=ppath)
        pid = _need_str(praw, "id", f"{ppath}.id")
        if pid in seen_pools:
            raise SpecError(f"duplicate pool id {pid!r}", path=f"{ppath}.id")
        seen_pools.add(pid)
        protocol = _need_str(praw, "protocol", f"{ppath}.protocol", PROTOCOLS)
        timeout_ms = _need_int(praw, "timeout_ms", f"{ppath}.timeout_ms", 1)
        interval_ms = _need_int(praw, "interval_ms",
                                f"{ppath}.interval_ms", 1)
        succ = _need_int(praw, "success_threshold",
                         f"{ppath}.success_threshold", 1)
        fail = _need_int(praw, "failure_threshold",
                         f"{ppath}.failure_threshold", 1)
        quota = _need_int(praw, "concurrency_quota",
                          f"{ppath}.concurrency_quota", 1)
        expect = praw.get("expect_body", "OK")
        if not isinstance(expect, str) or not expect:
            raise SpecError("must be a non-empty string",
                            path=f"{ppath}.expect_body")

        braw = praw.get("backends")
        if not isinstance(braw, list) or not braw:
            raise SpecError("must be a non-empty array",
                            path=f"{ppath}.backends")
        backends = []
        for bi, br in enumerate(braw):
            bpath = f"{ppath}.backends[{bi}]"
            if not isinstance(br, dict):
                raise SpecError("backend must be an object", path=bpath)
            bid = _need_str(br, "id", f"{bpath}.id")
            if bid in seen_instances:
                raise SpecError(f"duplicate instance id {bid!r}",
                                path=f"{bpath}.id")
            seen_instances.add(bid)
            host = _need_str(br, "host", f"{bpath}.host")
            if host != LOOPBACK:
                raise SpecError(
                    f"host must be the loopback literal {LOOPBACK!r}, "
                    f"got {host!r}", path=f"{bpath}.host")
            port = _need_int(br, "port", f"{bpath}.port", 1024)
            if port > 65535:
                raise SpecError("port must be <= 65535",
                                path=f"{bpath}.port")
            if port in seen_ports:
                raise SpecError(
                    f"port {port} already used by instance "
                    f"{seen_ports[port]!r}; ports must be globally unique",
                    path=f"{bpath}.port")
            seen_ports[port] = bid
            weight = _need_int(br, "weight", f"{bpath}.weight", 0)
            role = _need_str(br, "role", f"{bpath}.role", ROLES)
            behavior = _need_str(br, "behavior", f"{bpath}.behavior",
                                 BEHAVIORS)
            backends.append(Backend(bid, host, port, weight, role, behavior))
        pools.append(Pool(
            id=pid, protocol=protocol, timeout_ms=timeout_ms,
            interval_ms=interval_ms, success_threshold=succ,
            failure_threshold=fail, concurrency_quota=quota,
            expect_body=expect, backends=tuple(backends)))
    return Spec(version=version, concurrency_budget=budget,
                rate_limit_per_sec=rate, pools=tuple(pools))


def spec_to_dict(spec):
    return {
        "version": spec.version,
        "global": {
            "concurrency_budget": spec.concurrency_budget,
            "rate_limit_per_sec": spec.rate_limit_per_sec,
        },
        "pools": [
            {
                "id": p.id,
                "protocol": p.protocol,
                "timeout_ms": p.timeout_ms,
                "interval_ms": p.interval_ms,
                "success_threshold": p.success_threshold,
                "failure_threshold": p.failure_threshold,
                "concurrency_quota": p.concurrency_quota,
                "expect_body": p.expect_body,
                "backends": [
                    {"id": b.id, "host": b.host, "port": b.port,
                     "weight": b.weight, "role": b.role,
                     "behavior": b.behavior}
                    for b in p.backends
                ],
            }
            for p in spec.pools
        ],
    }


def spec_digest(spec):
    """Stable digest of the validated declaration (for ledger tagging)."""
    payload = json.dumps(spec_to_dict(spec), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def write_spec(spec_dict, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec_dict, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
