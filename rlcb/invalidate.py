"""Parameter evolution, version bumping and invalidation propagation.

The dependency chain is explicit and one-directional::

    service params -> breaker windows -> leases -> batches -> requests

A change to rate limiting, breaker threshold, degrade level or window length
bumps that service's config version and marks every window, lease and batch
referencing the *old* parameter digest as invalid. Requests of unaffected
services/batches keep byte-identical conclusion summaries and lease ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hashing import sha256_json
from .spec import ServiceSpec

PARAM_FIELDS = (
    "rps_limit",
    "concurrency_limit",
    "failure_threshold_bp",
    "window_length",
    "degrade_level",
    "recovery_windows",
)


def param_digest(spec: ServiceSpec) -> str:
    return sha256_json(spec.param_digest_input())


@dataclass
class ChangeSet:
    service_id: str
    changes: dict[str, Any]
    from_version: int
    to_version: int
    old_digest: str
    new_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "changes": self.changes,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "old_digest": self.old_digest,
            "new_digest": self.new_digest,
        }


@dataclass
class InvalidationTrace:
    changes: list[ChangeSet] = field(default_factory=list)
    invalid_windows: list[str] = field(default_factory=list)
    invalid_leases: list[str] = field(default_factory=list)
    invalid_batches: list[str] = field(default_factory=list)
    affected_requests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": [c.to_dict() for c in self.changes],
            "invalid_windows": self.invalid_windows,
            "invalid_leases": self.invalid_leases,
            "invalid_batches": self.invalid_batches,
            "affected_requests": self.affected_requests,
        }


def apply_changes(services: list[ServiceSpec],
                  edits: dict[str, dict[str, Any]]) -> tuple[list[ServiceSpec],
                                                             list[ChangeSet]]:
    """Return new immutable service specs with monotonic version bumps."""
    by_id = {s.service_id: s for s in services}
    change_sets: list[ChangeSet] = []
    for service_id in sorted(edits):
        if service_id not in by_id:
            raise KeyError(f"unknown service {service_id}")
        spec = by_id[service_id]
        changes = edits[service_id]
        unknown = sorted(set(changes) - set(PARAM_FIELDS))
        if unknown:
            raise ValueError(f"unsupported parameter fields {unknown}")
        old_digest = param_digest(spec)
        updated_kwargs = {
            "service_id": spec.service_id, "host": spec.host,
            "port": spec.port, "seed": spec.seed,
            "config_version": spec.config_version + 1,
        }
        for name in ("rps_limit", "concurrency_limit",
                     "failure_threshold_bp", "window_length",
                     "degrade_level", "recovery_windows"):
            updated_kwargs[name] = changes.get(name, getattr(spec, name))
        new_spec = ServiceSpec(**updated_kwargs)
        new_digest = param_digest(new_spec)
        by_id[service_id] = new_spec
        change_sets.append(ChangeSet(
            service_id=service_id, changes=dict(sorted(changes.items())),
            from_version=spec.config_version,
            to_version=new_spec.config_version,
            old_digest=old_digest, new_digest=new_digest,
        ))
    return [by_id[s.service_id] for s in services], change_sets


def propagate(services: list[ServiceSpec],
              change_sets: list[ChangeSet],
              requests: list[Any],
              lease_records: list[dict[str, Any]]) -> InvalidationTrace:
    trace = InvalidationTrace(changes=list(change_sets))
    changed = {c.service_id for c in change_sets}
    window_lengths = {s.service_id: s.window_length for s in services}
    for request in sorted(requests, key=lambda r: (r.service_id, r.arrival_seq,
                                                   r.request_id)):
        if request.service_id not in changed:
            continue
        win = request.arrival_seq // window_lengths[request.service_id]
        trace.invalid_windows.append(
            f"{request.service_id}:window{win}")
        trace.invalid_batches.append(
            f"{request.service_id}:{request.batch_id}")
        trace.affected_requests.append(request.request_id)
    for lease in lease_records:
        if lease.get("service_id") in changed:
            trace.invalid_leases.append(lease.get("lease_id"))
    # De-duplicate while preserving canonical sorted order for byte stability.
    trace.invalid_windows = sorted(set(trace.invalid_windows))
    trace.invalid_batches = sorted(set(trace.invalid_batches))
    trace.invalid_leases = sorted(set(trace.invalid_leases))
    trace.affected_requests = sorted(set(trace.affected_requests))
    return trace
