"""Stable-hash gray routing and order-independent hit attribution."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .constants import BUCKETS, POOL_GRAY, POOL_STABLE, WEIGHT_MAX
from .spec import Request, RuleRevision, Spec, Backend


def stable_bucket(client_key: str, rule_id: str) -> int:
    """Map (client_key, rule_id) deterministically onto [0, 9999].

    The digest is computed over an explicit canonical key string so the result
    never depends on dict or hash iteration order.
    """
    material = f"rule={rule_id}|client={client_key}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest, 16) % BUCKETS


def route_pool(client_key: str, rule_id: str,
               weight_permyriad: int) -> str:
    """Return ``stable`` or ``gray`` for one request under one weight."""
    bucket = stable_bucket(client_key, rule_id)
    return POOL_GRAY if bucket < weight_permyriad else POOL_STABLE


def stable_worker(client_key: str, rule_id: str, role: str,
                  candidates: tuple[str, ...]) -> str:
    """Pick one worker of the target pool with a second independent hash."""
    ordered = tuple(sorted(candidates))
    material = f"worker|rule={rule_id}|role={role}|client={client_key}" \
        .encode("utf-8")
    index = int(hashlib.sha256(material).hexdigest(), 16) % len(ordered)
    return ordered[index]


@dataclass(frozen=True)
class Hit:
    request_id: str
    rule_id: str
    revision: int
    bucket: int
    pool: str
    worker_id: str
    backend_id: str
    param_version_seq: int
    param_summary: str
    weight_permyriad: int

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "rule_id": self.rule_id,
            "revision": self.revision,
            "bucket": self.bucket,
            "pool": self.pool,
            "worker_id": self.worker_id,
            "backend_id": self.backend_id,
            "param_version_seq": self.param_version_seq,
            "param_summary": self.param_summary,
            "weight_permyriad": self.weight_permyriad,
        }


def attribute(spec: Spec, request: Request,
              revision: RuleRevision) -> Hit:
    """Compute the full hit attribution for one request under a revision."""
    bucket = stable_bucket(request.client_key, request.rule_id)
    pool = POOL_GRAY if bucket < revision.weight_permyriad else POOL_STABLE
    candidates = tuple(
        bid for bid in revision.backend_ids
        if spec.backends[bid].role == pool
    )
    if not candidates:  # weight 0 -> no gray traffic; weight max -> no stable
        pool = POOL_STABLE
        candidates = tuple(
            bid for bid in revision.backend_ids
            if spec.backends[bid].role == pool
        )
    backend_id = stable_worker(request.client_key, request.rule_id,
                               pool, candidates)
    worker_id = f"worker-{spec.backends[backend_id].backend_id}"
    return Hit(
        request_id=request.request_id,
        rule_id=request.rule_id,
        revision=revision.revision,
        bucket=bucket,
        pool=pool,
        worker_id=worker_id,
        backend_id=backend_id,
        param_version_seq=revision.param_version_seq,
        param_summary=revision.param_summary,
        weight_permyriad=revision.weight_permyriad,
    )


def attribute_all(spec: Spec) -> dict[str, Hit]:
    """Attribute every request; keyed by request id, order independent."""
    return {
        req.request_id: attribute(spec, req, spec.revision_for(req))
        for req in sorted(spec.requests, key=lambda r: r.arrival_seq)
    }
