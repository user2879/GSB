"""Compile a compiled version into a node-oriented distribution plan."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..util.canonical import canonical_bytes, digest
from .compiler import CompiledVersion
from .manifest import ManifestVersion


@dataclass(frozen=True)
class PlanRequest:
    request_id: str
    node_id: str
    version: int
    keys: tuple[str, ...]
    depends_on: tuple[str, ...]          # request ids that must succeed first
    slice_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "version": self.version,
            "keys": list(self.keys),
            "depends_on": list(self.depends_on),
            "slice_digest": digest(self.slice_payload),
        }


@dataclass
class DistributionPlan:
    version: int
    strategy: str
    grid: dict[str, int]
    requests: list[PlanRequest]
    content_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "strategy": self.strategy,
            "grid": self.grid,
            "requests": [r.to_dict() for r in self.requests],
            "content_digest": self.content_digest,
        }


def build_plan(
    manifest_version: ManifestVersion,
    compiled: CompiledVersion,
    grid_q: int,
) -> DistributionPlan:
    # node -> ordered item keys that this node must receive.
    node_keys: dict[str, list[str]] = {}
    for key in manifest_version.order:
        item = manifest_version.items[key]
        for node_id in item.targets:
            node_keys.setdefault(node_id, []).append(key)

    requests: list[PlanRequest] = []
    request_id_by_node: dict[str, str] = {}
    for node_id in sorted(node_keys):
        keys = tuple(sorted(node_keys[node_id]))
        request_id = f"r{manifest_version.version}-{node_id}"
        request_id_by_node[node_id] = request_id
        slice_payload = {
            "version": manifest_version.version,
            "node_id": node_id,
            "grid_q": grid_q,
            "items": [
                {
                    "key": key,
                    "mantissa": compiled.values[key].mantissa,
                    "value_exact": compiled.items[key].value_exact,
                    "targets": list(compiled.items[key].targets),
                    "derivation": manifest_version.items[key].derivation.to_dict()
                    if manifest_version.items[key].derivation is not None
                    else None,
                }
                for key in keys
            ],
        }
        requests.append(
            PlanRequest(
                request_id=request_id,
                node_id=node_id,
                version=manifest_version.version,
                keys=keys,
                depends_on=(),
                slice_payload=slice_payload,
            )
        )

    if manifest_version.strategy == "shard":
        # Shard mode: a node receives only the items that explicitly target it.
        # A distribution dependency A -> B exists when B holds an upstream item
        # of an item that A derives locally.  This follows the derivation DAG,
        # so the resulting request graph is always acyclic.
        node_membership = {node_id: set(keys) for node_id, keys in node_keys.items()}
        for request in requests:
            own = node_membership[request.node_id]
            required_upstreams: set[str] = set()
            for key in own:
                item = manifest_version.items[key]
                if item.derivation is not None:
                    required_upstreams.update(item.derivation.coefficients)
            deps: list[str] = []
            for other in requests:
                if other.node_id == request.node_id:
                    continue
                if required_upstreams & node_membership[other.node_id]:
                    deps.append(other.request_id)
            object.__setattr__(request, "depends_on", tuple(sorted(set(deps))))
    # "full" mode keeps empty dependency lists: every node gets the full slice
    # independently but in one bounded parallel wave.

    plan_digest = digest(
        {
            "version": manifest_version.version,
            "strategy": manifest_version.strategy,
            "grid_q": grid_q,
            "requests": [r.to_dict() for r in requests],
        }
    )
    return DistributionPlan(
        version=manifest_version.version,
        strategy=manifest_version.strategy,
        grid={"q": grid_q, "scale": 10**grid_q},
        requests=requests,
        content_digest=plan_digest,
    )
