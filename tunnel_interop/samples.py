"""Deterministic inner-payload and message-id derivation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .spec import Endpoint, Route, TunnelSpec
from .util import canonical_json, sha256_hex, sha256_json


SAMPLE_EPOCH_MS = 1_700_000_000_000
SAMPLE_DAY_MS = 86_400_000


@dataclass(frozen=True)
class MessagePlan:
    message_key: str
    route: Route
    endpoint: Endpoint
    payload: bytes
    payload_sha256: str
    message_id: str
    message_id_basis: dict[str, Any]
    rng_seed: int
    rng_seed_basis: str
    generated_at_ms: int
    fragment_size: int

    @property
    def tunnel_id(self) -> str:
        return self.route.tunnel_id

    @property
    def destination_prefix(self) -> str:
        return self.route.prefix

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        import base64

        result = {
            "message_key": self.message_key,
            "tunnel_id": self.tunnel_id,
            "destination_prefix": self.destination_prefix,
            "next_hop": self.route.next_hop,
            "priority": self.route.priority,
            "endpoint_id": self.endpoint.endpoint_id,
            "fragment_size": self.fragment_size,
            "payload_length": len(self.payload),
            "payload_sha256": self.payload_sha256,
            "message_id": self.message_id,
            "message_id_basis": self.message_id_basis,
            "rng_seed": self.rng_seed,
            "rng_seed_hex": self.rng_seed.to_bytes(16, "big").hex(),
            "rng_seed_basis": self.rng_seed_basis,
            "generated_at_ms": self.generated_at_ms,
        }
        if include_payload:
            result["payload_b64"] = base64.b64encode(self.payload).decode("ascii")
        return result


def _route_sort_key(route: Route) -> tuple[Any, ...]:
    network = route.network
    return (
        network.version,
        int(network.network_address),
        network.prefixlen,
        route.tunnel_id,
        route.next_hop,
        route.priority,
        route.prefix,
    )


def ordered_routes(spec: TunnelSpec) -> tuple[Route, ...]:
    return tuple(sorted(spec.routes, key=_route_sort_key))


def derive_rng_seed(endpoint: Endpoint, route: Route) -> tuple[int, str]:
    material = canonical_json(
        {
            "domain": "tunnel-interop/sample-seed/v1",
            "endpoint_id": endpoint.endpoint_id,
            "endpoint_seed": endpoint.seed,
            "tunnel_id": route.tunnel_id,
            "destination_prefix": route.prefix,
        }
    )
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:16], "big"), digest.hex()


def generate_payload(seed: int, length: int) -> bytes:
    """PCG64 stream, explicitly pinned to the declared RNG seed and length."""

    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()


def derive_message_id(
    *,
    route: Route,
    payload_sha256: str,
    payload_length: int,
    fragment_size: int,
) -> tuple[str, dict[str, Any]]:
    basis = {
        "domain": "tunnel-interop/message-id/v1",
        "tunnel_id": route.tunnel_id,
        "destination_prefix": route.prefix,
        "next_hop": route.next_hop,
        "priority": route.priority,
        "payload_sha256": payload_sha256,
        "payload_length": payload_length,
        "fragment_size": fragment_size,
    }
    return sha256_json(basis), basis


def derive_tunnel_basis(route: Route) -> dict[str, Any]:
    basis = {
        "domain": "tunnel-interop/tunnel-id-reference/v1",
        "declared_tunnel_id": route.tunnel_id,
        "destination_prefix": route.prefix,
        "next_hop": route.next_hop,
        "priority": route.priority,
    }
    return {
        **basis,
        "route_fingerprint": sha256_json(basis),
        "wire_value_source": "route.tunnel_id",
    }


def generate_messages(spec: TunnelSpec) -> tuple[MessagePlan, ...]:
    endpoints = spec.endpoint_map
    fragment_size = spec.fragment_policy.fragment_size
    sample_length = spec.samples.message_size(fragment_size)
    if sample_length > spec.runtime.max_message_bytes:
        from .errors import E_SPEC, fail

        fail(
            E_SPEC,
            "$.sample.bytes_per_message",
            "generated sample exceeds runtime.max_message_bytes",
            sample_length=sample_length,
            max_message_bytes=spec.runtime.max_message_bytes,
        )
    messages: list[MessagePlan] = []
    for route in ordered_routes(spec):
        endpoint = endpoints[route.next_hop]
        rng_seed, seed_basis_hash = derive_rng_seed(endpoint, route)
        payload = generate_payload(rng_seed, sample_length)
        payload_sha = sha256_hex(payload)
        message_id, message_id_basis = derive_message_id(
            route=route,
            payload_sha256=payload_sha,
            payload_length=len(payload),
            fragment_size=fragment_size,
        )
        generated_offset = int(hashlib.sha256(message_id.encode("ascii")).hexdigest()[:12], 16) % SAMPLE_DAY_MS
        generated_at_ms = SAMPLE_EPOCH_MS + generated_offset
        messages.append(
            MessagePlan(
                message_key=route.tunnel_id,
                route=route,
                endpoint=endpoint,
                payload=payload,
                payload_sha256=payload_sha,
                message_id=message_id,
                message_id_basis=message_id_basis,
                rng_seed=rng_seed,
                rng_seed_basis=f"sha256(sample-seed/v1 basis)={seed_basis_hash}",
                generated_at_ms=generated_at_ms,
                fragment_size=fragment_size,
            )
        )
    return tuple(messages)
