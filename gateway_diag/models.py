from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .util import canonical_json, digest_value

SUPPORTED_QUANTIZATION = frozenset({"fp32", "fp16", "int8"})
TARGET_SELECTS = frozenset({"conclusion", "stable_target", "canary_target"})


@dataclass(frozen=True)
class Backend:
    id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    quantization: str

    @property
    def param_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "quantization": self.quantization,
        }

    @property
    def param_digest(self) -> str:
        return digest_value(self.param_payload)

    @property
    def connection_payload(self) -> dict[str, Any]:
        return {"id": self.id, "host": self.host, "port": self.port}

    @property
    def declaration_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            **{key: value for key, value in self.param_payload.items() if key != "id"},
        }


@dataclass(frozen=True)
class TargetRef:
    kind: str
    backend_id: str | None = None
    from_rule: str | None = None
    select: str = "conclusion"

    def __post_init__(self) -> None:
        if self.kind not in {"backend", "derived"}:
            raise ValueError(f"unknown target kind {self.kind!r}")
        if self.kind == "backend" and not self.backend_id:
            raise ValueError("backend target requires backend_id")
        if self.kind == "derived" and not self.from_rule:
            raise ValueError("derived target requires from_rule")

    @property
    def derived_rule_id(self) -> str | None:
        return self.from_rule if self.kind == "derived" else None

    @property
    def direct_backend_id(self) -> str | None:
        return self.backend_id if self.kind == "backend" else None

    def payload(self) -> dict[str, Any]:
        if self.kind == "backend":
            return {"kind": "backend", "backend_id": self.backend_id}
        return {
            "kind": "derived",
            "from_rule": self.from_rule,
            "select": self.select,
        }


@dataclass(frozen=True)
class Rule:
    id: str
    request_prefix: str
    stable_backend: str
    canary_target: TargetRef
    canary_weight_bp: int
    batch_size: int
    version: int
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dependency_ids(self) -> tuple[str, ...]:
        values = set(self.depends_on)
        if self.canary_target.derived_rule_id:
            values.add(self.canary_target.derived_rule_id)
        return tuple(sorted(values))

    @property
    def direct_backend_ids(self) -> tuple[str, ...]:
        values = {self.stable_backend}
        if self.canary_target.direct_backend_id:
            values.add(self.canary_target.direct_backend_id)
        return tuple(sorted(values))

    @property
    def declaration_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_prefix": self.request_prefix,
            "stable_backend": self.stable_backend,
            "canary_target": self.canary_target.payload(),
            "canary_weight_bp": self.canary_weight_bp,
            "batch_size": self.batch_size,
            "version": self.version,
            "depends_on": list(self.depends_on),
        }

    def comparable_payload(self) -> dict[str, Any]:
        return {
            "request_prefix": self.request_prefix,
            "stable_backend": self.stable_backend,
            "canary_target": self.canary_target.payload(),
            "canary_weight_bp": self.canary_weight_bp,
            "batch_size": self.batch_size,
            "version": self.version,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_seq: int
    client_key: str
    prefix: str
    vector: tuple[float, ...]
    rule_id: str | None = None

    @property
    def payload(self) -> list[Any]:
        return [
            self.request_id,
            self.arrival_seq,
            self.client_key,
            self.prefix,
            list(self.vector),
            self.rule_id,
        ]


@dataclass(frozen=True)
class Spec:
    batch_limit: int
    backends: dict[str, Backend]
    rules: dict[str, Rule]
    requests: tuple[Request, ...]

    @property
    def sorted_backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.backends))

    @property
    def sorted_rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.rules))

    def declaration_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "batch_limit": self.batch_limit,
            "backends": [self.backends[key].declaration_payload for key in self.sorted_backend_ids],
            "rules": [self.rules[key].declaration_payload for key in self.sorted_rule_ids],
        }

    @property
    def declaration_digest(self) -> str:
        return digest_value(self.declaration_payload())

    @property
    def rule_versions(self) -> dict[str, int]:
        return {key: self.rules[key].version for key in self.sorted_rule_ids}

    @property
    def backend_versions(self) -> dict[str, str]:
        return {key: self.backends[key].param_digest for key in self.sorted_backend_ids}

    @property
    def backend_set_digest(self) -> str:
        payload = [self.backends[key].declaration_payload for key in self.sorted_backend_ids]
        return digest_value(payload)

    @property
    def request_digest(self) -> str:
        ordered = sorted(self.requests, key=lambda item: (item.arrival_seq, item.request_id))
        return digest_value([request.payload for request in ordered])

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def request_by_id(self) -> dict[str, Request]:
        return {request.request_id: request for request in self.requests}

    def canonical_declaration_json(self) -> str:
        return canonical_json(self.declaration_payload())

    def rule_receipt_set_digest(self, receipt_digests: Iterable[str]) -> str:
        return digest_value(list(receipt_digests))


def request_payload_digest(requests: Iterable[Request]) -> str:
    """Digest used inside a receipt (request IDs remain the primary binding)."""
    return digest_value([request.payload for request in requests])


def batch_output_material(
    *, rule_id: str, batch_index: int, destination: str, backend_param_digest: str, request_ids: list[str]
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "batch_index": batch_index,
        "destination": destination,
        "backend_param_digest": backend_param_digest,
        "request_ids": sorted(request_ids),
    }
