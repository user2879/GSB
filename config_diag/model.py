from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FormulaTerm:
    ref: str
    weight: int

    def to_obj(self) -> dict[str, Any]:
        return {"ref": self.ref, "weight": self.weight}


@dataclass(frozen=True)
class Formula:
    op: str
    terms: tuple[FormulaTerm, ...] = ()
    divisor: int = 1
    bias: int = 0

    @property
    def refs(self) -> tuple[str, ...]:
        return tuple(term.ref for term in self.terms)

    def to_obj(self) -> dict[str, Any]:
        if self.op == "identity":
            return {"op": "identity"}
        return {
            "op": self.op,
            "terms": [term.to_obj() for term in sorted(self.terms, key=lambda item: item.ref)],
            "divisor": self.divisor,
            "bias": self.bias,
        }


@dataclass(frozen=True)
class KeySpec:
    key: str
    value: int
    version: int
    ttl_ms: int
    refs: tuple[str, ...]
    formula: Formula

    def to_obj(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "version": self.version,
            "ttl_ms": self.ttl_ms,
            "refs": list(sorted(self.refs)),
            "formula": self.formula.to_obj(),
        }


@dataclass(frozen=True)
class ConfigSpec:
    schema_version: int
    derivation_version: int
    keys: tuple[KeySpec, ...]

    @property
    def by_key(self) -> dict[str, KeySpec]:
        return {item.key: item for item in self.keys}

    @property
    def refs_by_key(self) -> dict[str, tuple[str, ...]]:
        return {item.key: item.refs for item in self.keys}

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "derivation_version": self.derivation_version,
            "keys": [item.to_obj() for item in self.keys],
        }


@dataclass(frozen=True)
class SubscriptionClient:
    client_id: str
    seed: int
    poll_interval_ms: int
    keys: tuple[str, ...]
    sample_size: int

    def to_obj(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "seed": self.seed,
            "poll_interval_ms": self.poll_interval_ms,
            "keys": list(self.keys),
            "sample_size": self.sample_size,
        }


@dataclass(frozen=True)
class SubscriptionSample:
    schema_version: int
    clients: tuple[SubscriptionClient, ...]

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "clients": [item.to_obj() for item in self.clients],
        }


@dataclass(frozen=True)
class ChangeEdit:
    key: str
    version: int
    value: int | None = None
    ttl_ms: int | None = None
    refs: tuple[str, ...] | None = None
    formula: Formula | None = None

    def to_obj(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"key": self.key, "version": self.version}
        if self.value is not None:
            payload["value"] = self.value
        if self.ttl_ms is not None:
            payload["ttl_ms"] = self.ttl_ms
        if self.refs is not None:
            payload["refs"] = list(sorted(self.refs))
        if self.formula is not None:
            payload["formula"] = self.formula.to_obj()
        return payload

    def changed_fields(self) -> list[str]:
        fields = ["version"]
        if self.value is not None:
            fields.append("value")
        if self.ttl_ms is not None:
            fields.append("ttl_ms")
        if self.refs is not None:
            fields.append("refs")
        if self.formula is not None:
            fields.append("formula")
        return sorted(fields)


@dataclass(frozen=True)
class ChangeOperation:
    operation_id: str
    sequence: int
    description: str
    changes: tuple[ChangeEdit, ...]

    def to_obj(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "sequence": self.sequence,
            "description": self.description,
            "changes": [item.to_obj() for item in self.changes],
        }


@dataclass(frozen=True)
class ChangeSequence:
    schema_version: int
    operations: tuple[ChangeOperation, ...]

    def to_obj(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operations": [item.to_obj() for item in self.operations],
        }
