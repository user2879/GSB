"""Deterministic fragmentation, format migration and reassembly."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .errors import E_FIELD, fail
from .samples import MessagePlan, derive_tunnel_basis, generate_messages
from .spec import TunnelSpec
from .util import sha256_hex, sha256_json
from .wire import (
    NormalizedFragment,
    WireContext,
    decode_fragment,
    encode_fragment,
    migrate_fragment,
)


@dataclass(frozen=True)
class FragmentRecord:
    message_key: str
    fragment_index: int
    fragment_offset: int
    fragment_length: int
    total_fragments: int
    message_id: str
    tunnel_id: str
    tunnel_id_basis: Mapping[str, Any]
    target_format: str
    fragment_version: int
    normalized: NormalizedFragment
    wire: bytes
    wire_sha256: str
    format_variants: Mapping[str, bytes]
    semantic_signature: str

    def to_dict(self, *, include_wire: bool = False) -> dict[str, Any]:
        result = {
            "message_key": self.message_key,
            "fragment_index": self.fragment_index,
            "fragment_offset": self.fragment_offset,
            "fragment_length": self.fragment_length,
            "total_fragments": self.total_fragments,
            "message_id": self.message_id,
            "tunnel_id": self.tunnel_id,
            "tunnel_id_basis": dict(self.tunnel_id_basis),
            "format": self.target_format,
            "fragment_version": self.fragment_version,
            "wire_length": len(self.wire),
            "wire_sha256": self.wire_sha256,
            "semantic_signature": self.semantic_signature,
            "normalized": self.normalized.to_dict(),
            "available_formats": sorted(self.format_variants),
        }
        if include_wire:
            result["wire_b64"] = base64.b64encode(self.wire).decode("ascii")
            result["format_variants_b64"] = {
                key: base64.b64encode(value).decode("ascii")
                for key, value in sorted(self.format_variants.items())
            }
        return result


@dataclass(frozen=True)
class ReassemblyResult:
    message_key: str
    message_id: str
    tunnel_id: str
    payload: bytes
    payload_sha256: str
    fragment_indices: tuple[int, ...]
    arrival_indices: tuple[int, ...]
    duplicate_indices: tuple[int, ...]
    reassembly_version: int
    complete: bool = True

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        result = {
            "message_key": self.message_key,
            "message_id": self.message_id,
            "tunnel_id": self.tunnel_id,
            "payload_length": len(self.payload),
            "payload_sha256": self.payload_sha256,
            "fragment_indices": list(self.fragment_indices),
            "arrival_indices": list(self.arrival_indices),
            "duplicate_indices": list(self.duplicate_indices),
            "reassembly_version": self.reassembly_version,
            "complete": self.complete,
        }
        if include_payload:
            result["payload_b64"] = base64.b64encode(self.payload).decode("ascii")
        return result


@dataclass(frozen=True)
class MessageBundle:
    plan: MessagePlan
    fragments: tuple[FragmentRecord, ...]
    reassembly: ReassemblyResult

    def to_dict(self, *, include_wire: bool = False, include_payload: bool = False) -> dict[str, Any]:
        return {
            "message": self.plan.to_dict(include_payload=include_payload),
            "fragments": [item.to_dict(include_wire=include_wire) for item in self.fragments],
            "reassembly": self.reassembly.to_dict(include_payload=include_payload),
        }


@dataclass(frozen=True)
class FragmentBundle:
    spec: TunnelSpec
    target_format: str
    messages: tuple[MessageBundle, ...]
    declaration_fingerprint: str

    @property
    def all_fragments(self) -> tuple[FragmentRecord, ...]:
        return tuple(fragment for message in self.messages for fragment in message.fragments)

    @property
    def message_map(self) -> dict[str, MessageBundle]:
        return {message.plan.message_key: message for message in self.messages}

    def to_dict(self, *, include_wire: bool = False, include_payload: bool = False) -> dict[str, Any]:
        return {
            "declaration_fingerprint": self.declaration_fingerprint,
            "target_format": self.target_format,
            "message_count": len(self.messages),
            "fragment_count": len(self.all_fragments),
            "messages": [
                message.to_dict(include_wire=include_wire, include_payload=include_payload)
                for message in self.messages
            ],
        }


def _wire_context(plan: MessagePlan, total_fragments: int) -> WireContext:
    return WireContext(
        fragment_size=plan.fragment_size,
        effective_payload_limit=plan.endpoint.effective_payload_limit,
        max_message_bytes=max(plan.endpoint.effective_payload_limit * total_fragments, len(plan.payload)),
        total_fragments=total_fragments,
        message_id=plan.message_id,
        generated_at_ms=plan.generated_at_ms,
        encapsulation_version=1,
        retry_count=0,
    )


def context_for_plan(plan: MessagePlan, total_fragments: int) -> WireContext:
    return _wire_context(plan, total_fragments)


def _normalized_template(
    plan: MessagePlan,
    *,
    fragment_index: int,
    fragment_offset: int,
    payload: bytes,
    total_fragments: int,
) -> NormalizedFragment:
    return NormalizedFragment(
        tunnel_id=plan.tunnel_id,
        fragment_index=fragment_index,
        fragment_offset=fragment_offset,
        payload=payload,
        total_fragments=total_fragments,
        message_id=plan.message_id,
        generated_at_ms=plan.generated_at_ms,
        checksum="00" * 32,
        encapsulation_version=1,
        fragment_header_digest="00" * 32,
        retry_count=0,
        source_format="f3",
        padded_fields=(),
        padding_basis={},
    )


def fragment_message(
    plan: MessagePlan,
    *,
    target_format: str,
    fragment_versions: Mapping[int, int] | None = None,
    reassembly_version: int = 1,
) -> MessageBundle:
    if target_format not in {"f1", "f2", "f3"}:
        fail(E_FIELD, "fragment.format", "unknown target format", actual=target_format)
    size = plan.fragment_size
    if size <= 0:
        fail(E_FIELD, "fragment_policy.fragment_size", "fragment size must be positive")
    payload = plan.payload
    total_fragments = max(1, (len(payload) + size - 1) // size)
    context = _wire_context(plan, total_fragments)
    tunnel_basis = derive_tunnel_basis(plan.route)
    records: list[FragmentRecord] = []
    for index in range(total_fragments):
        offset = index * size
        chunk = payload[offset : offset + size]
        template = _normalized_template(
            plan,
            fragment_index=index,
            fragment_offset=offset,
            payload=chunk,
            total_fragments=total_fragments,
        )
        variants: dict[str, bytes] = {}
        variant_norms: dict[str, NormalizedFragment] = {}
        for wire_format in ("f1", "f2", "f3"):
            variant_wire = encode_fragment(template, wire_format)
            variants[wire_format] = variant_wire
            variant_norms[wire_format] = decode_fragment(variant_wire, context)
        semantic_values = {sha256_json(item.semantic_key()) for item in variant_norms.values()}
        if len(semantic_values) != 1:
            fail(
                E_FIELD,
                f"fragments[{index}]",
                "format migration did not preserve normalized key fields",
            )
        normalized = variant_norms[target_format]
        wire = variants[target_format]
        version = fragment_versions.get(index, 1) if fragment_versions else 1
        records.append(
            FragmentRecord(
                message_key=plan.message_key,
                fragment_index=index,
                fragment_offset=offset,
                fragment_length=len(chunk),
                total_fragments=total_fragments,
                message_id=plan.message_id,
                tunnel_id=plan.tunnel_id,
                tunnel_id_basis=tunnel_basis,
                target_format=target_format,
                fragment_version=version,
                normalized=normalized,
                wire=wire,
                wire_sha256=sha256_hex(wire),
                format_variants=variants,
                semantic_signature=next(iter(semantic_values)),
            )
        )
    reassembly = reassemble_fragments(
        [item.normalized for item in records],
        expected_message_key=plan.message_key,
        expected_sha256=plan.payload_sha256,
        reassembly_version=reassembly_version,
    )
    return MessageBundle(plan=plan, fragments=tuple(records), reassembly=reassembly)


def build_bundle(
    spec: TunnelSpec,
    *,
    target_format: str | None = None,
    plans: Iterable[MessagePlan] | None = None,
    fragment_versions: Mapping[str, Mapping[int, int]] | None = None,
    reassembly_versions: Mapping[str, int] | None = None,
) -> FragmentBundle:
    wire_format = target_format or spec.runtime.fragment_format
    messages = []
    for plan in plans if plans is not None else generate_messages(spec):
        messages.append(
            fragment_message(
                plan,
                target_format=wire_format,
                fragment_versions=(fragment_versions or {}).get(plan.message_key),
                reassembly_version=(reassembly_versions or {}).get(plan.message_key, 1),
            )
        )
    return FragmentBundle(
        spec=spec,
        target_format=wire_format,
        messages=tuple(sorted(messages, key=lambda item: item.plan.message_key)),
        declaration_fingerprint=spec.declaration_fingerprint(),
    )


def reassemble_fragments(
    fragments: Iterable[NormalizedFragment],
    *,
    expected_message_key: str,
    expected_sha256: str | None = None,
    expected_total: int | None = None,
    reassembly_version: int = 1,
) -> ReassemblyResult:
    items = list(fragments)
    if not items:
        fail(E_FIELD, "reassembly.fragments", "no fragments were supplied")
    identities = {(item.tunnel_id, item.message_id) for item in items}
    if len(identities) != 1:
        fail(E_FIELD, "reassembly.message_id", "fragments span multiple message identities")
    tunnel_id, message_id = next(iter(identities))
    totals = {item.total_fragments for item in items}
    if len(totals) != 1:
        fail(E_FIELD, "reassembly.total_fragments", "fragments disagree on total fragment count")
    total = next(iter(totals))
    if expected_total is not None and total != expected_total:
        fail(E_FIELD, "reassembly.total_fragments", "total count differs from expectation", expected=expected_total, actual=total)

    by_offset: dict[int, NormalizedFragment] = {}
    duplicate_indices: list[int] = []
    arrival_indices: list[int] = []
    for item in items:
        arrival_indices.append(item.fragment_index)
        previous = by_offset.get(item.fragment_offset)
        if previous is not None:
            if previous.fragment_length != item.fragment_length or previous.payload != item.payload:
                fail(
                    E_FIELD,
                    f"reassembly.fragments[{item.fragment_index}]",
                    "duplicate fragment offset carries conflicting bytes",
                    fragment_index=item.fragment_index,
                    fragment_offset=item.fragment_offset,
                )
            duplicate_indices.append(item.fragment_index)
            continue
        by_offset[item.fragment_offset] = item

    ordered = sorted(by_offset.values(), key=lambda item: (item.fragment_offset, item.fragment_index))
    if len(ordered) != total:
        fail(
            E_FIELD,
            "reassembly.fragments",
            "fragment set is incomplete",
            expected=total,
            actual=len(ordered),
            missing_indices=[index for index in range(total) if index not in {item.fragment_index for item in ordered}],
        )
    payload_parts: list[bytes] = []
    cursor = 0
    seen_indices: set[int] = set()
    for item in ordered:
        if item.fragment_index in seen_indices:
            fail(E_FIELD, "reassembly.fragment_index", "duplicate fragment index with different offsets", fragment_index=item.fragment_index)
        seen_indices.add(item.fragment_index)
        if item.fragment_offset != cursor:
            fail(
                E_FIELD,
                "reassembly.fragment_offset",
                "fragment offsets are not contiguous",
                expected_offset=cursor,
                actual_offset=item.fragment_offset,
                fragment_index=item.fragment_index,
            )
        cursor += item.fragment_length
        payload_parts.append(item.payload)
    payload = b"".join(payload_parts)
    digest = sha256_hex(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        fail(
            E_FIELD,
            f"reassembly[{expected_message_key}].payload_sha256",
            "reassembled payload digest differs from the deterministic inner sample",
            expected=expected_sha256,
            actual=digest,
        )
    return ReassemblyResult(
        message_key=expected_message_key,
        message_id=message_id,
        tunnel_id=tunnel_id,
        payload=payload,
        payload_sha256=digest,
        fragment_indices=tuple(item.fragment_index for item in ordered),
        arrival_indices=tuple(arrival_indices),
        duplicate_indices=tuple(duplicate_indices),
        reassembly_version=reassembly_version,
        complete=True,
    )


def migrate_all_variants(record: FragmentRecord, *, message: MessagePlan) -> dict[str, NormalizedFragment]:
    context = _wire_context(message, record.total_fragments)
    result: dict[str, NormalizedFragment] = {}
    for target in ("f1", "f2", "f3"):
        source = record.normalized
        if target == source.source_format:
            result[target] = source
        else:
            source_wire = encode_fragment(source, source.source_format)
            result[target] = migrate_fragment(
                source_wire,
                source_context=context,
                target_format=target,
            )
    return result
