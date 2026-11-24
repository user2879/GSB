"""Binary wire formats f1, f2 and f3 plus a format-neutral normalized view."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .errors import E_FIELD, E_FORMAT, fail


MAGIC = b"TUNF"
FORMAT_TO_VERSION = {"f1": 1, "f2": 2, "f3": 3}
VERSION_TO_FORMAT = {value: key for key, value in FORMAT_TO_VERSION.items()}
MESSAGE_ID_BYTES = 32
CHECKSUM_BYTES = 32
HEADER_DIGEST_BYTES = 32


@dataclass(frozen=True)
class WireContext:
    fragment_size: int
    effective_payload_limit: int
    max_message_bytes: int = 8 * 1024 * 1024
    total_fragments: int | None = None
    message_id: str | None = None
    generated_at_ms: int | None = None
    encapsulation_version: int = 1
    retry_count: int = 0


@dataclass(frozen=True)
class NormalizedFragment:
    tunnel_id: str
    fragment_index: int
    fragment_offset: int
    payload: bytes
    total_fragments: int
    message_id: str
    generated_at_ms: int
    checksum: str
    encapsulation_version: int
    fragment_header_digest: str
    retry_count: int
    source_format: str
    padded_fields: tuple[str, ...]
    padding_basis: Mapping[str, str]

    @property
    def fragment_length(self) -> int:
        return len(self.payload)

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def semantic_key(self) -> dict[str, Any]:
        """Fields that must be byte-for-byte stable after format migration."""

        return {
            "tunnel_id": self.tunnel_id,
            "fragment_index": self.fragment_index,
            "fragment_offset": self.fragment_offset,
            "fragment_length": self.fragment_length,
            "payload_sha256": self.payload_sha256,
            "total_fragments": self.total_fragments,
            "message_id": self.message_id,
            "generated_at_ms": self.generated_at_ms,
            "checksum": self.checksum,
            "encapsulation_version": self.encapsulation_version,
            "fragment_header_digest": self.fragment_header_digest,
            "retry_count": self.retry_count,
        }

    def to_dict(self, *, include_payload: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_format": self.source_format,
            "tunnel_id": self.tunnel_id,
            "fragment_index": self.fragment_index,
            "fragment_offset": self.fragment_offset,
            "fragment_length": self.fragment_length,
            "total_fragments": self.total_fragments,
            "message_id": self.message_id,
            "generated_at_ms": self.generated_at_ms,
            "checksum": self.checksum,
            "encapsulation_version": self.encapsulation_version,
            "fragment_header_digest": self.fragment_header_digest,
            "retry_count": self.retry_count,
            "payload_sha256": self.payload_sha256,
            "padded_fields": list(self.padded_fields),
            "padding_basis": dict(sorted(self.padding_basis.items())),
        }
        if include_payload:
            import base64

            result["payload_b64"] = base64.b64encode(self.payload).decode("ascii")
        return result


def _common_bytes(
    tunnel_id: str,
    fragment_index: int,
    fragment_offset: int,
    payload_length: int,
) -> bytes:
    tunnel_bytes = tunnel_id.encode("utf-8")
    if len(tunnel_bytes) > 65535:
        fail(E_FIELD, "fragment.tunnel_id", "UTF-8 tunnel id is too long")
    return (
        struct.pack(">H", len(tunnel_bytes))
        + tunnel_bytes
        + struct.pack(">III", fragment_index, fragment_offset, payload_length)
    )


def _parse_common(data: bytes) -> tuple[str, int, int, int, bytes, int]:
    cursor = 5

    def need(size: int, field: str) -> bytes:
        nonlocal cursor
        if cursor + size > len(data):
            fail(E_FIELD, f"fragment.{field}", "required wire field is truncated")
        chunk = data[cursor : cursor + size]
        cursor += size
        return chunk

    tunnel_length = struct.unpack(">H", need(2, "tunnel_id_length"))[0]
    if tunnel_length == 0:
        fail(E_FIELD, "fragment.tunnel_id", "tunnel id must not be empty")
    tunnel_raw = need(tunnel_length, "tunnel_id")
    try:
        tunnel_id = tunnel_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(E_FIELD, "fragment.tunnel_id", "tunnel id is not valid UTF-8", error=str(exc))
    fragment_index, fragment_offset, payload_length = struct.unpack(
        ">III", need(12, "fragment_index")
    )
    common = data[5:cursor]
    return tunnel_id, fragment_index, fragment_offset, payload_length, common, cursor


def compute_checksum(
    *,
    common: bytes,
    total_fragments: int,
    message_id: str,
    generated_at_ms: int,
    payload: bytes,
) -> str:
    message_raw = bytes.fromhex(message_id)
    material = (
        b"TUNF-F2-CHECKSUM\x00"
        + common
        + struct.pack(">I", total_fragments)
        + message_raw
        + struct.pack(">Q", generated_at_ms)
        + payload
    )
    return hashlib.sha256(material).hexdigest()


def compute_header_digest(
    *,
    common: bytes,
    total_fragments: int,
    message_id: str,
    generated_at_ms: int,
    checksum: str,
    encapsulation_version: int,
    retry_count: int,
) -> str:
    material = (
        b"TUNF-F3-HEADER\x00"
        + common
        + struct.pack(">I", total_fragments)
        + bytes.fromhex(message_id)
        + struct.pack(">Q", generated_at_ms)
        + bytes.fromhex(checksum)
        + struct.pack(">Hi", encapsulation_version, retry_count)
    )
    return hashlib.sha256(material).hexdigest()


def encode_fragment(fragment: NormalizedFragment, target_format: str) -> bytes:
    if target_format not in FORMAT_TO_VERSION:
        fail(E_FORMAT, "fragment.format", "unknown target format", actual=target_format)
    tunnel_bytes = fragment.tunnel_id.encode("utf-8")
    if len(tunnel_bytes) > 65535:
        fail(E_FIELD, "fragment.tunnel_id", "UTF-8 tunnel id is too long")
    common = _common_bytes(
        fragment.tunnel_id,
        fragment.fragment_index,
        fragment.fragment_offset,
        len(fragment.payload),
    )
    checksum = compute_checksum(
        common=common,
        total_fragments=fragment.total_fragments,
        message_id=fragment.message_id,
        generated_at_ms=fragment.generated_at_ms,
        payload=fragment.payload,
    )
    prefix = MAGIC + bytes([FORMAT_TO_VERSION[target_format]]) + common
    if target_format == "f1":
        return prefix + fragment.payload

    f2_header = (
        struct.pack(">I", fragment.total_fragments)
        + bytes.fromhex(fragment.message_id)
        + struct.pack(">Q", fragment.generated_at_ms)
        + bytes.fromhex(checksum)
    )
    if target_format == "f2":
        return prefix + f2_header + fragment.payload

    header_digest = compute_header_digest(
        common=common,
        total_fragments=fragment.total_fragments,
        message_id=fragment.message_id,
        generated_at_ms=fragment.generated_at_ms,
        checksum=checksum,
        encapsulation_version=fragment.encapsulation_version,
        retry_count=fragment.retry_count,
    )
    f3_header = (
        f2_header
        + struct.pack(">Hi", fragment.encapsulation_version, fragment.retry_count)
        + bytes.fromhex(header_digest)
    )
    return prefix + f3_header + fragment.payload


def _validate_context_payload(
    *,
    fragment_index: int,
    fragment_offset: int,
    payload_length: int,
    total_fragments: int | None,
    context: WireContext,
) -> None:
    if context.fragment_size <= 0:
        fail(E_FIELD, "context.fragment_size", "fragment size must be positive")
    if payload_length > context.effective_payload_limit:
        fail(
            E_FIELD,
            "fragment.payload_length",
            "fragment payload exceeds endpoint effective payload limit",
            payload_length=payload_length,
            effective_payload_limit=context.effective_payload_limit,
        )
    if payload_length > context.fragment_size:
        fail(
            E_FIELD,
            "fragment.payload_length",
            "fragment payload exceeds the configured fragment size tier",
            payload_length=payload_length,
            fragment_size=context.fragment_size,
        )
    if fragment_offset % context.fragment_size != 0:
        fail(
            E_FIELD,
            "fragment.fragment_offset",
            "fragment offset is not aligned to the fragment size",
            fragment_offset=fragment_offset,
            fragment_size=context.fragment_size,
        )
    if fragment_offset + payload_length > context.max_message_bytes:
        fail(
            E_FIELD,
            "fragment.fragment_offset",
            "fragment offset plus payload length exceeds the reassembly payload limit",
            fragment_offset=fragment_offset,
            payload_length=payload_length,
            max_message_bytes=context.max_message_bytes,
        )
    if total_fragments is not None:
        if total_fragments == 0:
            fail(E_FIELD, "fragment.total_fragments", "total fragment count must be positive")
        if fragment_index >= total_fragments:
            fail(
                E_FIELD,
                "fragment.fragment_index",
                "fragment index must be smaller than total fragment count",
                fragment_index=fragment_index,
                total_fragments=total_fragments,
            )
        effective_window = min(
            total_fragments * context.effective_payload_limit,
            total_fragments * context.fragment_size,
        )
        if fragment_offset + payload_length > effective_window:
            fail(
                E_FIELD,
                "fragment.fragment_offset",
                "fragment offset plus payload length exceeds total-fragment payload capacity",
                fragment_offset=fragment_offset,
                payload_length=payload_length,
                capacity=effective_window,
            )
        if context.total_fragments is not None and total_fragments != context.total_fragments:
            fail(
                E_FIELD,
                "fragment.total_fragments",
                "wire total fragment count disagrees with normalization context",
                wire=total_fragments,
                context=context.total_fragments,
            )


def decode_fragment(data: bytes, context: WireContext) -> NormalizedFragment:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        fail(E_FORMAT, "fragment.wire", "wire datagram must be bytes")
    wire = bytes(data)
    if len(wire) < 4:
        fail(E_FORMAT, "fragment.magic", "wire datagram is shorter than the magic value")
    if wire[:4] != MAGIC:
        fail(E_FORMAT, "fragment.magic", "wire magic does not match TUNF")
    if len(wire) < 5:
        fail(E_FORMAT, "fragment.format", "wire datagram has no format version byte")
    wire_version = wire[4]
    source_format = VERSION_TO_FORMAT.get(wire_version)
    if source_format is None:
        fail(
            E_FORMAT,
            "fragment.format",
            "unknown fragment format version",
            version=wire_version,
        )

    tunnel_id, fragment_index, fragment_offset, payload_length, common, cursor = _parse_common(wire)

    def need(size: int, field: str) -> bytes:
        nonlocal cursor
        if cursor + size > len(wire):
            fail(E_FIELD, f"fragment.{field}", "required wire field is truncated")
        chunk = wire[cursor : cursor + size]
        cursor += size
        return chunk

    if source_format == "f1":
        payload = need(payload_length, "payload")
        if cursor != len(wire):
            fail(E_FORMAT, "fragment.wire", "f1 datagram contains trailing bytes")
        total_fragments: int | None = context.total_fragments
        if total_fragments is None:
            if fragment_offset == 0:
                total_fragments = 1
            else:
                fail(
                    E_FIELD,
                    "fragment.total_fragments",
                    "f1 has no total count; a normalization context is required for non-zero offsets",
                )
        _validate_context_payload(
            fragment_index=fragment_index,
            fragment_offset=fragment_offset,
            payload_length=len(payload),
            total_fragments=total_fragments,
            context=context,
        )
        message_id = context.message_id or hashlib.sha256(
            b"TUNF-F1-MESSAGE\x00" + tunnel_id.encode("utf-8") + struct.pack(">I", total_fragments)
        ).hexdigest()
        generated_at_ms = 0 if context.generated_at_ms is None else context.generated_at_ms
        encapsulation_version = context.encapsulation_version
        retry_count = 0
        checksum = compute_checksum(
            common=common,
            total_fragments=total_fragments,
            message_id=message_id,
            generated_at_ms=generated_at_ms,
            payload=payload,
        )
        header_digest = compute_header_digest(
            common=common,
            total_fragments=total_fragments,
            message_id=message_id,
            generated_at_ms=generated_at_ms,
            checksum=checksum,
            encapsulation_version=encapsulation_version,
            retry_count=retry_count,
        )
        basis = {
            "total_fragments": (
                "context.message_manifest.total_fragments"
                if context.total_fragments is not None
                else "inferred:single_fragment_with_zero_offset"
            ),
            "message_id": (
                "context.message_manifest.message_id"
                if context.message_id is not None
                else "derived:f1_tunnel_and_inferred_total"
            ),
            "generated_at_ms": (
                "context.message_manifest.generated_at_ms"
                if context.generated_at_ms is not None
                else "format_default:epoch_ms=0"
            ),
            "checksum": "computed:f2_checksum_rule",
            "encapsulation_version": "context_or_default:f3_encapsulation_version=1",
            "fragment_header_digest": "computed:f3_header_digest_rule",
            "retry_count": "format_default:f3_retry_count=0",
        }
        return NormalizedFragment(
            tunnel_id=tunnel_id,
            fragment_index=fragment_index,
            fragment_offset=fragment_offset,
            payload=payload,
            total_fragments=total_fragments,
            message_id=message_id,
            generated_at_ms=generated_at_ms,
            checksum=checksum,
            encapsulation_version=encapsulation_version,
            fragment_header_digest=header_digest,
            retry_count=retry_count,
            source_format="f1",
            padded_fields=(
                "checksum",
                "encapsulation_version",
                "fragment_header_digest",
                "generated_at_ms",
                "message_id",
                "retry_count",
                "total_fragments",
            ),
            padding_basis=basis,
        )

    total_fragments = struct.unpack(">I", need(4, "total_fragments"))[0]
    if total_fragments == 0:
        fail(E_FIELD, "fragment.total_fragments", "total fragment count must be positive")
    message_id_raw = need(MESSAGE_ID_BYTES, "message_id")
    message_id = message_id_raw.hex()
    generated_at_ms = struct.unpack(">Q", need(8, "generated_at_ms"))[0]
    checksum_raw = need(CHECKSUM_BYTES, "checksum")
    checksum = checksum_raw.hex()
    computed_checksum = compute_checksum(
        common=common,
        total_fragments=total_fragments,
        message_id=message_id,
        generated_at_ms=generated_at_ms,
        payload=wire[cursor + (6 + HEADER_DIGEST_BYTES if source_format == "f3" else 0) :],
    )

    if source_format == "f2":
        payload = need(payload_length, "payload")
        if cursor != len(wire):
            fail(E_FORMAT, "fragment.wire", "f2 datagram contains trailing bytes")
        if checksum != computed_checksum:
            fail(
                E_FIELD,
                "fragment.checksum",
                "f2 checksum does not match the explicit recomputation rule",
                expected=computed_checksum,
                actual=checksum,
            )
        _validate_context_payload(
            fragment_index=fragment_index,
            fragment_offset=fragment_offset,
            payload_length=len(payload),
            total_fragments=total_fragments,
            context=context,
        )
        encapsulation_version = context.encapsulation_version
        retry_count = 0
        header_digest = compute_header_digest(
            common=common,
            total_fragments=total_fragments,
            message_id=message_id,
            generated_at_ms=generated_at_ms,
            checksum=checksum,
            encapsulation_version=encapsulation_version,
            retry_count=retry_count,
        )
        return NormalizedFragment(
            tunnel_id=tunnel_id,
            fragment_index=fragment_index,
            fragment_offset=fragment_offset,
            payload=payload,
            total_fragments=total_fragments,
            message_id=message_id,
            generated_at_ms=generated_at_ms,
            checksum=checksum,
            encapsulation_version=encapsulation_version,
            fragment_header_digest=header_digest,
            retry_count=retry_count,
            source_format="f2",
            padded_fields=("encapsulation_version", "fragment_header_digest", "retry_count"),
            padding_basis={
                "encapsulation_version": "context_or_default:f3_encapsulation_version=1",
                "fragment_header_digest": "computed:f3_header_digest_rule",
                "retry_count": "format_default:f3_retry_count=0",
            },
        )

    encapsulation_version = struct.unpack(">H", need(2, "encapsulation_version"))[0]
    if encapsulation_version == 0:
        fail(
            E_FIELD,
            "fragment.encapsulation_version",
            "encapsulation version must be positive",
            actual=encapsulation_version,
        )
    retry_count = struct.unpack(">i", need(4, "retry_count"))[0]
    if retry_count < 0:
        fail(E_FIELD, "fragment.retry_count", "retry count must not be negative", actual=retry_count)
    digest_raw = need(HEADER_DIGEST_BYTES, "fragment_header_digest")
    fragment_header_digest = digest_raw.hex()
    payload = need(payload_length, "payload")
    if cursor != len(wire):
        fail(E_FORMAT, "fragment.wire", "f3 datagram contains trailing bytes")
    computed_checksum = compute_checksum(
        common=common,
        total_fragments=total_fragments,
        message_id=message_id,
        generated_at_ms=generated_at_ms,
        payload=payload,
    )
    if checksum != computed_checksum:
        fail(
            E_FIELD,
            "fragment.checksum",
            "f3 checksum does not match the explicit recomputation rule",
            expected=computed_checksum,
            actual=checksum,
        )
    computed_digest = compute_header_digest(
        common=common,
        total_fragments=total_fragments,
        message_id=message_id,
        generated_at_ms=generated_at_ms,
        checksum=checksum,
        encapsulation_version=encapsulation_version,
        retry_count=retry_count,
    )
    if fragment_header_digest != computed_digest:
        fail(
            E_FIELD,
            "fragment.fragment_header_digest",
            "f3 fragment header digest does not match the explicit recomputation rule",
            expected=computed_digest,
            actual=fragment_header_digest,
        )
    _validate_context_payload(
        fragment_index=fragment_index,
        fragment_offset=fragment_offset,
        payload_length=len(payload),
        total_fragments=total_fragments,
        context=context,
    )
    return NormalizedFragment(
        tunnel_id=tunnel_id,
        fragment_index=fragment_index,
        fragment_offset=fragment_offset,
        payload=payload,
        total_fragments=total_fragments,
        message_id=message_id,
        generated_at_ms=generated_at_ms,
        checksum=checksum,
        encapsulation_version=encapsulation_version,
        fragment_header_digest=fragment_header_digest,
        retry_count=retry_count,
        source_format="f3",
        padded_fields=(),
        padding_basis={},
    )


def migrate_fragment(
    wire: bytes,
    *,
    source_context: WireContext,
    target_format: str,
) -> NormalizedFragment:
    normalized = decode_fragment(wire, source_context)
    target_wire = encode_fragment(normalized, target_format)
    return decode_fragment(target_wire, source_context)


def peek_wire_identity(data: bytes) -> dict[str, Any]:
    """Parse only the fields needed by the loopback receiver for acknowledgements."""

    wire = bytes(data)
    if len(wire) < 5 or wire[:4] != MAGIC:
        fail(E_FORMAT, "fragment.magic", "wire magic does not match TUNF")
    version = wire[4]
    source_format = VERSION_TO_FORMAT.get(version)
    if source_format is None:
        fail(E_FORMAT, "fragment.format", "unknown fragment format version", version=version)
    tunnel_id, fragment_index, fragment_offset, payload_length, _common, cursor = _parse_common(wire)
    message_id: str | None = None
    if source_format in {"f2", "f3"}:
        if cursor + 4 + MESSAGE_ID_BYTES > len(wire):
            fail(E_FIELD, "fragment.message_id", "required wire field is truncated")
        message_id = wire[cursor + 4 : cursor + 4 + MESSAGE_ID_BYTES].hex()
    return {
        "format": source_format,
        "tunnel_id": tunnel_id,
        "fragment_index": fragment_index,
        "fragment_offset": fragment_offset,
        "fragment_length": payload_length,
        "message_id": message_id,
    }


def with_retry_count(fragment: NormalizedFragment, retry_count: int) -> NormalizedFragment:
    if retry_count < 0:
        fail(E_FIELD, "fragment.retry_count", "retry count must not be negative", actual=retry_count)
    common = _common_bytes(
        fragment.tunnel_id,
        fragment.fragment_index,
        fragment.fragment_offset,
        len(fragment.payload),
    )
    checksum = compute_checksum(
        common=common,
        total_fragments=fragment.total_fragments,
        message_id=fragment.message_id,
        generated_at_ms=fragment.generated_at_ms,
        payload=fragment.payload,
    )
    digest = compute_header_digest(
        common=common,
        total_fragments=fragment.total_fragments,
        message_id=fragment.message_id,
        generated_at_ms=fragment.generated_at_ms,
        checksum=checksum,
        encapsulation_version=fragment.encapsulation_version,
        retry_count=retry_count,
    )
    return replace(
        fragment,
        checksum=checksum,
        fragment_header_digest=digest,
        retry_count=retry_count,
    )
