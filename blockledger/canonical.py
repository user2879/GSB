from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(data: bytes | str | Any) -> str:
    digest = hashlib.sha256()
    if isinstance(data, bytes):
        digest.update(data)
    elif isinstance(data, str):
        digest.update(data.encode("utf-8"))
    else:
        digest.update(canonical_bytes(data))
    return digest.hexdigest()


def u64be(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("uint64 value must be an integer")
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("uint64 value is out of range")
    return value.to_bytes(8, byteorder="big", signed=False)


def _length_prefixed(data: bytes) -> bytes:
    return u64be(len(data)) + data


def content_digest(
    volume_id: str,
    block_index: int,
    baseline_version: int,
    content: bytes,
) -> str:
    """Return the explicit metadata-bound SHA-256 digest for one block.

    Preimage format, all integers big-endian::

        b"BLOCKLEDGER-CONTENT-V1\\n"
        || u64be(length(volume_id)) || volume_id UTF-8
        || u64be(block_index)
        || u64be(baseline_version)
        || u64be(length(content)) || content

    The length prefixes make the encoding unambiguous.
    """
    if not isinstance(volume_id, str) or not volume_id:
        raise ValueError("volume_id must be a non-empty string")
    if not isinstance(block_index, int) or isinstance(block_index, bool):
        raise ValueError("block_index must be an integer")
    if block_index < 0 or block_index > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("block_index is out of range")
    if not isinstance(baseline_version, int) or isinstance(baseline_version, bool):
        raise ValueError("baseline_version must be an integer")
    if baseline_version < 0 or baseline_version > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("baseline_version is out of range")
    if not isinstance(content, bytes):
        raise ValueError("content must be bytes")

    hasher = hashlib.sha256()
    hasher.update(b"BLOCKLEDGER-CONTENT-V1\n")
    hasher.update(_length_prefixed(volume_id.encode("utf-8")))
    hasher.update(u64be(block_index))
    hasher.update(u64be(baseline_version))
    hasher.update(u64be(len(content)))
    hasher.update(content)
    return hasher.hexdigest()


def block_seed(material: str | bytes | int) -> int:
    if isinstance(material, int):
        raw = f"int:{material}".encode()
    elif isinstance(material, str):
        raw = material.encode("utf-8")
    else:
        raw = material
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def volume_set_version(volumes: list[dict[str, Any]]) -> int:
    """Deterministic positive 63-bit version identifying the volume set."""
    identities = [
        {
            "id": volume["id"],
            "block_size": volume["block_size"],
            "block_count": volume["block_count"],
            "seed": volume["seed"],
            "baseline_version": volume["baseline_version"],
        }
        for volume in sorted(volumes, key=lambda item: item["id"])
    ]
    raw = sha256_hex(b"blockledger-volume-set-v1\0" + canonical_bytes(identities))
    return int(raw[:16], 16) & 0x7FFFFFFFFFFFFFFF


def spec_identity(spec: dict[str, Any]) -> dict[str, Any]:
    from . import FORMAT_VERSION

    return {
        "format_version": FORMAT_VERSION,
        "volume_set_version": volume_set_version(spec["volumes"]),
        "content_digest": sha256_hex(canonical_bytes(spec)),
    }
