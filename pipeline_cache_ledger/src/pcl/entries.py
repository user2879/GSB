"""Cache entry codec for on-disk format versions v1 / v2 / v3.

On-disk envelope (big-endian, one self-describing binary file per key):

    magic   = b"PCL1"
    header  = magic | uint8 format_version | uint32 payload_len
    payload = payload_len bytes (JSON UTF-8)
    trailer = b"END1" | uint32 trailing_len | hex(sha256(payload))

The duplicated length field makes truncation / byte-flip corruption
detectable with an exact byte offset, and the payload digest proves content
integrity.  Decoding failures always raise E_CORRUPT with key + offset.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field

from .digest import digest_bytes
from .errors import CorruptEntryError

MAGIC = b"PCL1"
TRAILER = b"END1"
HEADER_LEN = 9
TRAILER_LEN = 4 + 4 + 64

FORMAT_V1 = 1
FORMAT_V2 = 2
FORMAT_V3 = 3
SUPPORTED_ENTRY_VERSIONS = (FORMAT_V1, FORMAT_V2, FORMAT_V3)
CURRENT_ENTRY_VERSION = FORMAT_V3

DEFAULT_COMPRESSED = False
DEFAULT_REFS = ()


@dataclass
class Entry:
    key: str
    digest: str
    version: int
    format_version: int = CURRENT_ENTRY_VERSION
    size: int | None = None
    stage_id: str | None = None
    refs: tuple = field(default_factory=tuple)  # tuples of (key, digest, version)
    compressed: bool = False
    backfilled: tuple = field(default_factory=tuple)
    source_format: int | None = None

    def normalized_refs(self):
        return tuple(sorted(self.refs, key=lambda r: r[0]))

    def to_v3_dict(self):
        return {
            "key": self.key,
            "version": self.version,
            "digest": self.digest,
            "size": self.size,
            "stage_id": self.stage_id,
            "refs": [
                {"key": k, "digest": d, "version": v}
                for (k, d, v) in self.normalized_refs()
            ],
            "compressed": self.compressed,
        }


def _payload_for(version, entry: Entry) -> bytes:
    if version == FORMAT_V1:
        payload = {"key": entry.key, "version": entry.version, "digest": entry.digest}
    elif version == FORMAT_V2:
        payload = {
            "key": entry.key,
            "version": entry.version,
            "digest": entry.digest,
            "size": entry.size if entry.size is not None else 0,
            "stage_id": entry.stage_id if entry.stage_id is not None else "",
        }
    elif version == FORMAT_V3:
        payload = entry.to_v3_dict()
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported entry version {version}")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_entry(entry: Entry, format_version: int = CURRENT_ENTRY_VERSION) -> bytes:
    if format_version not in SUPPORTED_ENTRY_VERSIONS:
        raise ValueError(f"unsupported entry version {format_version}")
    payload = _payload_for(format_version, entry)
    out = bytearray()
    out += MAGIC
    out += struct.pack(">BI", format_version, len(payload))
    out += payload
    out += TRAILER
    out += struct.pack(">I", len(payload))
    out += hashlib.sha256(payload).hexdigest().encode("ascii")
    return bytes(out)


def decode_entry(blob: bytes, key: str) -> Entry:
    offset = _check_header(blob, key)
    version, payload_len = struct.unpack(">BI", blob[4:9])
    if version not in SUPPORTED_ENTRY_VERSIONS:
        raise CorruptEntryError(
            f"unsupported entry format version {version}", key, 4
        )
    payload_end = HEADER_LEN + payload_len
    if len(blob) < payload_end + TRAILER_LEN:
        raise CorruptEntryError(
            "declared payload length exceeds file size", key, HEADER_LEN
        )
    payload = blob[HEADER_LEN:payload_end]
    trailer_pos = payload_end
    if blob[trailer_pos : trailer_pos + 4] != TRAILER:
        raise CorruptEntryError("missing trailer marker", key, trailer_pos)
    (trailing_len,) = struct.unpack(
        ">I", blob[trailer_pos + 4 : trailer_pos + 8]
    )
    if trailing_len != payload_len:
        raise CorruptEntryError(
            "length fields disagree", key, trailer_pos + 4
        )
    stored_hex = blob[trailer_pos + 8 : trailer_pos + 72].decode("ascii", "replace")
    actual_hex = hashlib.sha256(payload).hexdigest()
    if stored_hex != actual_hex:
        raise CorruptEntryError(
            "payload digest mismatch", key, trailer_pos + 8
        )
    try:
        data = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise CorruptEntryError("payload is not valid JSON", key, HEADER_LEN)

    return _entry_from_payload(data, version, key)


def _check_header(blob, key):
    if len(blob) < HEADER_LEN:
        raise CorruptEntryError("entry shorter than header", key, 0)
    if blob[:4] != MAGIC:
        raise CorruptEntryError("bad magic marker", key, 0)
    return HEADER_LEN


def _entry_from_payload(data, version, key) -> Entry:
    for required in ("key", "version", "digest"):
        if required not in data:
            raise CorruptEntryError(f"payload missing field {required}", key, HEADER_LEN)
    entry = Entry(
        key=data["key"],
        version=data["version"],
        digest=data["digest"],
        format_version=version,
        source_format=version,
    )
    if version >= FORMAT_V2:
        raw_size = data.get("size", 0)
        entry.size = int(raw_size) if isinstance(raw_size, int) else 0
        entry.stage_id = data.get("stage_id") or None
    if version >= FORMAT_V3:
        entry.refs = tuple(
            sorted(
                (
                    (r["key"], r["digest"], r["version"])
                    for r in data.get("refs", [])
                ),
                key=lambda r: r[0],
            )
        )
        entry.compressed = bool(data.get("compressed", DEFAULT_COMPRESSED))
    return entry


def normalize_to_v3(
    entry: Entry, *, size: int | None = None, stage_id: str | None = None
) -> Entry:
    """Return an in-memory v3 copy plus the set of fields that were filled."""
    if entry.format_version == FORMAT_V3 and entry.source_format == FORMAT_V3:
        return Entry(
            key=entry.key,
            digest=entry.digest,
            version=entry.version,
            format_version=FORMAT_V3,
            size=entry.size,
            stage_id=entry.stage_id,
            refs=entry.normalized_refs(),
            compressed=entry.compressed,
            backfilled=(),
            source_format=FORMAT_V3,
        )

    backfilled: list[str] = []
    new_size = entry.size
    if new_size is None:
        new_size = size if size is not None else 0
        backfilled.append("size")
    new_stage = entry.stage_id
    if new_stage is None:
        new_stage = stage_id if stage_id is not None else ""
        backfilled.append("stage_id")
    if entry.source_format and entry.source_format < FORMAT_V3:
        backfilled.extend(["refs", "compressed"])
    return Entry(
        key=entry.key,
        digest=entry.digest,
        version=entry.version,
        format_version=FORMAT_V3,
        size=new_size,
        stage_id=new_stage,
        refs=entry.normalized_refs(),
        compressed=entry.compressed,
        backfilled=tuple(sorted(set(backfilled))),
        source_format=entry.source_format,
    )


def verify_digest(entry: Entry, data: bytes, *, key: str):
    actual = digest_bytes(data)
    if actual != entry.digest:
        raise _digest_error(key, entry.digest, actual)
    return actual


def _digest_error(key, expected, actual):
    from .errors import DigestMismatchError

    return DigestMismatchError(
        "content digest does not match cache entry", key, expected, actual
    )
