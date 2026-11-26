"""Receipt exchange formats c1/c2/c3 with a single internal representation.

* c1 - fixed-width text line: shard index, integer permille check rate and
  integer quota version code.  Everything that cannot fit (float residual,
  digest, duplicate-delivery override reference) is parked in a sidecar so a
  c3 -> c1 -> c3 round trip loses nothing.
* c2 - JSON object with float check rate and named quota version.
* c3 - ``key=value`` text line that additionally keeps the shard digest and
  the overridden duplicate-delivery reference.  Fields absent in lower
  formats are promoted with an explicit missing marker and covered by an
  integrity label.

The parser/serializer pair is intentionally side-effect free; the round-trip
orchestration (sidecar bookkeeping, integrity labels) lives in
:class:`ReceiptFormatCodec`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import E_PROTOCOL, EdgeCacheError, Failure

C1 = "c1"
C2 = "c2"
C3 = "c3"
FORMATS = (C1, C2, C3)
MISSING_MARKER = "\u2205MISSING\u2205"
C1_WIDTHS = (8, 8, 8)
C1_LINE_WIDTH = sum(C1_WIDTHS)
C1_FIELD_NAMES = ("index", "check_rate_permille", "quota_code")


@dataclass
class Receipt:
    """Canonical internal representation shared by all three formats."""

    shard_id: str
    index: int
    node_id: str
    check_rate_permille: int
    residual: int
    check_rate_float: float
    quota_version: str
    quota_code: int
    digest: str | None
    duplicate_ref: str | None
    fixed_point: int
    precision_label: str
    clock_ms: int
    integrity_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Receipt":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def integrity_label_for(receipt: Receipt) -> str:
    payload = {
        "shard_id": receipt.shard_id,
        "index": receipt.index,
        "node_id": receipt.node_id,
        "check_rate_permille": receipt.check_rate_permille,
        "residual": receipt.residual,
        "check_rate_float": receipt.check_rate_float,
        "quota_version": receipt.quota_version,
        "quota_code": receipt.quota_code,
        "digest": receipt.digest if receipt.digest is not None else MISSING_MARKER,
        "duplicate_ref": receipt.duplicate_ref if receipt.duplicate_ref is not None else MISSING_MARKER,
        "fixed_point": receipt.fixed_point,
        "precision_label": receipt.precision_label,
        "clock_ms": receipt.clock_ms,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def precision_label_for(fixed_point: int, residual: int) -> str:
    return f"fp{fixed_point}:res{residual}"


def _c1_fixed(value: str, width: int) -> str:
    if len(value) > width:
        raise EdgeCacheError(
            Failure(E_PROTOCOL, "c1 field exceeds fixed width",
                    {"value": value, "width": width})
        )
    return value.rjust(width)


def _c1_slice(line: str, start: int, width: int) -> str:
    if len(line) < start + width:
        raise EdgeCacheError(
            Failure(E_PROTOCOL, "c1 line shorter than declared width", {"line": line})
        )
    return line[start:start + width]


def serialize_c1(receipt: Receipt) -> str:
    parts = [
        _c1_fixed(str(receipt.index), C1_WIDTHS[0]),
        _c1_fixed(str(receipt.check_rate_permille), C1_WIDTHS[1]),
        _c1_fixed(str(receipt.quota_code), C1_WIDTHS[2]),
    ]
    return "".join(parts)


def parse_c1(line: str) -> dict[str, int]:
    line = line.rstrip("\n")
    if len(line) != C1_LINE_WIDTH:
        raise EdgeCacheError(
            Failure(E_PROTOCOL, "c1 line has wrong fixed width",
                    {"length": len(line), "expected": C1_LINE_WIDTH})
        )
    values: dict[str, int] = {}
    start = 0
    for name, width in zip(C1_FIELD_NAMES, C1_WIDTHS):
        raw = _c1_slice(line, start, width).strip()
        start += width
        try:
            values[name] = int(raw)
        except ValueError as exc:
            raise EdgeCacheError(
                Failure(E_PROTOCOL, "c1 field is not an integer", {"field": name, "raw": raw})
            ) from exc
    return values


def serialize_c2(receipt: Receipt) -> str:
    payload = {
        "format": C2,
        "shard_id": receipt.shard_id,
        "index": receipt.index,
        "node_id": receipt.node_id,
        "check_rate_float": receipt.check_rate_float,
        "quota_version": receipt.quota_version,
        "quota_code": receipt.quota_code,
        "clock_ms": receipt.clock_ms,
        "fixed_point": receipt.fixed_point,
        "residual": receipt.residual,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def parse_c2(line: str) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise EdgeCacheError(Failure(E_PROTOCOL, "malformed c2 JSON", {"error": str(exc)})) from exc
    if not isinstance(payload, dict) or payload.get("format") != C2:
        raise EdgeCacheError(Failure(E_PROTOCOL, "not a c2 receipt"))
    return payload


def _c3_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("=", "\\=").replace("|", "\\|")


def _c3_unescape(value: str) -> str:
    out: list[str] = []
    escaped = False
    for char in value:
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            out.append(char)
    return "".join(out)


def serialize_c3(receipt: Receipt) -> str:
    fields = [
        ("format", C3),
        ("shard_id", receipt.shard_id),
        ("index", str(receipt.index)),
        ("node_id", receipt.node_id),
        ("check_rate_permille", str(receipt.check_rate_permille)),
        ("residual", str(receipt.residual)),
        ("check_rate_float", repr(receipt.check_rate_float)),
        ("quota_version", receipt.quota_version),
        ("quota_code", str(receipt.quota_code)),
        ("digest", receipt.digest if receipt.digest is not None else MISSING_MARKER),
        ("duplicate_ref", receipt.duplicate_ref if receipt.duplicate_ref is not None else MISSING_MARKER),
        ("fixed_point", str(receipt.fixed_point)),
        ("precision_label", receipt.precision_label),
        ("clock_ms", str(receipt.clock_ms)),
        ("integrity_label", receipt.integrity_label or integrity_label_for(receipt)),
    ]
    return "|".join(f"{_c3_escape(key)}={_c3_escape(value)}" for key, value in fields)


def parse_c3(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    token: list[str] = []
    tokens: list[str] = []
    escaped = False
    for char in line.rstrip("\n"):
        if escaped:
            token.append(char)
            escaped = False
        elif char == "\\":
            token.append(char)
            escaped = True
        elif char == "|":
            tokens.append("".join(token))
            token = []
        else:
            token.append(char)
    tokens.append("".join(token))
    for item in tokens:
        if "=" not in item:
            raise EdgeCacheError(Failure(E_PROTOCOL, "c3 token missing '='", {"token": item}))
        key, _, value = item.partition("=")
        out[_c3_unescape(key)] = _c3_unescape(value)
    if out.get("format") != C3:
        raise EdgeCacheError(Failure(E_PROTOCOL, "not a c3 receipt"))
    return out


class ReceiptFormatCodec:
    """Bidirectional c1/c2/c3 conversion around one internal representation.

    ``quota_index`` maps named quota versions to their integer codes (c1 can
    only carry integers).  ``sidecar`` is a JSON-serializable dict keyed by
    shard index; it preserves everything c1 cannot hold so that a downgrade
    followed by an upgrade is byte-faithful.
    """

    def __init__(self, quota_index: dict[str, int]) -> None:
        self.quota_index = dict(quota_index)
        self.name_by_code = {code: name for name, code in self.quota_index.items()}
        self.sidecar: dict[str, dict[str, Any]] = {}

    # -- canonical -> wire -------------------------------------------------
    def encode(self, receipt: Receipt, fmt: str) -> str:
        if fmt == C1:
            key = f"idx-{receipt.index}"
            self.sidecar[key] = {
                "shard_id": receipt.shard_id,
                "node_id": receipt.node_id,
                "residual": receipt.residual,
                "check_rate_float": receipt.check_rate_float,
                "quota_version": receipt.quota_version,
                "digest": receipt.digest,
                "duplicate_ref": receipt.duplicate_ref,
                "fixed_point": receipt.fixed_point,
                "precision_label": receipt.precision_label,
                "clock_ms": receipt.clock_ms,
                "integrity_label": receipt.integrity_label or integrity_label_for(receipt),
                "truncated_residual": receipt.residual,
            }
            return serialize_c1(receipt)
        if fmt == C2:
            return serialize_c2(receipt)
        if fmt == C3:
            if not receipt.integrity_label:
                receipt.integrity_label = integrity_label_for(receipt)
            return serialize_c3(receipt)
        raise EdgeCacheError(Failure(E_PROTOCOL, "unknown exchange format", {"format": fmt}))

    # -- wire -> canonical -------------------------------------------------
    def decode(self, line: str, fmt: str) -> Receipt:
        if fmt == C1:
            return self._decode_c1(line)
        if fmt == C2:
            return self._decode_c2(parse_c2(line))
        if fmt == C3:
            return self._decode_c3(parse_c3(line))
        raise EdgeCacheError(Failure(E_PROTOCOL, "unknown exchange format", {"format": fmt}))

    def _decode_c1(self, line: str) -> Receipt:
        values = parse_c1(line)
        extra = self.sidecar.get(f"idx-{values['index']}", {})
        quota_version = extra.get("quota_version") or self.name_by_code.get(values["quota_code"], "")
        digest = extra.get("digest")
        duplicate_ref = extra.get("duplicate_ref")
        fixed_point = int(extra.get("fixed_point", 0))
        residual = int(extra.get("residual", 0))
        check_rate_float = float(
            extra.get("check_rate_float", values["check_rate_permille"] / 1000.0)
        )
        receipt = Receipt(
            shard_id=str(extra.get("shard_id", f"idx-{values['index']}")),
            index=values["index"],
            node_id=str(extra.get("node_id", MISSING_MARKER)),
            check_rate_permille=values["check_rate_permille"],
            residual=residual,
            check_rate_float=check_rate_float,
            quota_version=quota_version,
            quota_code=values["quota_code"],
            digest=digest,
            duplicate_ref=duplicate_ref,
            fixed_point=fixed_point,
            precision_label=str(extra.get("precision_label", precision_label_for(fixed_point, residual))),
            clock_ms=int(extra.get("clock_ms", 0)),
            integrity_label=str(extra.get("integrity_label", "")),
        )
        if not receipt.integrity_label:
            receipt.integrity_label = integrity_label_for(receipt)
        return receipt

    def _decode_c2(self, payload: dict[str, Any]) -> Receipt:
        check_rate_float = float(payload["check_rate_float"])
        permille = round(check_rate_float * 1000.0)
        residual = int(payload.get("residual", 0))
        fixed_point = int(payload.get("fixed_point", 0))
        quota_version = str(payload["quota_version"])
        receipt = Receipt(
            shard_id=str(payload["shard_id"]),
            index=int(payload["index"]),
            node_id=str(payload["node_id"]),
            check_rate_permille=int(permille),
            residual=residual,
            check_rate_float=check_rate_float,
            quota_version=quota_version,
            quota_code=int(payload.get("quota_code", self.quota_index.get(quota_version, 0))),
            digest=None,
            duplicate_ref=None,
            fixed_point=fixed_point,
            precision_label=precision_label_for(fixed_point, residual),
            clock_ms=int(payload.get("clock_ms", 0)),
        )
        receipt.integrity_label = integrity_label_for(receipt)
        return receipt

    def _decode_c3(self, raw: dict[str, str]) -> Receipt:
        digest = raw.get("digest")
        duplicate_ref = raw.get("duplicate_ref")
        receipt = Receipt(
            shard_id=raw["shard_id"],
            index=int(raw["index"]),
            node_id=raw["node_id"],
            check_rate_permille=int(raw["check_rate_permille"]),
            residual=int(raw["residual"]),
            check_rate_float=float(raw["check_rate_float"]),
            quota_version=raw["quota_version"],
            quota_code=int(raw["quota_code"]),
            digest=None if digest == MISSING_MARKER else digest,
            duplicate_ref=None if duplicate_ref == MISSING_MARKER else duplicate_ref,
            fixed_point=int(raw["fixed_point"]),
            precision_label=raw["precision_label"],
            clock_ms=int(raw["clock_ms"]),
            integrity_label=raw["integrity_label"],
        )
        expected = integrity_label_for(receipt)
        if expected != receipt.integrity_label:
            raise EdgeCacheError(
                Failure(E_PROTOCOL, "c3 integrity label mismatch",
                        {"expected": expected, "actual": receipt.integrity_label})
            )
        return receipt
