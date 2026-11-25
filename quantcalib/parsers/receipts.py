"""Parsers for the three historical quant receipt formats.

q1: fixed-width text lines (layer name padded to 16 chars + 6-digit permille).
q2: JSON array adding param_version (int32) and time_ms (int64).
q3: partitioned big-endian binary container adding a group scale digest,
    covered duplicate calibration references and a weight package snapshot.

All three parse into ``UnifiedReceipt`` preserving source version, missing
field markers and per-field precision labels.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from quantcalib.errors import (
    E_LAYER_UNKNOWN,
    E_RECEIPT_FORMAT,
    E_RECEIPT_REFERENCE,
    E_TIME_MS_NEGATIVE,
    Issue,
)
from quantcalib.models import (
    PRECISION_INT32,
    PRECISION_INT64,
    PRECISION_UTF8,
    UnifiedReceipt,
)

Q1_NAME_WIDTH = 16
Q1_COV_WIDTH = 6
Q3_MAGIC = b"QCQ3"
Q3_CONTAINER_VERSION = 1

_Q1_MISSING = [
    "param_version", "time_ms", "group_scale_digest",
    "duplicate_refs", "weight_snapshot",
]
_Q2_MISSING = ["group_scale_digest", "duplicate_refs", "weight_snapshot"]


def _base_precision(version: str) -> dict[str, str]:
    labels = {"layer": PRECISION_UTF8, "coverage_permille": PRECISION_INT32}
    if version == "q2":
        labels["param_version"] = PRECISION_INT32
        labels["time_ms"] = PRECISION_INT64
    if version == "q3":
        labels["param_version"] = PRECISION_INT64
        labels["time_ms"] = PRECISION_INT64
        labels["group_scale_digest"] = PRECISION_UTF8
        labels["duplicate_refs"] = PRECISION_UTF8
        labels["weight_snapshot"] = PRECISION_UTF8
    return labels


# --------------------------------------------------------------------------- #
# q1
# --------------------------------------------------------------------------- #
def parse_q1(path: str | Path) -> list[UnifiedReceipt]:
    receipts: list[UnifiedReceipt] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            if len(line) < Q1_NAME_WIDTH + Q1_COV_WIDTH:
                raise ValueError(f"{E_RECEIPT_FORMAT}: q1 line {line_no} too short")
            name = line[:Q1_NAME_WIDTH].rstrip()
            coverage = int(line[Q1_NAME_WIDTH:Q1_NAME_WIDTH + Q1_COV_WIDTH])
            receipts.append(UnifiedReceipt(
                layer=name,
                coverage_permille=coverage,
                source_version="q1",
                missing_fields=list(_Q1_MISSING),
                precision_labels=_base_precision("q1"),
            ))
    return receipts


def write_q1(path: str | Path, receipts: list[UnifiedReceipt]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for receipt in receipts:
            if len(receipt.layer) > Q1_NAME_WIDTH:
                raise ValueError(f"{E_RECEIPT_FORMAT}: layer name too wide for q1")
            handle.write(
                f"{receipt.layer:<{Q1_NAME_WIDTH}}"
                f"{receipt.coverage_permille:0{Q1_COV_WIDTH}d}\n"
            )


# --------------------------------------------------------------------------- #
# q2
# --------------------------------------------------------------------------- #
def parse_q2(path: str | Path) -> list[UnifiedReceipt]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{E_RECEIPT_FORMAT}: q2 root must be a JSON array")

    receipts: list[UnifiedReceipt] = []
    for record in payload:
        receipts.append(UnifiedReceipt(
            layer=str(record["layer"]),
            coverage_permille=int(record["coverage_permille"]),
            source_version="q2",
            missing_fields=list(_Q2_MISSING),
            precision_labels=_base_precision("q2"),
            param_version=int(record["param_version"]),
            time_ms=int(record["time_ms"]),
        ))
    return receipts


def write_q2(path: str | Path, receipts: list[UnifiedReceipt]) -> None:
    payload = [
        {
            "layer": r.layer,
            "coverage_permille": r.coverage_permille,
            "param_version": int(r.param_version),
            "time_ms": int(r.time_ms),
        }
        for r in receipts
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)


# --------------------------------------------------------------------------- #
# q3 partitioned binary
# --------------------------------------------------------------------------- #
def _pack_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack(">H", len(raw)) + raw


def _unpack_str(buf: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from(">H", buf, offset)
    offset += 2
    value = buf[offset:offset + length].decode("utf-8")
    return value, offset + length


def write_q3(path: str | Path, receipts: list[UnifiedReceipt]) -> None:
    out = bytearray()
    out += Q3_MAGIC
    out += struct.pack(">HI", Q3_CONTAINER_VERSION, len(receipts))
    for r in receipts:
        out += _pack_str(r.layer)
        out += struct.pack(">i", r.coverage_permille)
        out += struct.pack(">q", int(r.param_version))
        out += struct.pack(">q", int(r.time_ms))
        out += _pack_str(r.group_scale_digest or "")
        refs = list(r.duplicate_refs)
        out += struct.pack(">H", len(refs))
        for ref in refs:
            out += _pack_str(ref)
        out += _pack_str(r.weight_snapshot or "")
    with open(path, "wb") as handle:
        handle.write(bytes(out))


def parse_q3(path: str | Path) -> list[UnifiedReceipt]:
    buf = Path(path).read_bytes()
    if buf[:4] != Q3_MAGIC:
        raise ValueError(f"{E_RECEIPT_FORMAT}: bad q3 magic in {path}")
    version, count = struct.unpack_from(">HI", buf, 4)
    if version != Q3_CONTAINER_VERSION:
        raise ValueError(f"{E_RECEIPT_FORMAT}: unsupported q3 version {version}")
    offset = 10
    receipts: list[UnifiedReceipt] = []
    for _ in range(count):
        layer, offset = _unpack_str(buf, offset)
        coverage, param_version, time_ms = struct.unpack_from(">iqq", buf, offset)
        offset += struct.calcsize(">iqq")
        digest, offset = _unpack_str(buf, offset)
        (ref_count,) = struct.unpack_from(">H", buf, offset)
        offset += 2
        refs: list[str] = []
        for _ in range(ref_count):
            ref, offset = _unpack_str(buf, offset)
            refs.append(ref)
        snapshot, offset = _unpack_str(buf, offset)
        receipts.append(UnifiedReceipt(
            layer=layer,
            coverage_permille=int(coverage),
            source_version="q3",
            missing_fields=[],
            precision_labels=_base_precision("q3"),
            param_version=int(param_version),
            time_ms=int(time_ms),
            group_scale_digest=digest or None,
            duplicate_refs=refs,
            weight_snapshot=snapshot or None,
        ))
    return receipts


def parse_any(path: str | Path) -> list[UnifiedReceipt]:
    suffix = Path(path).suffix.lower()
    if suffix == ".q1" or suffix == ".txt":
        return parse_q1(path)
    if suffix in {".json", ".q2"}:
        return parse_q2(path)
    if suffix in {".bin", ".q3"}:
        return parse_q3(path)
    raise ValueError(f"{E_RECEIPT_FORMAT}: unknown receipt extension {suffix}")


# --------------------------------------------------------------------------- #
# Cross-reference validation against the known layer universe.
# --------------------------------------------------------------------------- #
def validate_references(
    receipts: list[UnifiedReceipt],
    known_layers: dict[str, int],
) -> list[Issue]:
    """Check layer names, parameter version references and time_ms."""
    issues: list[Issue] = []
    known_names = set(known_layers)
    for receipt in receipts:
        if receipt.layer not in known_names:
            issues.append(Issue(
                E_LAYER_UNKNOWN,
                f"receipt references unknown layer {receipt.layer!r}",
                receipt.layer,
                {"source_version": receipt.source_version},
            ))
            continue
        if receipt.param_version is not None:
            if receipt.param_version > known_layers[receipt.layer]:
                issues.append(Issue(
                    E_RECEIPT_REFERENCE,
                    "param_version is ahead of the layer content_version",
                    receipt.layer,
                    {"param_version": receipt.param_version,
                     "content_version": known_layers[receipt.layer]},
                ))
        if receipt.time_ms is not None and receipt.time_ms < 0:
            issues.append(Issue(
                E_TIME_MS_NEGATIVE, "time_ms must be a non-negative integer",
                receipt.layer, {"time_ms": receipt.time_ms},
            ))
        for ref in receipt.duplicate_refs:
            if ref not in known_names:
                issues.append(Issue(
                    E_RECEIPT_REFERENCE,
                    f"duplicate calibration ref {ref!r} is unknown",
                    receipt.layer, {"ref": ref},
                ))
    return issues
