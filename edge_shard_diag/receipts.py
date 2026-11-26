from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import Codes, DiagnosticError
from .spec import FormatSpec, SpecBundle


FORMAT_ORDER = {"d1": 1, "d2": 2, "d3": 3}
DEFAULT_CAPACITY_VERSION = 0
DEFAULT_SCALE_FACTOR = 1
DEFAULT_STABLE_RANK = -1

BASIS_CAPACITY = "d1 未承载容量版本，补 0；0 表示未知基线，采纳时低于所有正整数容量版本"
BASIS_SCALE = "d1/d2 未承载缩放因子时补受支持的最小 10^0=1"
BASIS_RANK = "d1/d2 未承载批次稳定名次，补 -1；-1 不进入 1..batch_size 合法性区间"
BASIS_DUPS = "低版本未承载重复投递引用，补空列表"
BASIS_DURATION = "d1 未承载耗时，补 0 并标记为补齐字段"
BASIS_SNAPSHOT = "低版本未承载样本摘要快照，补空串；空串表示无快照可校验"


@dataclass
class NormalizedReceipt:
    value: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.value)


def _error(code: str, message: str, *, line: int, source: str | None, field: str | None = None, details: dict[str, Any] | None = None) -> DiagnosticError:
    return DiagnosticError(code, message, field=field, line=line, source=source, details=details)


def encode_d1(record: dict[str, Any], format_spec: FormatSpec = FormatSpec()) -> str:
    w_node, w_start, w_end, w_summary = format_spec.d1_widths
    node_seq = int(record["node_seq"])
    start = int(record["seq_start"])
    end = int(record["seq_end"])
    prefix = str(record["summary_prefix"])[: format_spec.prefix_chars].ljust(w_summary)
    fields = [
        f"{node_seq:0{w_node}d}",
        f"{start:0{w_start}d}",
        f"{end:0{w_end}d}",
        prefix,
    ]
    for index, (field_value, width) in enumerate(zip(fields, format_spec.d1_widths)):
        if len(field_value) != width:
            raise DiagnosticError(Codes.FORMAT, f"d1 field {format_spec.d1_fields[index]} overflows width {width}", field=format_spec.d1_fields[index])
    line = "".join(fields)
    if len(line) != format_spec.d1_line_width:
        raise DiagnosticError(Codes.FORMAT, "d1 line width does not match declaration", field="d1.widths")
    return line


def decode_d1_line(line: str, line_no: int, source: str | Path | None = None, format_spec: FormatSpec = FormatSpec()) -> dict[str, Any]:
    source_name = str(source) if source else None
    text = line.rstrip("\r\n")
    if len(text) != format_spec.d1_line_width:
        raise _error(
            Codes.FORMAT,
            f"d1 fixed-width line must be {format_spec.d1_line_width} chars, got {len(text)}",
            line=line_no,
            source=source_name,
            field="d1.line_width",
        )
    widths = format_spec.d1_widths
    pos = 0
    slices: dict[str, str] = {}
    for name, width in zip(format_spec.d1_fields, widths):
        slices[name] = text[pos : pos + width]
        pos += width
    try:
        node_seq = int(slices["node_seq"])
        seq_start = int(slices["seq_start"])
        seq_end = int(slices["seq_end"])
    except ValueError as exc:
        raise _error(Codes.FORMAT, "d1 numeric field is not decodable", line=line_no, source=source_name, field="d1.numeric") from exc
    return {
        "format": "d1",
        "node_seq": node_seq,
        "seq_start": seq_start,
        "seq_end": seq_end,
        "summary_prefix": slices["summary_prefix"],
    }


def encode_json_receipt(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _missing(raw: dict[str, Any], required: Iterable[str], line_no: int, source: str | None) -> None:
    missing = [key for key in required if key not in raw or raw[key] is None or raw[key] == ""]
    if missing:
        raise _error(Codes.FIELD, f"missing required field(s): {', '.join(missing)}", line=line_no, source=source, field=missing[0], details={"missing": missing})


def _resolve_shard(
    raw: dict[str, Any],
    decoded: dict[str, Any],
    line_no: int,
    source: str | None,
    bundle: SpecBundle | None,
) -> str | None:
    shard_id = raw.get("shard_id")
    if shard_id is not None:
        if not isinstance(shard_id, str):
            raise _error(Codes.FIELD, "shard_id must be a string", line=line_no, source=source, field="shard_id")
        if bundle is not None and shard_id not in bundle.shards_by_id:
            raise _error(Codes.FIELD, f"shard_id does not exist in spec: {shard_id}", line=line_no, source=source, field="shard_id")
        return shard_id

    if bundle is None:
        return None
    node_seq = decoded.get("node_seq")
    node = bundle.nodes_by_seq.get(int(node_seq)) if node_seq is not None else None
    if node is None:
        raise _error(Codes.FIELD, f"node_seq does not exist in spec: {node_seq}", line=line_no, source=source, field="node_seq")
    matches = [
        shard
        for shard in bundle.shards
        if shard.node_id == node.id and shard.seq_start == decoded.get("seq_start") and shard.seq_end == decoded.get("seq_end")
    ]
    if len(matches) != 1:
        raise _error(
            Codes.FIELD,
            f"d1 interval must match exactly one shard, found {len(matches)}",
            line=line_no,
            source=source,
            field="seq_start",
        )
    return matches[0].id


def _batch_size(shard_id: str | None, raw: dict[str, Any], bundle: SpecBundle | None) -> int | None:
    if isinstance(raw.get("batch_size"), int) and not isinstance(raw["batch_size"], bool):
        return int(raw["batch_size"])
    if bundle is None or shard_id is None:
        return None
    shard = bundle.shards_by_id[shard_id]
    return sum(1 for item in bundle.shards if item.batch_id == shard.batch_id)


def normalize_receipt_dict(
    raw: dict[str, Any],
    line_no: int,
    source: str | Path | None = None,
    *,
    bundle: SpecBundle | None = None,
    known_delivery_refs: set[str] | frozenset[str] | None = None,
    format_spec: FormatSpec | None = None,
) -> NormalizedReceipt:
    source_name = str(source) if source else None
    if not raw.get("format"):
        raise _error(Codes.FIELD, "missing required field: format", line=line_no, source=source_name, field="format")
    fmt = raw["format"]
    if fmt not in FORMAT_ORDER:
        raise _error(Codes.FORMAT, f"unknown receipt format/version: {fmt!r}", line=line_no, source=source_name, field="format")

    decoded = dict(raw)
    if fmt == "d1":
        if set(raw) <= {"format"}:
            decoded = decode_d1_line(raw.get("line", ""), line_no, source, format_spec or (bundle.format_spec if bundle else FormatSpec()))
        # Raw d1 is normally produced by decode_d1_line before reaching here.
        decoded.setdefault("node_seq", raw.get("node_seq"))
        decoded.setdefault("seq_start", raw.get("seq_start"))
        decoded.setdefault("seq_end", raw.get("seq_end"))
        decoded.setdefault("summary_prefix", raw.get("summary_prefix"))

    required = ["receipt_id", "summary_prefix"] if fmt == "d1" else ["receipt_id", "shard_id", "summary_prefix", "capacity_version", "scale_factor", "duration_ms"]
    if fmt in {"d2", "d3"}:
        _missing(raw, required, line_no, source_name)
        if not isinstance(raw["receipt_id"], str):
            raise _error(Codes.FIELD, "receipt_id must be a string", line=line_no, source=source_name, field="receipt_id")
        if not isinstance(raw["summary_prefix"], str):
            raise _error(Codes.FIELD, "summary_prefix must be a string", line=line_no, source=source_name, field="summary_prefix")
        cap_version = raw["capacity_version"]
        scale = raw["scale_factor"]
        duration = raw["duration_ms"]
        if not isinstance(cap_version, int) or isinstance(cap_version, bool) or cap_version < 0:
            raise _error(Codes.FIELD, "capacity_version must be a non-negative integer", line=line_no, source=source_name, field="capacity_version")
        if not isinstance(scale, int) or isinstance(scale, bool) or scale not in (1, 10, 100, 1000, 10000, 1_000_000):
            raise _error(Codes.FIELD, "scale_factor must be a supported power of ten", line=line_no, source=source_name, field="scale_factor")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
            raise _error(Codes.FIELD, "duration_ms must be a non-negative number", line=line_no, source=source_name, field="duration_ms")

    shard_id = _resolve_shard(raw, decoded, line_no, source_name, bundle)
    supplemented: list[str] = []
    basis: dict[str, str] = {}

    capacity_version = raw.get("capacity_version", DEFAULT_CAPACITY_VERSION)
    if "capacity_version" not in raw:
        supplemented.append("capacity_version")
        basis["capacity_version"] = BASIS_CAPACITY
    scale_factor = raw.get("scale_factor", DEFAULT_SCALE_FACTOR)
    if "scale_factor" not in raw:
        supplemented.append("scale_factor")
        basis["scale_factor"] = BASIS_SCALE
    duration_ms = raw.get("duration_ms", 0)
    if "duration_ms" not in raw:
        supplemented.append("duration_ms")
        basis["duration_ms"] = BASIS_DURATION
    stable_rank = raw.get("stable_rank", DEFAULT_STABLE_RANK)
    if "stable_rank" not in raw:
        supplemented.append("stable_rank")
        basis["stable_rank"] = BASIS_RANK
    duplicate_refs = list(raw.get("duplicate_refs", []))
    if "duplicate_refs" not in raw:
        supplemented.append("duplicate_refs")
        basis["duplicate_refs"] = BASIS_DUPS
    sample_snapshot = raw.get("sample_snapshot", "")
    if "sample_snapshot" not in raw:
        supplemented.append("sample_snapshot")
        basis["sample_snapshot"] = BASIS_SNAPSHOT

    if fmt == "d3":
        _missing(raw, ["stable_rank", "duplicate_refs", "sample_snapshot"], line_no, source_name)
        if not isinstance(stable_rank, int) or isinstance(stable_rank, bool):
            raise _error(Codes.FIELD, "stable_rank must be an integer", line=line_no, source=source_name, field="stable_rank")
        batch_size = _batch_size(shard_id, raw, bundle)
        if stable_rank < 1 or (batch_size is not None and stable_rank > batch_size):
            raise _error(
                Codes.FIELD,
                f"stable_rank {stable_rank} is outside 1..{batch_size if batch_size is not None else '?'}",
                line=line_no,
                source=source_name,
                field="stable_rank",
                details={"batch_size": batch_size, "stable_rank": stable_rank, "reason": "stable_rank_out_of_bounds"},
            )
        if not isinstance(duplicate_refs, list) or not all(isinstance(ref, str) and ref for ref in duplicate_refs):
            raise _error(Codes.FIELD, "duplicate_refs must be a list of non-empty strings", line=line_no, source=source_name, field="duplicate_refs")
        if not isinstance(sample_snapshot, str):
            raise _error(Codes.FIELD, "sample_snapshot must be a string", line=line_no, source=source_name, field="sample_snapshot")
        refs = set(bundle.known_delivery_refs if bundle else frozenset())
        if known_delivery_refs is not None:
            refs.update(known_delivery_refs)
        dangling = [ref for ref in duplicate_refs if ref not in refs]
        if dangling:
            raise _error(Codes.DUP, f"duplicate delivery reference is dangling: {dangling[0]}", line=line_no, source=source_name, field="duplicate_refs", details={"dangling": dangling})
    else:
        # Explicit low-version defaults remain outside the legal d3 rank range.
        pass

    node_seq = decoded.get("node_seq")
    node_id = None
    if bundle is not None and node_seq is not None:
        node = bundle.nodes_by_seq.get(int(node_seq))
        node_id = node.id if node else None
    if bundle is not None and shard_id is not None and node_id is None:
        node_id = bundle.shards_by_id[shard_id].node_id
        node_seq = next(node.seq for node in bundle.nodes if node.id == node_id)

    receipt_id = raw.get("receipt_id")
    if not receipt_id:
        receipt_id = f"d1:L{line_no}:N{node_seq}:{decoded.get('seq_start')}-{decoded.get('seq_end')}"
    return NormalizedReceipt(
        {
            "receipt_id": receipt_id,
            "shard_id": shard_id,
            "format_version": fmt,
            "source_file": source_name,
            "source_line": line_no,
            "node_seq": node_seq,
            "node_id": node_id,
            "seq_start": decoded.get("seq_start"),
            "seq_end": decoded.get("seq_end"),
            "summary_prefix": raw.get("summary_prefix", decoded.get("summary_prefix")),
            "capacity_version": capacity_version,
            "scale_factor": scale_factor,
            "duration_ms": duration_ms,
            "stable_rank": stable_rank,
            "duplicate_refs": duplicate_refs,
            "sample_snapshot": sample_snapshot,
            "supplemented_fields": supplemented,
            "supplement_basis": basis,
        }
    )


def normalize_receipt_file(
    path: str | Path,
    format_version: str,
    *,
    bundle: SpecBundle | None = None,
    known_delivery_refs: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    path = Path(path)
    format_spec = bundle.format_spec if bundle else FormatSpec()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DiagnosticError(Codes.FIELD, f"cannot read receipt file: {exc}", source=str(path)) from exc

    results: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if format_version == "d1":
            decoded = decode_d1_line(line, line_no, path, format_spec)
            raw = dict(decoded)
        else:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise _error(Codes.FORMAT, f"invalid JSON receipt: {exc.msg}", line=line_no, source=str(path), field="$") from exc
            if not isinstance(raw, dict):
                raise _error(Codes.FORMAT, "JSON receipt must be an object", line=line_no, source=str(path), field="$")
        results.append(
            normalize_receipt_dict(
                raw,
                line_no,
                path,
                bundle=bundle,
                known_delivery_refs=known_delivery_refs,
                format_spec=format_spec,
            ).to_dict()
        )
    return adopt_receipts(results)


def adoption_key(receipt: dict[str, Any]) -> tuple[int, int, int, str, int]:
    rank = receipt.get("stable_rank")
    rank_key = int(rank) if isinstance(rank, int) and rank > 0 else 10**9
    source_file = receipt.get("source_file") or ""
    return (-int(receipt["capacity_version"]), -FORMAT_ORDER[receipt["format_version"]], rank_key, source_file, int(receipt["source_line"]))


def adopt_receipts(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(receipts, key=adoption_key)
    first_seen: set[str] = set()
    for order, receipt in enumerate(ordered, start=1):
        shard_id = receipt.get("shard_id") or receipt["receipt_id"]
        receipt["adoption_order"] = order
        receipt["adopted"] = shard_id not in first_seen
        first_seen.add(shard_id)
    return ordered
