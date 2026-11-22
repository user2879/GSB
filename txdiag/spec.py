"""Strict deterministic transaction declaration and operation-sequence loading."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_text, digest
from .errors import SpecError
from .model import (
    DependencyEdge,
    InitialVersion,
    ISOLATION_ALIASES,
    Operation,
    Spec,
    TransactionSpec,
)


_SCALAR_TYPES = (str, int, float, bool, type(None))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_scalar(value: Any) -> bool:
    if not isinstance(value, _SCALAR_TYPES):
        return False
    return not (isinstance(value, float) and not math.isfinite(value))


def _read_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read spec file: {exc}", txid="<spec>", field="path") from exc
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SpecError(f"invalid JSON: {exc}", txid="<spec>", field="json") from exc


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read operations file: {exc}", txid="<spec>",
                        field="operations_file") from exc
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SpecError(f"invalid NDJSON at line {line_no}: {exc}", txid="<spec>",
                            field=f"operations[{line_no}].json") from exc
        if not isinstance(value, dict):
            raise SpecError(f"operation line {line_no} must be an object", txid="<spec>",
                            field=f"operations[{line_no}]")
        value["__line__"] = line_no
        rows.append(value)
    return rows


def _validate_row_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SpecError("row_keys must be a non-empty array", txid="<spec>", field="row_keys")
    out: list[str] = []
    seen: set[str] = set()
    for idx, row in enumerate(value):
        if not isinstance(row, str) or not row:
            raise SpecError("row key must be a non-empty string", txid="<spec>",
                            field=f"row_keys[{idx}]")
        if row in seen:
            raise SpecError("row keys must be unique", txid="<spec>", row=row,
                            field=f"row_keys[{idx}]")
        seen.add(row)
        out.append(row)
    return tuple(out)


def _validate_edges(value: Any, row_keys: set[str]) -> tuple[DependencyEdge, ...]:
    if value is None:
        value = []
    if not isinstance(value, list):
        raise SpecError("dependency_edges must be an array", txid="<spec>",
                        field="dependency_edges")
    edges: list[DependencyEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for idx, raw in enumerate(value):
        field = f"dependency_edges[{idx}]"
        if not isinstance(raw, dict):
            raise SpecError("dependency edge must be an object", txid="<spec>", field=field)
        source = raw.get("from", raw.get("source"))
        target = raw.get("to", raw.get("target"))
        kind = raw.get("kind", "reference")
        if not isinstance(source, str) or source not in row_keys:
            raise SpecError("dependency source is not a declared row key", txid="<spec>",
                            field=f"{field}.from")
        if not isinstance(target, str) or target not in row_keys:
            raise SpecError("dependency target is not a declared row key", txid="<spec>",
                            field=f"{field}.to")
        if source == target:
            raise SpecError("self dependency is not allowed", txid="<spec>", row=source,
                            field=f"{field}.to")
        if not isinstance(kind, str) or not kind:
            raise SpecError("dependency kind must be a non-empty string", txid="<spec>",
                            field=f"{field}.kind")
        key = (source, target, kind)
        if key in seen:
            raise SpecError("duplicate dependency edge", txid="<spec>", row=source,
                            field=field)
        seen.add(key)
        edges.append(DependencyEdge(source, target, kind))
    return tuple(sorted(edges, key=lambda e: (e.source, e.target, e.kind)))


def _validate_initial_versions(value: Any, row_keys: tuple[str, ...]) -> dict[str, InitialVersion]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise SpecError("initial_versions must be an object", txid="<spec>",
                        field="initial_versions")
    allowed = set(row_keys)
    out: dict[str, InitialVersion] = {}
    for row in sorted(value):
        if row not in allowed:
            raise SpecError("initial version references an unknown row key", txid="<spec>",
                            row=row, field=f"initial_versions.{row}")
        raw = value[row]
        if not isinstance(raw, dict):
            raw = {"value": raw, "version": 0, "status": "committed"}
        ver = raw.get("version", 0)
        status = raw.get("status", "committed")
        if not _is_int(ver) or ver < 0:
            raise SpecError("initial version must be a non-negative integer", txid="<spec>",
                            row=row, field=f"initial_versions.{row}.version")
        if status not in ("committed", "uncommitted"):
            raise SpecError("initial status must be committed or uncommitted", txid="<spec>",
                            row=row, field=f"initial_versions.{row}.status")
        if "value" not in raw or not _is_scalar(raw["value"]):
            raise SpecError("initial value must be a JSON scalar", txid="<spec>", row=row,
                            field=f"initial_versions.{row}.value")
        out[row] = InitialVersion(raw["value"], ver, status)
    for row in row_keys:
        out.setdefault(row, InitialVersion(0, 0, "committed"))
    return out


def _normalize_isolation(value: Any, txid: str) -> str:
    if not isinstance(value, str):
        raise SpecError("isolation level must be a string", txid=txid, field="isolation")
    key = value.strip().lower()
    canonical = ISOLATION_ALIASES.get(key)
    if canonical is None:
        raise SpecError("unknown isolation level", txid=txid, field="isolation",
                        details={"value": value, "allowed": list(ISOLATION_ALIASES)})
    return canonical


def _validate_transactions(raw_value: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_value, list) or not raw_value:
        raise SpecError("transactions must be a non-empty array", txid="<spec>",
                        field="transactions")
    out: list[dict[str, Any]] = []
    ids: set[str] = set()
    starts: set[int] = set()
    for idx, raw in enumerate(raw_value, 1):
        if not isinstance(raw, dict):
            raise SpecError("transaction must be an object", txid=f"<transaction:{idx}>",
                            field="transactions")
        txid = raw.get("id", raw.get("txid"))
        if not isinstance(txid, str) or not txid:
            raise SpecError("transaction id must be a non-empty string",
                            txid=f"<transaction:{idx}>", field="id")
        if txid in ids:
            raise SpecError("transaction id must be unique", txid=txid, field="id")
        ids.add(txid)
        start = raw.get("start_seq")
        if not _is_int(start) or start < 0:
            raise SpecError("start_seq must be a non-negative integer", txid=txid,
                            field="start_seq")
        if start in starts:
            raise SpecError("start_seq must be unique", txid=txid, field="start_seq")
        starts.add(start)
        isolation = _normalize_isolation(raw.get("isolation"), txid)
        out.append({"id": txid, "start_seq": start, "isolation": isolation,
                    "operations": raw.get("operations"), "raw_index": idx})
    return out


def _normalize_operations(raw: Any, txid: str, row_keys: set[str], *,
                          allow_embedded_txid: bool = False,
                          line_no: int | None = None) -> list[Operation]:
    field_base = f"operations[{line_no}]" if line_no is not None else "operations"
    if allow_embedded_txid:
        op_txid = raw.get("txid", raw.get("tx_id"))
        if op_txid != txid:
            raise SpecError("operation txid does not match its transaction", txid=txid,
                            field=f"{field_base}.txid")
    seq = raw.get("seq")
    if not _is_int(seq) or seq < 1:
        raise SpecError("operation seq must be a positive integer", txid=txid,
                        field=f"{field_base}.seq")
    kind = raw.get("kind")
    if kind not in ("read", "write"):
        raise SpecError("operation kind must be read or write", txid=txid,
                        field=f"{field_base}.kind")
    row = raw.get("row")
    if not isinstance(row, str) or row not in row_keys:
        raise SpecError("operation row is not a declared row key", txid=txid, row=row,
                        field=f"{field_base}.row")
    has_value = "value" in raw
    value = raw.get("value")
    if kind == "write":
        if not has_value:
            raise SpecError("write operation requires value", txid=txid, row=row,
                            field=f"{field_base}.value")
        if not _is_scalar(value):
            raise SpecError("write value must be a JSON scalar", txid=txid, row=row,
                            field=f"{field_base}.value")
    return [Operation(seq, kind, row, value, has_value)]


def _operations_by_tx(raw_ops: Iterable[dict[str, Any]], txids: set[str],
                      row_keys: set[str], *, external: bool) -> dict[str, list[Operation]]:
    grouped: dict[str, list[Operation]] = {txid: [] for txid in txids}
    embedded_txid = None if external or len(txids) != 1 else next(iter(txids))
    for raw in raw_ops:
        txid = (raw.get("txid", raw.get("tx_id")) if external else embedded_txid)
        if (not external and txid is not None
                and raw.get("txid", raw.get("tx_id")) not in (None, txid)):
            raise SpecError("embedded operation txid does not match its transaction",
                            txid=txid, field="operations.txid")
        if not isinstance(txid, str) or txid not in txids:
            line = raw.get("__line__")
            raise SpecError("operation references an unknown transaction", txid=str(txid),
                            field=f"operations[{line}].txid" if line else "operations.txid")
        line = raw.get("__line__") if external else None
        grouped[txid].extend(_normalize_operations(
            raw, txid, row_keys, allow_embedded_txid=external, line_no=line))
    return grouped


def _finish_transactions(txdefs: list[dict[str, Any]], grouped: dict[str, list[Operation]],
                         row_keys: set[str]) -> tuple[TransactionSpec, ...]:
    result: list[TransactionSpec] = []
    for raw in txdefs:
        txid = raw["id"]
        ops = grouped[txid]
        if not isinstance(ops, list) and ops is not None:
            raise SpecError("operations must be an array", txid=txid, field="operations")
        if not ops:
            raise SpecError("transaction must contain at least one operation", txid=txid,
                            field="operations")
        by_seq: dict[int, Operation] = {}
        for op in ops:
            if op.seq in by_seq:
                raise SpecError("operation sequence numbers must be unique", txid=txid,
                                row=op.row, field=f"operations[{op.seq}].seq")
            by_seq[op.seq] = op
        expected = list(range(1, len(ops) + 1))
        if sorted(by_seq) != expected:
            missing = next((n for n in expected if n not in by_seq), None)
            if missing is not None:
                raise SpecError("operation sequence must be continuous from 1", txid=txid,
                                field=f"operations[{missing}].seq")
            raise SpecError("operation sequence must be continuous from 1", txid=txid,
                            field="operations.seq")
        result.append(TransactionSpec(
            txid=txid,
            start_seq=int(raw["start_seq"]),
            isolation=str(raw["isolation"]),
            operations=tuple(by_seq[n] for n in expected),
        ))
    return tuple(sorted(result, key=lambda tx: tx.priority))


def load_spec(spec_path: str | Path, operations_path: str | Path | None = None) -> Spec:
    path = Path(spec_path).resolve()
    doc = _read_json(path)
    if not isinstance(doc, dict):
        raise SpecError("top-level JSON value must be an object", txid="<spec>", field="$")
    schema_version = doc.get("schema_version", 1)
    if not _is_int(schema_version) or schema_version != 1:
        raise SpecError("unsupported schema_version", txid="<spec>", field="schema_version")

    row_keys = _validate_row_keys(doc.get("row_keys"))
    row_set = set(row_keys)
    edges = _validate_edges(doc.get("dependency_edges", []), row_set)
    initial = _validate_initial_versions(doc.get("initial_versions", {}), row_keys)
    txdefs = _validate_transactions(doc.get("transactions"))
    txids = {tx["id"] for tx in txdefs}

    external_ops: list[dict[str, Any]] | None = None
    resolved_ops_path: str | None = None
    declared_ops = doc.get("operations_file")
    if operations_path is not None:
        ops_path = Path(operations_path).resolve()
    elif isinstance(declared_ops, str) and declared_ops:
        ops_path = (path.parent / declared_ops).resolve()
    elif declared_ops is not None:
        raise SpecError("operations_file must be a non-empty string", txid="<spec>",
                        field="operations_file")
    else:
        ops_path = None
    if ops_path is not None:
        external_ops = _read_ndjson(ops_path)
        resolved_ops_path = str(ops_path)

    embedded_present = any(tx.get("operations") is not None for tx in txdefs)
    if external_ops is not None and embedded_present:
        raise SpecError("cannot combine embedded operations and operations_file",
                        txid="<spec>", field="operations")

    if external_ops is not None:
        grouped = _operations_by_tx(external_ops, txids, row_set, external=True)
    else:
        grouped: dict[str, list[Operation]] = {}
        for raw in txdefs:
            txid = raw["id"]
            ops_raw = raw.get("operations")
            if not isinstance(ops_raw, list):
                raise SpecError("operations must be an array", txid=txid, field="operations")
            grouped[txid] = _operations_by_tx(
                ops_raw, {txid}, row_set, external=False)[txid]

    transactions = _finish_transactions(txdefs, grouped, row_set)
    normalized = {
        "schema_version": 1,
        "row_keys": list(row_keys),
        "dependency_edges": [
            {"from": e.source, "to": e.target, "kind": e.kind} for e in edges
        ],
        "transactions": [
            {
                "id": tx.txid,
                "start_seq": tx.start_seq,
                "isolation": tx.isolation,
                "operations": [
                    {
                        "seq": op.seq,
                        "kind": op.kind,
                        "row": op.row,
                        **({"value": op.value} if op.kind == "write" else {}),
                    }
                    for op in tx.operations
                ],
            }
            for tx in transactions
        ],
        "initial_versions": {
            row: {"value": iv.value, "version": iv.version_no, "status": iv.status}
            for row, iv in sorted(initial.items())
        },
    }
    return Spec(
        schema_version=1,
        row_keys=row_keys,
        dependency_edges=edges,
        transactions=transactions,
        initial_versions=initial,
        spec_sha256=digest(normalized),
        source_path=str(path),
        operations_path=resolved_ops_path,
        normalized=normalized,
    )


def spec_summary(spec: Spec) -> dict[str, Any]:
    return {
        "source_path": spec.source_path,
        "operations_path": spec.operations_path,
        "spec_sha256": spec.spec_sha256,
        "transaction_count": len(spec.transactions),
        "row_key_count": len(spec.row_keys),
        "dependency_edge_count": len(spec.dependency_edges),
        "canonical_text_sha256": digest(canonical_text(spec.normalized)),
    }
