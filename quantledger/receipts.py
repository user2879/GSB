"""Parsing, normalization, conversion and arbitration for q1/q2/q3 receipts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import E_BAD_RECEIPT, MISSING, BusinessIssue, QLabError, issue
from .util import atomic_write_json, canonical_json, digest_json

SUPPORTED_FORMATS = (1, 2, 3)
ALL_MISSING_FIELDS = (
    "revision",
    "duration_ms",
    "worker_id",
    "stable_rank",
    "superseded_refs",
    "quant_params_summary",
)


def _coerce_format(raw: dict[str, Any]) -> int:
    value = raw.get("q", raw.get("format", raw.get("version")))
    if isinstance(value, str):
        value = value.lower().lstrip("q")
        try:
            value = int(value)
        except ValueError as exc:
            raise QLabError(E_BAD_RECEIPT, f"unknown receipt format {raw.get('format')!r}") from exc
    if value not in SUPPORTED_FORMATS:
        raise QLabError(E_BAD_RECEIPT, f"unsupported receipt format {value!r}")
    return int(value)


def _get(raw: dict[str, Any], *names: str, default: Any = MISSING) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _missing(value: Any) -> bool:
    return value is MISSING or value is None or value == MISSING


@dataclass
class Receipt:
    format: int
    batch_id: str
    layer_name: str
    output_summary: str
    revision: int | str = MISSING
    duration_ms: int | str = MISSING
    worker_id: str = MISSING
    stable_rank: int | str = MISSING
    superseded_refs: list[str] | str = MISSING
    quant_params_summary: dict[str, Any] | str = MISSING
    completeness: dict[str, Any] = field(default_factory=dict)
    sidecar_used: bool = False
    source_file: str | None = None

    def __post_init__(self) -> None:
        self.batch_id = str(self.batch_id)
        self.layer_name = str(self.layer_name)
        self.output_summary = str(self.output_summary)
        if not self.completeness:
            self.completeness = completeness_for(self)

    def missing_fields(self) -> list[str]:
        fields = []
        for name in ALL_MISSING_FIELDS:
            if _missing(getattr(self, name)):
                fields.append(name)
        return fields

    def refresh_completeness(self) -> None:
        self.completeness = completeness_for(self)

    def semantic_core(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "layer_name": self.layer_name,
            "output_summary": self.output_summary,
            "stable_rank": self.stable_rank,
            "superseded_refs": self.superseded_refs,
            "quant_params_summary": self.quant_params_summary,
        }

    def receipt_ref(self) -> str:
        return digest_json(
            {
                "format": self.format,
                "batch_id": self.batch_id,
                "layer_name": self.layer_name,
                "output_summary": self.output_summary,
                "revision": self.revision,
                "duration_ms": self.duration_ms,
                "worker_id": self.worker_id,
                "stable_rank": self.stable_rank,
                "superseded_refs": self.superseded_refs,
                "quant_params_summary": self.quant_params_summary,
            }
        )

    def arbitration_values(self) -> dict[str, Any]:
        revision = -1 if _missing(self.revision) else int(self.revision)
        duration = -1 if _missing(self.duration_ms) else int(self.duration_ms)
        worker = "\uffff" if _missing(self.worker_id) else str(self.worker_id)
        return {
            "revision": revision,
            "duration_ms": duration,
            "worker_id": worker,
            "receipt_ref": self.receipt_ref(),
        }

    def arbitration_sort_key(self) -> tuple[Any, ...]:
        values = self.arbitration_values()
        return (
            -values["revision"],
            -values["duration_ms"],
            values["worker_id"],
            -self.format,
            values["receipt_ref"],
        )

    def to_internal_dict(self, *, include_completeness: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "batch_id": self.batch_id,
            "layer_name": self.layer_name,
            "output_summary": self.output_summary,
            "revision": self.revision,
            "duration_ms": self.duration_ms,
            "worker_id": self.worker_id,
            "stable_rank": self.stable_rank,
            "superseded_refs": copy.deepcopy(self.superseded_refs),
            "quant_params_summary": copy.deepcopy(self.quant_params_summary),
            "receipt_ref": self.receipt_ref(),
        }
        if include_completeness:
            payload["completeness"] = copy.deepcopy(self.completeness)
            payload["sidecar_used"] = self.sidecar_used
        return payload

    def to_q1(self) -> dict[str, Any]:
        return {
            "q": 1,
            "batch_id": self.batch_id,
            "layer_name": self.layer_name,
            "output_summary": self.output_summary,
        }

    def to_q2(self) -> dict[str, Any]:
        return {
            "q": 2,
            "batch_id": self.batch_id,
            "layer_name": self.layer_name,
            "output_summary": self.output_summary,
            "revision": self.revision,
            "duration_ms": self.duration_ms,
            "worker_id": self.worker_id,
        }

    def to_q3(self) -> dict[str, Any]:
        return {
            "q": 3,
            "batch_id": self.batch_id,
            "layer_name": self.layer_name,
            "output_summary": self.output_summary,
            "revision": self.revision,
            "duration_ms": self.duration_ms,
            "worker_id": self.worker_id,
            "stable_rank": self.stable_rank,
            "superseded_refs": copy.deepcopy(self.superseded_refs),
            "quant_params_summary": copy.deepcopy(self.quant_params_summary),
            "completeness": copy.deepcopy(self.completeness),
        }

    def with_q3_metadata(self, *, stable_rank: int, superseded_refs: list[str], quant_params_summary: dict[str, Any]) -> "Receipt":
        updated = copy.deepcopy(self)
        updated.stable_rank = int(stable_rank)
        updated.superseded_refs = list(superseded_refs)
        updated.quant_params_summary = copy.deepcopy(quant_params_summary)
        updated.refresh_completeness()
        return updated


def completeness_for(receipt: Receipt) -> dict[str, Any]:
    missing = receipt.missing_fields()
    if not missing:
        label = "restored_from_sidecar" if receipt.sidecar_used else "complete"
    elif receipt.format == 1 and receipt.sidecar_used:
        label = "q1_upgraded_sidecar_partial"
    elif receipt.format == 1:
        label = "q1_upgraded_partial"
    elif receipt.format == 2:
        label = "q2_upgraded_partial"
    else:
        label = "q3_partial"
    return {"label": label, "missing_fields": missing}


def _parse_q2_fields(raw: dict[str, Any]) -> dict[str, Any]:
    revision = _get(raw, "revision", "operator_revision")
    duration = _get(raw, "duration_ms", "elapsed_ms", "integer_duration_ms")
    worker = _get(raw, "worker_id", "worker")
    if not _missing(revision) and not isinstance(revision, int):
        raise QLabError(E_BAD_RECEIPT, "revision must be an integer")
    if not _missing(duration) and (not isinstance(duration, int) or duration < 0):
        raise QLabError(E_BAD_RECEIPT, "duration_ms must be a non-negative integer")
    if not _missing(worker) and not isinstance(worker, str):
        raise QLabError(E_BAD_RECEIPT, "worker_id must be a string")
    return {"revision": revision, "duration_ms": duration, "worker_id": worker}


def _parse_q3_fields(raw: dict[str, Any]) -> dict[str, Any]:
    rank = _get(raw, "stable_rank", "rank")
    superseded = _get(raw, "superseded_refs", "supersedes", "overridden_refs")
    quant_summary = _get(raw, "quant_params_summary", "quant_summary")
    if not _missing(rank) and (not isinstance(rank, int) or rank < 1):
        raise QLabError(E_BAD_RECEIPT, "stable_rank must be a positive integer")
    if not _missing(superseded) and not isinstance(superseded, list):
        raise QLabError(E_BAD_RECEIPT, "superseded_refs must be a list")
    if not _missing(quant_summary) and not isinstance(quant_summary, dict):
        raise QLabError(E_BAD_RECEIPT, "quant_params_summary must be an object")
    return {"stable_rank": rank, "superseded_refs": superseded, "quant_params_summary": quant_summary}


def _sidecar_key(batch_id: str, layer_name: str, output_summary: str) -> str:
    return digest_json({"batch_id": batch_id, "layer_name": layer_name, "output_summary": output_summary})


def _delivery_sidecar_key(receipt: Receipt) -> str:
    return digest_json(
        {
            "batch_id": receipt.batch_id,
            "layer_name": receipt.layer_name,
            "output_summary": receipt.output_summary,
            "revision": receipt.revision,
            "duration_ms": receipt.duration_ms,
            "worker_id": receipt.worker_id,
            "stable_rank": receipt.stable_rank,
            "superseded_refs": receipt.superseded_refs,
            "quant_params_summary": receipt.quant_params_summary,
        }
    )


def parse_receipt(raw: Any, *, sidecars: dict[str, Any] | None = None, source_file: str | None = None) -> Receipt:
    if not isinstance(raw, dict):
        raise QLabError(E_BAD_RECEIPT, "receipt must be an object")
    version = _coerce_format(raw)
    batch_id = _get(raw, "batch_id", "batch")
    layer_name = _get(raw, "layer_name", "layer")
    output_summary = _get(raw, "output_summary", "output_digest", "dequantized_output_summary")
    if _missing(batch_id) or _missing(layer_name) or _missing(output_summary):
        raise QLabError(E_BAD_RECEIPT, "receipt requires batch_id, layer_name and output_summary")
    kwargs: dict[str, Any] = {}
    if version >= 2:
        kwargs.update(_parse_q2_fields(raw))
    if version >= 3:
        kwargs.update(_parse_q3_fields(raw))
    receipt = Receipt(
        format=version,
        batch_id=str(batch_id),
        layer_name=str(layer_name),
        output_summary=str(output_summary),
        **kwargs,
        source_file=source_file,
    )
    if version == 1 and sidecars:
        reference = raw.get("sidecar_key") if isinstance(raw.get("sidecar_key"), str) else None
        sidecar = sidecars.get(reference) if reference else None
        if sidecar is None:
            sidecar = sidecars.get(_sidecar_key(receipt.batch_id, receipt.layer_name, receipt.output_summary))
        if sidecar is not None:
            for name in ALL_MISSING_FIELDS:
                if name not in sidecar:
                    continue
                setattr(receipt, name, copy.deepcopy(sidecar[name]))
            receipt.sidecar_used = True
            receipt.refresh_completeness()
    return receipt


def upgrade_to_q3(receipt: Receipt, *, sidecars: dict[str, Any] | None = None) -> Receipt:
    upgraded = copy.deepcopy(receipt)
    source_format = receipt.format
    if receipt.format == 1 and sidecars:
        sidecar = sidecars.get(_delivery_sidecar_key(receipt))
        if sidecar is None:
            sidecar = sidecars.get(_sidecar_key(receipt.batch_id, receipt.layer_name, receipt.output_summary))
        if sidecar:
            for name in ALL_MISSING_FIELDS:
                if name in sidecar:
                    setattr(upgraded, name, copy.deepcopy(sidecar[name]))
            upgraded.sidecar_used = True
    upgraded.format = 3
    for name in ALL_MISSING_FIELDS:
        if _missing(getattr(upgraded, name)):
            setattr(upgraded, name, MISSING)
    upgraded.refresh_completeness()
    if source_format == 1 and not upgraded.sidecar_used:
        upgraded.completeness = {
            "label": "q1_upgraded_partial",
            "missing_fields": upgraded.missing_fields(),
        }
    elif source_format == 2:
        upgraded.completeness = {
            "label": "q2_upgraded_partial",
            "missing_fields": upgraded.missing_fields(),
        }
    return upgraded


def downgrade_q3_to_q1(receipt: Receipt) -> tuple[dict[str, Any], dict[str, Any]]:
    q3 = upgrade_to_q3(receipt)
    sidecar_key = _delivery_sidecar_key(q3)
    q1 = Receipt(
        format=1,
        batch_id=q3.batch_id,
        layer_name=q3.layer_name,
        output_summary=q3.output_summary,
    ).to_q1()
    q1["sidecar_key"] = sidecar_key
    sidecar = {
        "batch_id": q3.batch_id,
        "layer_name": q3.layer_name,
        "output_summary": q3.output_summary,
        "revision": q3.revision,
        "duration_ms": q3.duration_ms,
        "worker_id": q3.worker_id,
        "stable_rank": q3.stable_rank,
        "superseded_refs": copy.deepcopy(q3.superseded_refs),
        "quant_params_summary": copy.deepcopy(q3.quant_params_summary),
        "sidecar_key": sidecar_key,
    }
    return q1, sidecar


def _read_payload(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise QLabError(E_BAD_RECEIPT, f"{path}:{line_no}: invalid JSON line: {exc}") from exc
        return records
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("receipts"), list):
        return parsed["receipts"]
    return [parsed]


def load_sidecars(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("sidecars", []) if isinstance(payload, dict) else payload
    result: dict[str, Any] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        key = item.get("sidecar_key") or _sidecar_key(
            str(item.get("batch_id")), str(item.get("layer_name")), str(item.get("output_summary"))
        )
        result[str(key)] = item
    return result


def load_receipts(paths: Iterable[str | Path], *, sidecar_path: str | Path | None = None) -> tuple[list[Receipt], list[BusinessIssue]]:
    sidecars = load_sidecars(sidecar_path)
    receipts: list[Receipt] = []
    issues: list[BusinessIssue] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            issues.append(issue(E_BAD_RECEIPT, str(path), str(path), "receipt file does not exist"))
            continue
        try:
            records = _read_payload(path)
        except QLabError as exc:
            issues.append(issue(exc.code, str(path), str(path), exc.message))
            continue
        for index, raw in enumerate(records):
            try:
                receipts.append(parse_receipt(raw, sidecars=sidecars, source_file=str(path)))
            except QLabError as exc:
                issues.append(issue(exc.code, f"{path}#{index}", f"{path}[{index}]", exc.message))
    receipts.sort(key=lambda item: (item.batch_id, item.layer_name, item.format, item.receipt_ref()))
    return receipts, issues


def arbitrate_receipts(receipts: list[Receipt]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Receipt]] = {}
    for receipt in receipts:
        groups.setdefault((receipt.batch_id, receipt.layer_name), []).append(receipt)
    decisions: list[dict[str, Any]] = []
    authority_by_batch: dict[str, Any] = {}
    for (batch_id, layer_name), candidates in sorted(groups.items()):
        ordered = sorted(candidates, key=lambda item: item.arbitration_sort_key())
        winner = ordered[0]
        candidate_rows = []
        for candidate in ordered:
            candidate_rows.append(
                {
                    "format": candidate.format,
                    "receipt_ref": candidate.receipt_ref(),
                    "arbitration_values": candidate.arbitration_values(),
                    "completeness": candidate.completeness,
                }
            )
        rationale = {
            "order": [
                "revision desc",
                "duration_ms desc",
                "worker_id asc",
                "receipt format desc (only after the three mandated keys tie)",
                "receipt_ref asc (final deterministic tie-break)",
            ],
            "missing_policy": {
                "revision": "sorts last (effective -1)",
                "duration_ms": "sorts last (effective -1)",
                "worker_id": "sorts last (effective U+FFFF)",
            },
            "winner_key": winner.arbitration_values(),
        }
        decision = {
            "batch_id": batch_id,
            "layer_name": layer_name,
            "winner": winner.to_internal_dict(),
            "winner_ref": winner.receipt_ref(),
            "candidates": candidate_rows,
            "rationale": rationale,
            "decision_digest": digest_json(
                {
                    "batch_id": batch_id,
                    "layer_name": layer_name,
                    "winner_ref": winner.receipt_ref(),
                    "candidate_refs": [candidate.receipt_ref() for candidate in ordered],
                }
            ),
        }
        decisions.append(decision)
        current = authority_by_batch.get(batch_id)
        if current is None or decision["decision_digest"] < current["decision_digest"]:
            authority_by_batch[batch_id] = decision
    return {
        "decisions": decisions,
        "authority": {batch_id: authority_by_batch[batch_id] for batch_id in sorted(authority_by_batch)},
        "candidate_count": len(receipts),
        "decision_count": len(decisions),
    }


def materialize_receipt_versions(
    receipts: Iterable[Receipt | dict[str, Any]],
    directory: str | Path,
) -> dict[str, Any]:
    """Write all three historical formats and the q3->q1 sidecar atomically."""

    normalized: list[Receipt] = []
    for item in receipts:
        if isinstance(item, Receipt):
            normalized.append(upgrade_to_q3(item))
        else:
            normalized.append(parse_receipt(item))
    normalized.sort(key=lambda item: (item.batch_id, item.layer_name, item.receipt_ref()))
    destination = Path(directory)
    q1: list[dict[str, Any]] = []
    q2: list[dict[str, Any]] = []
    q3: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    for receipt in normalized:
        q1_record, sidecar = downgrade_q3_to_q1(receipt)
        q1.append(q1_record)
        q2.append(receipt.to_q2())
        q3.append(receipt.to_q3())
        sidecars.append(sidecar)
    for name, records in (("q1.json", q1), ("q2.json", q2), ("q3.json", q3)):
        atomic_write_json(destination / name, records)
    atomic_write_json(
        destination / "q3_to_q1_sidecar.json",
        {"schema": "q3-to-q1-sidecar-v1", "sidecars": sidecars},
    )
    return {
        "q1": len(q1),
        "q2": len(q2),
        "q3": len(q3),
        "sidecars": len(sidecars),
        "files": [str(destination / name) for name in ("q1.json", "q2.json", "q3.json", "q3_to_q1_sidecar.json")],
    }


def make_receipt(
    *,
    batch_id: str,
    layer_name: str,
    output_summary: str,
    revision: int,
    duration_ms: int,
    worker_id: str,
    stable_rank: int,
    superseded_refs: list[str],
    quant_params_summary: dict[str, Any],
) -> Receipt:
    return Receipt(
        format=3,
        batch_id=batch_id,
        layer_name=layer_name,
        output_summary=output_summary,
        revision=int(revision),
        duration_ms=int(duration_ms),
        worker_id=worker_id,
        stable_rank=int(stable_rank),
        superseded_refs=list(superseded_refs),
        quant_params_summary=copy.deepcopy(quant_params_summary),
    )


def receipt_roundtrip(receipt: Receipt) -> dict[str, Any]:
    """Exercise q3 -> q1 -> q3 with the sidecar and prove conclusion equality."""

    q3 = upgrade_to_q3(receipt)
    q1, sidecar = downgrade_q3_to_q1(q3)
    sidecars = {sidecar["sidecar_key"]: sidecar}
    restored_q1 = parse_receipt(q1, sidecars=sidecars)
    restored = upgrade_to_q3(restored_q1, sidecars=sidecars)
    before_core = q3.semantic_core()
    after_core = restored.semantic_core()
    return {
        "q1": q1,
        "restored_q3": restored.to_q3(),
        "semantic_core_before": before_core,
        "semantic_core_after": after_core,
        "conclusion_unchanged": canonical_json(before_core) == canonical_json(after_core),
        "completeness_before": q3.completeness,
        "completeness_after": restored.completeness,
    }
