"""Version-normalized receipt ledger and lossless q1/q2/q3 conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import E_RECEIPT_INVALID, FailureList
from .model import Receipt, ReceiptCorpus
from .utils import canonical_json_text, stable_hash


def _reference_matches(reference: str, receipt: Receipt) -> bool:
    if reference == receipt.receipt_id:
        return True
    if reference == receipt.source_file:
        return True
    if reference == Path(receipt.source_file).stem:
        return True
    if reference == receipt.fingerprint:
        return True
    if reference.startswith(receipt.receipt_id):
        return True
    return False


def _candidate_sort_key(
    receipt: Receipt, *, q3_cover: bool
) -> tuple[Any, ...]:
    parameter_version = receipt.parameter_version if receipt.parameter_version is not None else -1
    elapsed = receipt.elapsed_ms if receipt.elapsed_ms is not None else 2**63 - 1
    if q3_cover:
        return (
            elapsed,
            -parameter_version,
            receipt.source_file,
            receipt.source_line,
            receipt.receipt_id,
        )
    return (
        -parameter_version,
        receipt.source_file,
        receipt.source_line,
        elapsed,
        receipt.receipt_id,
    )


def build_receipt_ledger(corpus: ReceiptCorpus) -> dict[str, Any]:
    receipts = sorted(
        corpus.receipts,
        key=lambda item: (item.layer or "", item.source_file, item.source_line, item.receipt_id),
    )
    overridden_by: dict[str, str] = {}
    unresolved: list[dict[str, Any]] = []

    for source in receipts:
        if source.source_version != "q3" or not source.supersedes:
            continue
        for reference in source.supersedes:
            matches = [
                target
                for target in receipts
                if target.receipt_id != source.receipt_id and _reference_matches(reference, target)
            ]
            same_layer = [target for target in matches if target.layer == source.layer]
            selected = same_layer if same_layer else matches
            if not selected:
                unresolved.append(
                    {
                        "source_receipt_id": source.receipt_id,
                        "reference": reference,
                        "reason": "reference_not_found",
                    }
                )
                continue
            for target in selected:
                overridden_by[target.receipt_id] = source.receipt_id

    groups_map: dict[tuple[str, int | None], list[Receipt]] = {}
    layers_map: dict[str, list[Receipt]] = {}
    for receipt in receipts:
        layer = receipt.layer or ""
        layers_map.setdefault(layer, []).append(receipt)
        groups_map.setdefault((layer, receipt.parameter_version), []).append(receipt)

    def elect(candidates: list[Receipt]) -> tuple[Receipt | None, str, list[Receipt]]:
        eligible = [item for item in candidates if item.receipt_id not in overridden_by]
        if not eligible:
            eligible = list(candidates)
        cover = [item for item in eligible if item.source_version == "q3" and item.supersedes]
        if cover:
            winner = min(cover, key=lambda item: _candidate_sort_key(item, q3_cover=True))
            return winner, "q3_supersede_then_elapsed_ms", eligible
        if eligible:
            winner = min(eligible, key=lambda item: _candidate_sort_key(item, q3_cover=False))
            return winner, "parameter_version_desc_then_filename_asc", eligible
        return None, "no_candidate", eligible

    groups: list[dict[str, Any]] = []
    for (layer, parameter_version), candidates in sorted(
        groups_map.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])
    ):
        winner, basis, eligible = elect(candidates)
        groups.append(
            {
                "layer": layer,
                "parameter_version": parameter_version,
                "parameter_version_missing": parameter_version is None,
                "receipt_ids": [item.receipt_id for item in candidates],
                "eligible_receipt_ids": [item.receipt_id for item in eligible],
                "authoritative_receipt_id": winner.receipt_id if winner else None,
                "decision_basis": basis,
                "overridden": [
                    {"receipt_id": item.receipt_id, "overridden_by": overridden_by[item.receipt_id]}
                    for item in candidates
                    if item.receipt_id in overridden_by
                ],
            }
        )

    layer_entries: list[dict[str, Any]] = []
    layer_authority: dict[str, str] = {}
    for layer, candidates in sorted(layers_map.items()):
        winner, basis, eligible = elect(candidates)
        if winner:
            layer_authority[layer] = winner.receipt_id
        layer_entries.append(
            {
                "layer": layer,
                "authoritative_receipt_id": winner.receipt_id if winner else None,
                "decision_basis": basis,
                "candidate_count": len(candidates),
                "eligible_count": len(eligible),
                "overridden_count": sum(1 for item in candidates if item.receipt_id in overridden_by),
            }
        )

    entries: list[dict[str, Any]] = []
    for receipt in receipts:
        authoritative = layer_authority.get(receipt.layer or "") == receipt.receipt_id
        entries.append(
            {
                **receipt.normalized_payload(),
                "authoritative": authoritative,
                "overridden_by": overridden_by.get(receipt.receipt_id),
                "election_scope": "per_layer_with_parameter_version_grouping",
                "receipt_fingerprint": receipt.fingerprint,
            }
        )

    return {
        "schema": "quantcalib-receipt-ledger-v1",
        "receipt_count": len(receipts),
        "source_versions": sorted({item.source_version for item in receipts}),
        "reused_source_versions": sorted(
            {version for version in (entry["source_version"] for entry in entries)}
        ),
        "groups": groups,
        "layers": layer_entries,
        "entries": entries,
        "overridden_receipts": [
            {"receipt_id": receipt_id, "overridden_by": source_id}
            for receipt_id, source_id in sorted(overridden_by.items())
        ],
        "unresolved_supersedes": unresolved,
        "election_rules": {
            "with_q3_cover": "q3 cover references remove overridden candidates; then elapsed_ms asc, parameter_version desc, filename asc",
            "without_q3_cover": "parameter_version desc (missing=-1), filename asc, source line asc, elapsed_ms asc",
            "aggregation_key": "layer_name + parameter_version",
        },
    }


def compute_ledger_cache_key(
    *,
    corpus_fingerprint: str,
    layers: Iterable[dict[str, Any]],
) -> str:
    return stable_hash(
        {
            "stage": "receipt_ledger",
            "stage_version": 1,
            "corpus": corpus_fingerprint,
            "layers": list(layers),
        }
    )


def _classify_version(record: dict[str, Any]) -> str:
    declared = record.get("receipt_version", record.get("schema_version", record.get("qversion")))
    if declared is not None:
        text = str(declared).lower()
        if text in {"1", "q1"}:
            return "q1"
        if text in {"2", "q2"}:
            return "q2"
        if text in {"3", "q3"}:
            return "q3"
    if "group_scale_digest" in record or "supersedes" in record:
        return "q3"
    if "parameter_version" in record or "elapsed_ms" in record:
        return "q2"
    return "q1"


def _layer_and_ratio(record: dict[str, Any]) -> tuple[Any, Any]:
    layer = record.get("layer_name", record.get("layer", record.get("name")))
    ratio = record.get(
        "compression_permille",
        record.get("compression_ratio_permille", record.get("ratio_permille")),
    )
    return layer, ratio


def convert_receipt_file(
    *,
    input_path: Path,
    target_version: str,
    output_path: Path,
    sidecar_path: Path | None = None,
    failures: FailureList | None = None,
) -> dict[str, Any]:
    target = target_version.lower()
    if target not in {"q1", "q2", "q3"}:
        raise ValueError("target_version must be q1, q2 or q3")
    failure_list = failures or FailureList()
    records: list[tuple[int, dict[str, Any]]] = []
    actions: list[dict[str, Any]] = []
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read receipt input: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            failure_list.add(
                E_RECEIPT_INVALID,
                f"invalid receipt JSON: {exc.msg}",
                "receipt_conversion",
                path=str(input_path),
                details={"line": line_number},
            )
            continue
        if not isinstance(record, dict):
            failure_list.add(
                E_RECEIPT_INVALID,
                "receipt must be an object",
                "receipt_conversion",
                path=str(input_path),
                details={"line": line_number},
            )
            continue
        records.append((line_number, record))

    sidecar_entries: list[dict[str, Any]] = []
    restore_entries: dict[int, dict[str, Any]] = {}
    if sidecar_path is not None and sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            for entry in sidecar.get("entries", []):
                output_line = int(entry["output_line"])
                restore_entries[output_line] = entry.get("omitted", {})
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            failure_list.add(
                E_RECEIPT_INVALID,
                f"cannot load receipt sidecar: {exc}",
                "receipt_conversion",
                path=str(sidecar_path),
            )

    output_records: list[dict[str, Any]] = []
    for output_line, (input_line, original) in enumerate(records, 1):
        source = _classify_version(original)
        layer, ratio = _layer_and_ratio(original)
        if not isinstance(layer, str) or not isinstance(ratio, int):
            failure_list.add(
                E_RECEIPT_INVALID,
                "receipt must have a layer name and integer compression permille",
                "receipt_conversion",
                path=str(input_path),
                details={"line": input_line},
            )
            continue
        restored = dict(restore_entries.get(output_line, {}))
        source_fields = {
            "parameter_version": original.get("parameter_version"),
            "elapsed_ms": original.get("elapsed_ms"),
            "group_scale_digest": original.get("group_scale_digest"),
            "supersedes": original.get("supersedes"),
        }
        for key, value in restored.items():
            if value is not None:
                source_fields[key] = value

        target_record: dict[str, Any] = {
            "layer_name": layer,
            "compression_permille": ratio,
        }
        omitted: dict[str, Any] = {}
        if target in {"q2", "q3"}:
            if source_fields["parameter_version"] is not None:
                target_record["parameter_version"] = source_fields["parameter_version"]
            else:
                omitted["parameter_version"] = None
            if source_fields["elapsed_ms"] is not None:
                target_record["elapsed_ms"] = source_fields["elapsed_ms"]
            else:
                omitted["elapsed_ms"] = None
        else:
            if source_fields["parameter_version"] is not None:
                omitted["parameter_version"] = source_fields["parameter_version"]
            if source_fields["elapsed_ms"] is not None:
                omitted["elapsed_ms"] = source_fields["elapsed_ms"]

        if target == "q3":
            if source_fields["group_scale_digest"] is not None:
                target_record["group_scale_digest"] = source_fields["group_scale_digest"]
            else:
                target_record["group_scale_digest"] = None
                omitted["group_scale_digest"] = None
            if source_fields["supersedes"] is not None:
                target_record["supersedes"] = list(source_fields["supersedes"])
            else:
                target_record["supersedes"] = []
                omitted["supersedes"] = []
        else:
            if source_fields["group_scale_digest"] is not None:
                omitted["group_scale_digest"] = source_fields["group_scale_digest"]
            if source_fields["supersedes"] is not None:
                omitted["supersedes"] = list(source_fields["supersedes"])

        output_records.append(target_record)
        if omitted:
            sidecar_entries.append(
                {
                    "output_line": output_line,
                    "input_line": input_line,
                    "source_version": source,
                    "target_version": target,
                    "omitted": omitted,
                }
            )
        action = {
            "output_line": output_line,
            "input_line": input_line,
            "source_version": source,
            "target_version": target,
            "restored_from_sidecar": bool(restored),
            "omitted_fields": sorted(omitted),
        }
        if source != target:
            action["action"] = f"convert_{source}_to_{target}"
        elif restored:
            action["action"] = "restore_in_place"
        else:
            action["action"] = "preserve"
        actions.append(action)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output_records
    )
    output_path.write_text(output_text, encoding="utf-8", newline="\n")
    if sidecar_path is not None:
        sidecar = {
            "schema": "quantcalib-receipt-sidecar-v1",
            "input": input_path.name,
            "target_version": target,
            "entries": sidecar_entries,
        }
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(canonical_json_text(sidecar), encoding="utf-8", newline="\n")

    q3_fields_preserved = any(
        "group_scale_digest" in entry["omitted"] or "supersedes" in entry["omitted"]
        for entry in sidecar_entries
    )
    return {
        "schema": "quantcalib-receipt-conversion-report-v1",
        "input": str(input_path),
        "output": str(output_path),
        "sidecar": str(sidecar_path) if sidecar_path else None,
        "target_version": target,
        "record_count": len(output_records),
        "sidecar_entry_count": len(sidecar_entries),
        "q3_bypass_fields_preserved": q3_fields_preserved,
        "restored_from_sidecar": any(action["restored_from_sidecar"] for action in actions),
        "actions": actions,
        "failure_count": len(failure_list.items),
    }
