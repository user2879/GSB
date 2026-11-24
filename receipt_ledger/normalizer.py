from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import failure as make_failure
from .constants import FIXED_POINT_SCALE, MISSING, Code
from .receipts import build_sidecar, parse_receipt_text, rank_batch_receipts, upgrade_to_r3
from .utils import sha256_json


def _batch_key(batch_id: Any, worker_id: Any) -> tuple[str, str]:
    return (str(batch_id), str(worker_id))


def build_receipt_stage(
    receipt_texts: dict[str, str],
    workers: list[dict[str, Any]],
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    worker_map = {w["worker_id"]: w for w in workers}
    batch_map = {_batch_key(b["batch_id"], b["worker_id"]): b for b in batches}
    batches_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batches:
        batches_by_id[batch["batch_id"]].append(batch)

    for source in sorted(receipt_texts):
        parsed, parse_failures = parse_receipt_text(receipt_texts[source], source=source)
        failures.extend(parse_failures)
        for receipt in parsed:
            batch_id = receipt["batch_id"]
            worker_id = receipt.get("worker_id")
            if not worker_id:
                candidates = batches_by_id.get(batch_id, [])
                if len(candidates) == 1:
                    worker_id = candidates[0]["worker_id"]
                    receipt["worker_id"] = worker_id
                    receipt["worker_id_source"] = "inferred_from_batch_registry"
                    receipt["precision_labels"].append("worker_id:inferred_from_batch_registry")
                elif candidates:
                    worker_id = candidates[0]["worker_id"]
                    receipt["worker_id"] = worker_id
                    receipt["worker_id_source"] = "ambiguous_batch_registry_first"
                    failures.append(make_failure(
                        source, "r1 batch maps to multiple workers; using deterministic first candidate",
                        index=receipt["line_no"], item_id=receipt["receipt_uid"], field="worker_id",
                        code=Code.SCHEMA_INVALID,
                    ))
            key = _batch_key(batch_id, worker_id) if worker_id else None
            batch_def = batch_map.get(key) if key else None
            if worker_id and worker_id not in worker_map:
                receipt["eligible"] = False
                failures.append(make_failure(
                    source, "receipt references unknown worker", index=receipt["line_no"],
                    item_id=receipt["receipt_uid"], field="worker_id", code=Code.SCHEMA_INVALID,
                ))
            elif batch_def and receipt.get("record_version") in {"r2", "r3"} and worker_id != batch_def["worker_id"]:
                receipt["eligible"] = False
                failures.append(make_failure(
                    source, "receipt worker does not match batch registry", index=receipt["line_no"],
                    item_id=receipt["receipt_uid"], field="worker_id", code=Code.SCHEMA_INVALID,
                ))

            if batch_def:
                receipt["dependency_param_version"] = batch_def.get("param_version", 1)
                receipt["batch_size"] = batch_def.get("batch_size")
            else:
                receipt["dependency_param_version"] = receipt.get("param_version") or 0
                receipt["batch_size"] = None

            worker = worker_map.get(worker_id) if worker_id else None
            known_versions = set((worker or {}).get("param_versions", []))
            receipt["eligible"] = worker is not None
            if receipt.get("param_version") is None:
                receipt["param_version_known"] = True
                receipt["param_version_source"] = "missing_on_wire; dependency uses batch registry"
            elif known_versions and receipt["param_version"] not in known_versions:
                receipt["eligible"] = False
                receipt["param_version_known"] = False
                failures.append(make_failure(
                    source, f"unknown param_version {receipt['param_version']} for worker {worker_id}",
                    index=receipt["line_no"], item_id=receipt["receipt_uid"], field="param_version",
                    code=Code.VERSION_UNKNOWN,
                ))
            else:
                receipt["param_version_known"] = True
                receipt["param_version_source"] = "receipt"

            upgraded = upgrade_to_r3(receipt)
            receipt["upgrade_labels"] = {
                "missing_fields": upgraded["upgrade_missing_fields"],
                "precision_labels": upgraded["precision_labels"],
                "completeness_labels": upgraded["completeness_labels"],
            }
            receipt["upgraded_r3_line"] = upgraded.get("r3_line")
            if receipt["record_version"] == "r3":
                receipt["sidecar"] = build_sidecar(receipt)
            receipts.append(receipt)

    receipts.sort(key=lambda item: item["receipt_uid"])
    failures.sort(key=lambda item: (item.get("source", ""), item.get("index") or 0, item.get("code", ""), item.get("message", "")))
    counts = {"r1": 0, "r2": 0, "r3": 0, "total": len(receipts)}
    for receipt in receipts:
        counts[receipt["record_version"]] += 1
    sidecars = [receipt["sidecar"] for receipt in receipts if receipt.get("sidecar")]
    truncated_fields = sorted({field for sidecar in sidecars for field in sidecar["truncated_fields"]})
    return {
        "receipts": receipts,
        "sidecars": sidecars,
        "parse_failures": failures,
        "receipt_counts": counts,
        "r1_to_r3_missing_marker": MISSING,
        "r3_to_r1_truncated_fields": truncated_fields,
    }


def _version_for_decision(candidate: dict[str, Any]) -> int:
    value = candidate.get("param_version")
    if isinstance(value, int):
        return value
    dependency = candidate.get("dependency_param_version")
    return dependency if isinstance(dependency, int) else 0


def _latency_key(candidate: dict[str, Any]) -> tuple[int, int]:
    latency = candidate.get("latency_ms")
    if isinstance(latency, int):
        return (0, latency)
    return (1, 10**12)


def _tie_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -_version_for_decision(candidate),
        candidate.get("batch_id") or "",
        candidate.get("worker_id") or "",
        *_latency_key(candidate),
        candidate.get("output_digest") or "",
        candidate["receipt_uid"],
    )


def _decisive_coordinates(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _version_for_decision(candidate),
        candidate.get("batch_id") or "",
        candidate.get("worker_id") or "",
        *_latency_key(candidate),
    )


def _parse_override_reference(reference: Any) -> tuple[str, str] | None:
    if not isinstance(reference, str) or "@" not in reference:
        return None
    batch_id, worker_id = reference.split("@", 1)
    if not batch_id or not worker_id:
        return None
    return batch_id, worker_id


def attribute_requests(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [receipt for receipt in receipts if receipt.get("eligible", True)]
    by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for receipt in eligible:
        by_request[receipt["request_id"]].append(receipt)

    attributions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for request_id in sorted(by_request):
        candidates = sorted(by_request[request_id], key=lambda item: item["receipt_uid"])
        candidate_uids = [item["receipt_uid"] for item in candidates]
        basis = [
            "aggregate_by=(batch_id,worker_id)",
            "policy=r3_override_reference>param_version_desc>batch_id_asc>latency_ms_asc>receipt_uid_asc",
        ]
        missing_override = [
            item["receipt_uid"] for item in candidates
            if item.get("record_version") == "r3" and item.get("overrides_state") == "missing"
        ]
        target_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            if candidate.get("batch_id") and candidate.get("worker_id"):
                target_index[_batch_key(candidate["batch_id"], candidate["worker_id"])].append(candidate)

        dangling: list[str] = []
        edges: dict[str, tuple[str, str]] = {}
        for candidate in candidates:
            if candidate.get("overrides_state") == "reference":
                target = _parse_override_reference(candidate.get("overrides"))
                if target is None or target not in target_index:
                    dangling.append(candidate["receipt_uid"])
                else:
                    edges[candidate["receipt_uid"]] = target
                    basis.append(
                        f"override:{candidate['receipt_uid']}->{candidate['overrides']} resolves to "
                        + ",".join(target_item["receipt_uid"] for target_item in target_index[target])
                    )

        attribution: dict[str, Any] = {
            "request_id": request_id,
            "status": "attributed",
            "conflict": False,
            "undecidable": False,
            "authoritative_receipt_uid": None,
            "authoritative_batch_id": None,
            "authoritative_worker_id": None,
            "candidate_receipt_uids": candidate_uids,
            "covered_receipts": [],
            "decision_basis": basis,
        }

        if missing_override or dangling:
            attribution["status"] = "undecidable"
            attribution["undecidable"] = True
            attribution["missing_override_receipt_uids"] = sorted(missing_override)
            attribution["dangling_override_receipt_uids"] = sorted(dangling)
            message_parts = []
            if missing_override:
                message_parts.append(f"missing duplicate reference on {','.join(sorted(missing_override))}")
            if dangling:
                message_parts.append(f"dangling duplicate reference on {','.join(sorted(dangling))}")
            failure = make_failure(
                "attribution", "; ".join(message_parts), item_id=request_id,
                code=Code.ATTRIBUTION_UNDECIDABLE,
            )
            failures.append(failure)
            decisions.append({"request_id": request_id, "status": "undecidable", "basis": basis,
                              "candidate_receipt_uids": candidate_uids})
            attributions.append(attribution)
            continue

        winner: dict[str, Any] | None = None
        covered: list[dict[str, Any]] = []
        targeted_uids = {
            target["receipt_uid"]
            for target_key in edges.values()
            for target in target_index.get(target_key, [])
        }
        roots = [candidate for candidate in candidates if candidate["receipt_uid"] not in targeted_uids]
        if edges and len(roots) == 1:
            winner = roots[0]
            basis.append(f"unique_override_root={winner['receipt_uid']}")
            for candidate in candidates:
                if candidate["receipt_uid"] != winner["receipt_uid"]:
                    covered.append({
                        "receipt_uid": candidate["receipt_uid"],
                        "batch_id": candidate.get("batch_id"),
                        "worker_id": candidate.get("worker_id"),
                        "covered_by": winner["receipt_uid"],
                        "reason": "r3_override_reference",
                    })
        else:
            ordered = sorted(candidates, key=_tie_key)
            top = ordered[0]
            runner_up = ordered[1] if len(ordered) > 1 else None
            if runner_up and _decisive_coordinates(top) == _decisive_coordinates(runner_up) \
                    and top.get("output_digest") != runner_up.get("output_digest"):
                attribution["status"] = "conflict"
                attribution["conflict"] = True
                attribution["conflict_receipt_uids"] = [top["receipt_uid"], runner_up["receipt_uid"]]
                basis.append(
                    "unresolved_tie:same_version_batch_worker_latency_with_different_digest"
                )
                failures.append(make_failure(
                    "attribution",
                    "cannot choose between equal-coordinate receipts with different output digests",
                    item_id=request_id, code=Code.ATTRIBUTION_CONFLICT,
                ))
                decisions.append({"request_id": request_id, "status": "conflict", "basis": basis,
                                  "candidate_receipt_uids": candidate_uids})
                attributions.append(attribution)
                continue

            winner = top
            if edges:
                basis.append("multiple_override_roots; applied deterministic tie-break_to_roots_or_candidates")
            basis.append(
                "tie_break:param_version_desc="
                f"{_version_for_decision(winner)},batch_id_asc={winner.get('batch_id')},"
                f"latency_ms_asc={winner.get('latency_ms', MISSING)}"
            )
            for candidate in ordered[1:]:
                same_digest = candidate.get("output_digest") == winner.get("output_digest")
                covered.append({
                    "receipt_uid": candidate["receipt_uid"],
                    "batch_id": candidate.get("batch_id"),
                    "worker_id": candidate.get("worker_id"),
                    "covered_by": winner["receipt_uid"],
                    "reason": "exact_duplicate_same_digest" if same_digest else "deterministic_tie_break",
                })

        attribution.update({
            "status": "attributed",
            "authoritative_receipt_uid": winner["receipt_uid"],
            "authoritative_batch_id": winner.get("batch_id"),
            "authoritative_worker_id": winner.get("worker_id"),
            "authoritative_param_version": _version_for_decision(winner),
            "authoritative_latency_ms": winner.get("latency_ms"),
            "covered_receipts": sorted(covered, key=lambda item: item["receipt_uid"]),
            "decision_basis": basis,
        })
        attributions.append(attribution)
        decisions.append({
            "request_id": request_id,
            "status": "attributed",
            "authoritative_receipt_uid": winner["receipt_uid"],
            "basis": basis,
            "candidate_receipt_uids": candidate_uids,
        })

    attributions.sort(key=lambda item: item["request_id"])
    decisions.sort(key=lambda item: item["request_id"])
    failures.sort(key=lambda item: (item.get("item_id", ""), item.get("code", ""), item.get("message", "")))
    return {"attributions": attributions, "decision_log": decisions, "attribution_failures": failures}


def calculate_resources(
    receipt_stage: dict[str, Any],
    batches: list[dict[str, Any]],
    workers: list[dict[str, Any]],
    rules: dict[str, Any],
    param_digests: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    worker_map = {worker["worker_id"]: worker for worker in workers}
    receipts_by_batch: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipt_stage["receipts"]:
        if receipt.get("batch_id") and receipt.get("worker_id"):
            receipts_by_batch[_batch_key(receipt["batch_id"], receipt["worker_id"])].append(receipt)

    resource_config = rules.get("resource", {}) if rules else {}
    mac_factor = int(resource_config.get("mac_factor", 2))
    version_step_q = int(resource_config.get("version_weight_step_q", 10_000))
    scale = int((rules.get("policy", {}) or {}).get("fixed_point_scale", FIXED_POINT_SCALE))
    param_digests = param_digests or {}
    results: list[dict[str, Any]] = []

    for batch in sorted(batches, key=lambda item: (item["batch_id"], item["worker_id"])):
        key = _batch_key(batch["batch_id"], batch["worker_id"])
        batch_receipts = receipts_by_batch.get(key, [])
        ordered = rank_batch_receipts(batch_receipts)
        eligible_ordered = [item for item in ordered if item.get("eligible", True)]
        seen_requests: set[str] = set()
        ordered_request_ids: list[str] = []
        for receipt in eligible_ordered:
            request_id = receipt["request_id"]
            if request_id not in seen_requests:
                seen_requests.add(request_id)
                ordered_request_ids.append(request_id)
        finite_latencies = [item["latency_ms"] for item in eligible_ordered if isinstance(item.get("latency_ms"), int)]
        latency_ms = max(finite_latencies) if finite_latencies else None
        worker = worker_map.get(batch["worker_id"], {})
        model_dim = int(worker.get("model_dim", 0))
        hidden_dim = int(worker.get("hidden_dim", 0))
        version = int(batch.get("param_version", 1))
        version_weight_q = scale + (version - 1) * version_step_q
        base_mac_per_request = mac_factor * model_dim * hidden_dim
        mac_total = len(ordered_request_ids) * base_mac_per_request * version_weight_q // scale
        utilization_q = len(ordered_request_ids) * scale // int(batch["batch_size"])
        mac_per_ms_q = mac_total * scale // latency_ms if latency_ms and latency_ms > 0 else None
        result = {
            "entry_id": f"batch:{batch['batch_id']}:{batch['worker_id']}",
            "batch_id": batch["batch_id"],
            "worker_id": batch["worker_id"],
            "param_version": version,
            "param_digest": param_digests.get(f"{batch['worker_id']}:{version}"),
            "batch_size": batch["batch_size"],
            "request_count": len(ordered_request_ids),
            "receipt_count": len(batch_receipts),
            "ordered_request_ids": ordered_request_ids,
            "ordered_receipt_uids": [item["receipt_uid"] for item in ordered],
            "latency_ms": latency_ms,
            "latency_state": "present" if latency_ms is not None else "missing",
            "model_dim": model_dim,
            "hidden_dim": hidden_dim,
            "mac_factor": mac_factor,
            "version_weight_q": version_weight_q,
            "mac": mac_total,
            "batch_utilization_q": utilization_q,
            "mac_per_ms_q": mac_per_ms_q,
            "param_digest_input_hash": sha256_json(param_digests),
        }
        results.append(result)
    results.sort(key=lambda item: item["entry_id"])
    return results
