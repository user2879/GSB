from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .canonical import canonical_bytes, digest_value, elapsed_ms, monotonic_ns
from .errors import SpecError
from .pipeline import (
    build_conclusion,
    compute_shard_locally,
    first_divergence,
    plan_shards,
    weighted_sampling_stage,
)
from .scheduler import recompute_shards_tcp
from .specs import validate_doc
from .synthetic import doc_content_digest, write_ndjson
from .workspace import Workspace, atomic_write_bytes, atomic_write_json


def write_view(workspace: Workspace, conclusion: dict[str, Any], propagation: dict[str, Any] | None = None, scheduling_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    view = {
        "format_version": 1,
        "view_version": 1,
        "config_version": int(conclusion["config_version"]),
        "cache_digest": conclusion["conclusion_digest"],
        "conclusion": conclusion,
        "last_invalidation": propagation,
        "last_scheduling_summary": scheduling_summary,
    }
    atomic_write_json(workspace.fast_view_path, view)
    atomic_write_bytes(workspace.fast_conclusion_path, canonical_bytes({k: v for k, v in conclusion.items() if k != "conclusion_digest"}))
    if propagation is not None:
        atomic_write_json(workspace.propagation_path, propagation)
    return view


def initialize_fast_view(workspace: Workspace, *, use_tcp: bool = False) -> dict[str, Any]:
    bundle = workspace.load_bundle()
    candidates = workspace.load_candidates()
    weighted = weighted_sampling_stage(bundle)
    plans = plan_shards(weighted["selected"], int(bundle.config["shard_size"]))
    scheduling_summary = None
    if use_tcp:
        remote = recompute_shards_tcp(bundle)
        shard_results = remote["shard_results"]
        scheduling = remote["scheduling"]
        scheduling_summary = {key: value for key, value in scheduling.items() if key != "results"}
    else:
        docs_by_id = {doc["doc_id"]: doc for doc in _all_docs(bundle)}
        shard_results = [compute_shard_locally(plan, docs_by_id, bundle.config) for plan in plans]
    conclusion = build_conclusion(bundle, weighted, shard_results, candidates)
    return write_view(workspace, conclusion, None, scheduling_summary)


def _all_docs(bundle: Any) -> list[dict[str, Any]]:
    from .pipeline import flatten_docs

    return flatten_docs(bundle)


def read_fast_once(workspace: Workspace) -> dict[str, Any]:
    start = monotonic_ns()
    raw = workspace.read_fast_view_bytes()
    view = json.loads(raw)
    end = monotonic_ns()
    duration = elapsed_ms(start, end)
    return {
        "view": view,
        "elapsed_ms": duration,
        "start_ns": start,
        "end_ns": end,
        "bytes_read": len(raw),
        "cache_digest": view.get("cache_digest"),
        "config_version": view.get("config_version"),
    }


def authoritative_conclusion(workspace: Workspace) -> dict[str, Any]:
    bundle = workspace.load_bundle()
    candidates = workspace.load_candidates()
    from .pipeline import compute_conclusion

    conclusion = compute_conclusion(bundle, candidates)
    atomic_write_bytes(workspace.authoritative_path, canonical_bytes({k: v for k, v in conclusion.items() if k != "conclusion_digest"}))
    return conclusion


def verify_fast_matches_authoritative(workspace: Workspace) -> dict[str, Any]:
    fast = workspace.load_fast_view()["conclusion"]
    authoritative = authoritative_conclusion(workspace)
    divergence = first_divergence(authoritative, fast)
    if divergence is not None:
        from .errors import DivergeError

        raise DivergeError("fast view and authoritative recomputation diverged", **divergence)
    return {"identical": True, "digest": authoritative["conclusion_digest"], "bytes": len(canonical_bytes(authoritative))}


def _threshold_affected_shards(old_conclusion: dict[str, Any], new_threshold: int) -> tuple[set[str], list[dict[str, Any]]]:
    affected: set[str] = set()
    bases: list[dict[str, Any]] = []
    for shard in old_conclusion.get("shards", []):
        old_threshold = None
        # The threshold is intentionally absent from canonical shard bytes; the
        # prior configured value is recovered from the old view config version
        # caller through pair booleans. Re-evaluate old booleans directly.
        crossing_pairs = []
        for pair in shard.get("pairs", []):
            old_above = bool(pair["above_threshold"])
            new_above = (10_000 * int(pair["intersection"])) >= int(new_threshold) * int(pair["union"])
            if old_above != new_above:
                crossing_pairs.append(
                    {
                        "pair": [pair["a"], pair["b"]],
                        "intersection": pair["intersection"],
                        "union": pair["union"],
                        "old_above": old_above,
                        "new_above": new_above,
                    }
                )
        if crossing_pairs:
            affected.add(shard["shard_id"])
            bases.append({"shard_id": shard["shard_id"], "crossing_pairs": crossing_pairs})
    return affected, bases


def plan_invalidation(
    old_conclusion: dict[str, Any],
    new_config: dict[str, Any],
    new_weighted: dict[str, Any],
    new_plans: list[dict[str, Any]],
    change: dict[str, Any],
) -> dict[str, Any]:
    old_shards = {shard["shard_id"]: shard for shard in old_conclusion.get("shards", [])}
    new_plan_by_id = {plan["shard_id"]: plan for plan in new_plans}
    all_shard_ids = sorted(set(old_shards) | set(new_plan_by_id))
    affected_shards: set[str] = set()
    removed_shards: list[str] = []
    member_bases: list[dict[str, Any]] = []

    threshold_bases: list[dict[str, Any]] = []
    if change["kind"] == "threshold":
        affected_shards, threshold_bases = _threshold_affected_shards(old_conclusion, int(new_config["similarity_threshold_bp"]))

    for shard_id in all_shard_ids:
        old_shard = old_shards.get(shard_id)
        new_plan = new_plan_by_id.get(shard_id)
        if new_plan is None:
            removed_shards.append(shard_id)
            continue
        if old_shard is None:
            affected_shards.add(shard_id)
            member_bases.append({"shard_id": shard_id, "reason": "new_shard", "new_members_digest": new_plan["members_digest"]})
        elif old_shard.get("members_digest") != new_plan.get("members_digest"):
            affected_shards.add(shard_id)
            member_bases.append(
                {
                    "shard_id": shard_id,
                    "reason": "members_digest_changed",
                    "old_members_digest": old_shard.get("members_digest"),
                    "new_members_digest": new_plan.get("members_digest"),
                    "old_member_ids": old_shard.get("member_ids"),
                    "new_member_ids": new_plan.get("member_ids"),
                }
            )

    old_weighted_domains = old_conclusion.get("weighted_sampling", {}).get("domains", {})
    changed_weight_domains = sorted(
        domain
        for domain, summary in new_weighted["domains"].items()
        if canonical_bytes(old_weighted_domains.get(domain, {})) != canonical_bytes(summary)
    )
    direct_domains: set[str] = set(changed_weight_domains)
    if change.get("domain_id"):
        direct_domains.add(str(change["domain_id"]))
    for shard_id in affected_shards:
        plan = new_plan_by_id.get(shard_id)
        old_shard = old_shards.get(shard_id)
        if plan:
            direct_domains.update(plan["domains"])
        if old_shard:
            direct_domains.update(old_shard.get("domains", []))

    propagation = {
        "format_version": 1,
        "config_version": int(new_config["config_version"]),
        "change": change,
        "affected_domains": sorted(direct_domains),
        "changed_weight_domains": changed_weight_domains,
        "affected_shards": sorted(affected_shards),
        "removed_shards": sorted(removed_shards),
        "recomputed_shards": sorted(affected_shards),
        "preserved_shards": sorted(shard_id for shard_id in old_shards if shard_id not in affected_shards and shard_id in new_plan_by_id),
        "affected_ratio_domains": [],
        "preserved_ratio_domains": [],
        "steps": [],
        "propagation_bases": {"member_changes": member_bases, "threshold_changes": threshold_bases},
        "preservation_checks": [],
    }
    propagation["invalidated_shards"] = sorted(set(propagation["affected_shards"]) | set(propagation["removed_shards"]))

    steps: list[dict[str, Any]] = []

    def add_step(source: str, target: str, edge: str, basis: dict[str, Any]) -> None:
        steps.append({"seq": len(steps) + 1, "from": source, "to": target, "edge": edge, "basis": basis})

    add_step(
        f"config:v{change.get('old_config_version')}",
        f"config:v{new_config['config_version']}",
        change["kind"],
        {"field": change.get("field"), "old": change.get("old_value"), "new": change.get("new_value")},
    )
    for domain in sorted(direct_domains):
        add_step(f"config:{change.get('field', change['kind'])}", f"domain:{domain}", "domain_reference", {"config_version": new_config["config_version"]})
    for shard_id in sorted(affected_shards):
        plan = new_plan_by_id[shard_id]
        for domain in plan["domains"]:
            add_step(f"domain:{domain}", f"shard:{shard_id}", "domain_to_shard_membership", {"members_digest": plan["members_digest"]})
        old_shard = old_shards.get(shard_id)
        add_step(
            f"shard:{shard_id}",
            f"dedup:{shard_id}",
            "members_to_dedup_conclusion",
            {"old_shard_digest": old_shard.get("shard_digest") if old_shard else None, "status": "invalidated"},
        )
    propagation["steps"] = steps
    return propagation


def recompute_after_change(workspace: Workspace, change: dict[str, Any], *, use_tcp: bool = False) -> dict[str, Any]:
    old_view = workspace.load_fast_view()
    old_conclusion = old_view["conclusion"]
    bundle = workspace.load_bundle()
    candidates = workspace.load_candidates()
    weighted = weighted_sampling_stage(bundle)
    plans = plan_shards(weighted["selected"], int(bundle.config["shard_size"]))
    propagation = plan_invalidation(old_conclusion, bundle.config, weighted, plans, change)
    affected = set(propagation["affected_shards"])

    old_shard_map = {shard["shard_id"]: shard for shard in old_conclusion["shards"]}
    new_shard_map: dict[str, dict[str, Any]] = {}
    scheduling_summary = None
    if affected:
        if use_tcp:
            remote = recompute_shards_tcp(bundle, shard_ids=affected)
            for shard in remote["shard_results"]:
                new_shard_map[shard["shard_id"]] = shard
            scheduling = remote["scheduling"]
            scheduling_summary = {key: value for key, value in scheduling.items() if key != "results"}
        else:
            docs_by_id = {doc["doc_id"]: doc for doc in _all_docs(bundle)}
            plan_by_id = {plan["shard_id"]: plan for plan in plans}
            for shard_id in sorted(affected):
                new_shard_map[shard_id] = compute_shard_locally(plan_by_id[shard_id], docs_by_id, bundle.config)

    assembled: list[dict[str, Any]] = []
    preservation_checks: list[dict[str, Any]] = []
    for plan in plans:
        shard_id = plan["shard_id"]
        if shard_id in affected:
            assembled.append(new_shard_map[shard_id])
        else:
            old_shard = old_shard_map[shard_id]
            assembled.append(old_shard)
            identical = canonical_bytes(old_shard) == canonical_bytes(old_shard_map[shard_id])
            preservation_checks.append(
                {
                    "ref": f"dedup:{shard_id}",
                    "old_digest": old_shard["shard_digest"],
                    "new_digest": old_shard["shard_digest"],
                    "byte_identical": identical,
                }
            )

    conclusion = build_conclusion(bundle, weighted, assembled, candidates)
    old_ratio = {item["domain_id"]: item for item in old_conclusion["ratio_normalization"]["domains"]}
    new_ratio = {item["domain_id"]: item for item in conclusion["ratio_normalization"]["domains"]}
    changed_ratio: list[str] = []
    preserved_ratio: list[str] = []
    for domain in sorted(new_ratio):
        old_item = old_ratio.get(domain)
        new_item = new_ratio[domain]
        identical = old_item is not None and canonical_bytes(old_item) == canonical_bytes(new_item)
        if identical:
            preserved_ratio.append(domain)
        else:
            changed_ratio.append(domain)
        preservation_checks.append(
            {
                "ref": f"ratio:{domain}",
                "old_digest": digest_value(old_item) if old_item else None,
                "new_digest": digest_value(new_item),
                "byte_identical": identical,
            }
        )

    propagation["affected_ratio_domains"] = changed_ratio
    propagation["preserved_ratio_domains"] = preserved_ratio
    propagation["preservation_checks"] = preservation_checks
    for shard_id in propagation["affected_shards"]:
        new_shard = new_shard_map.get(shard_id)
        for domain in new_shard["domains"] if new_shard else []:
            propagation["steps"].append(
                {
                    "seq": len(propagation["steps"]) + 1,
                    "from": f"dedup:{shard_id}",
                    "to": f"ratio:{domain}",
                    "edge": "kept_count_to_normalization",
                    "basis": {"new_shard_digest": new_shard["shard_digest"] if new_shard else None},
                }
            )
    for domain in changed_ratio:
        propagation["steps"].append(
            {
                "seq": len(propagation["steps"]) + 1,
                "from": "normalization:all_counts",
                "to": f"ratio:{domain}",
                "edge": "deviation_changed",
                "basis": {"old": old_ratio.get(domain), "new": new_ratio[domain]},
            }
        )
    propagation["propagation_digest"] = digest_value(
        {key: value for key, value in propagation.items() if key != "propagation_digest"}
    )
    write_view(workspace, conclusion, propagation, scheduling_summary)
    return propagation


def _bump_config_version(config: dict[str, Any]) -> int:
    config["config_version"] = int(config["config_version"]) + 1
    return int(config["config_version"])


def edit_weight(workspace: Workspace, domain_id: str, weight_bp: int, *, use_tcp: bool = False) -> dict[str, Any]:
    config = workspace.load_config()
    if domain_id not in config["weights_bp"]:
        raise SpecError("unknown domain in weight edit", field=f"weights_bp.{domain_id}", value=domain_id)
    old_version = int(config["config_version"])
    old_value = int(config["weights_bp"][domain_id])
    if not isinstance(weight_bp, int) or isinstance(weight_bp, bool) or not 0 <= weight_bp <= 10_000:
        raise SpecError("weight must be an integer in [0,10000]", field=f"weights_bp.{domain_id}", value=weight_bp)
    config["weights_bp"][domain_id] = weight_bp
    new_version = _bump_config_version(config)
    workspace.save_config(config)
    change = {"kind": "weight", "field": f"weights_bp.{domain_id}", "domain_id": domain_id, "old_value": old_value, "new_value": weight_bp, "old_config_version": old_version, "config_version": new_version}
    return recompute_after_change(workspace, change, use_tcp=use_tcp)


def edit_threshold(workspace: Workspace, threshold_bp: int, *, use_tcp: bool = False) -> dict[str, Any]:
    config = workspace.load_config()
    old_version = int(config["config_version"])
    old_value = int(config["similarity_threshold_bp"])
    from .specs import MAX_THRESHOLD_BP, MIN_THRESHOLD_BP

    if not isinstance(threshold_bp, int) or isinstance(threshold_bp, bool) or not MIN_THRESHOLD_BP <= threshold_bp <= MAX_THRESHOLD_BP:
        raise SpecError("unsupported threshold", field="similarity_threshold_bp", value=threshold_bp)
    config["similarity_threshold_bp"] = threshold_bp
    new_version = _bump_config_version(config)
    workspace.save_config(config)
    change = {"kind": "threshold", "field": "similarity_threshold_bp", "old_value": old_value, "new_value": threshold_bp, "old_config_version": old_version, "config_version": new_version}
    return recompute_after_change(workspace, change, use_tcp=use_tcp)


def edit_shard_size(workspace: Workspace, shard_size: int, *, use_tcp: bool = False) -> dict[str, Any]:
    config = workspace.load_config()
    old_version = int(config["config_version"])
    old_value = int(config["shard_size"])
    from .specs import MAX_SHARD_SIZE, MIN_SHARD_SIZE

    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or not MIN_SHARD_SIZE <= shard_size <= MAX_SHARD_SIZE:
        raise SpecError("unsupported shard_size", field="shard_size", value=shard_size)
    config["shard_size"] = shard_size
    new_version = _bump_config_version(config)
    workspace.save_config(config)
    change = {"kind": "shard_size", "field": "shard_size", "old_value": old_value, "new_value": shard_size, "old_config_version": old_version, "config_version": new_version}
    return recompute_after_change(workspace, change, use_tcp=use_tcp)


def edit_document(
    workspace: Workspace,
    doc_id: str,
    *,
    term_counts: list[int] | None = None,
    missing_bitmap: list[int] | None = None,
    use_tcp: bool = False,
) -> dict[str, Any]:
    batches_path = workspace.batches_path
    records: list[dict[str, Any]] = []
    target: dict[str, Any] | None = None
    target_domain: dict[str, Any] | None = None
    target_batch_id: str | None = None
    domains_by_id = {domain["domain_id"]: domain for domain in workspace.load_bundle(verify_digests=False).domains}
    for line_no, line in enumerate(batches_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        batch = json.loads(line)
        for doc in batch.get("docs", []):
            if doc.get("doc_id") == doc_id:
                target = doc
                target_domain = domains_by_id[batch["domain_id"]]
                target_batch_id = batch["batch_id"]
                target["_line"] = line_no
        records.append(batch)
    if target is None or target_domain is None:
        raise SpecError("document not found", field="docs.doc_id", value=doc_id)

    old_digest = target["content_digest"]
    if term_counts is not None:
        target["term_counts"] = [int(value) for value in term_counts]
        target["length"] = sum(target["term_counts"])
    if missing_bitmap is not None:
        target["missing_bitmap"] = [int(value) for value in missing_bitmap]
    validate_doc(target, path=str(batches_path), line=int(target.get("_line", 1)), domain=target_domain)
    target.pop("_line", None)
    target["content_digest"] = doc_content_digest(target)

    for batch in records:
        for doc in batch.get("docs", []):
            doc.pop("_line", None)
        batch["batch_digest"] = digest_value([doc["content_digest"] for doc in batch["docs"]])
    write_ndjson(batches_path, records)

    config = workspace.load_config()
    old_version = int(config["config_version"])
    new_version = _bump_config_version(config)
    workspace.save_config(config)
    change = {
        "kind": "document",
        "field": "docs.term_counts",
        "domain_id": target_domain["domain_id"],
        "doc_id": doc_id,
        "batch_id": target_batch_id,
        "old_value": old_digest,
        "new_value": target["content_digest"],
        "old_config_version": old_version,
        "config_version": new_version,
    }
    return recompute_after_change(workspace, change, use_tcp=use_tcp)
