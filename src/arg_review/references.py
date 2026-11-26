from __future__ import annotations

from fractions import Fraction
from typing import Any

from .canonical import fingerprint


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, node: str) -> None:
        self.parent.setdefault(node, node)

    def find(self, node: str) -> str:
        self.add(node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: str, right: str) -> str:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            if left_root < right_root:
                self.parent[right_root] = left_root
                return left_root
            self.parent[left_root] = right_root
            return right_root
        return left_root


def ref_key(event_id: str, arg_id: str) -> str:
    return f"{event_id}/{arg_id}"


def build_reference_merges(
    valid_events: list[dict[str, Any]],
    argument_results: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    config: dict[str, Any],
    raw_arg_map: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    threshold = int(config.get("merge", {}).get("dedup_threshold_permille", 650))
    candidates_by_ref: dict[str, dict[str, Any]] = {}
    active_artifact_ids: set[str] = set()
    suppressed_artifact_ids: set[str] = set()

    for event_id, result in argument_results.items():
        for candidate in result.get("candidates", []):
            candidates_by_ref[ref_key(event_id, candidate["arg_id"])] = candidate
    for event_id, decision in decisions.items():
        active_artifact_ids.update(decision.get("accepted_candidate_ids", []))
        for suppressed in decision.get("suppressed_candidates", []):
            suppressed_artifact_ids.add(suppressed["candidate_id"])

    relations: list[dict[str, Any]] = []
    for event in valid_events:
        for relation_item in event.get("relations", []):
            source = relation_item["source"]
            target = relation_item["target"]
            source_ref = ref_key(source["event_id"], source["arg_id"])
            target_ref = ref_key(target["event_id"], target["arg_id"])
            source_arg = raw_arg_map.get((source["event_id"], source["arg_id"]), {})
            target_arg = raw_arg_map.get((target["event_id"], target["arg_id"]), {})
            relations.append(
                {
                    "relation_id": relation_item["relation_id"],
                    "kind": relation_item.get("kind", "unspecified"),
                    "source_ref": source_ref,
                    "target_ref": target_ref,
                    "source_event": source["event_id"],
                    "target_event": target["event_id"],
                    "source_start": source_arg.get("start"),
                    "target_start": target_arg.get("start"),
                    "injection_id": relation_item.get("_injection_id"),
                }
            )

    def order_key(item: dict[str, Any]) -> tuple:
        return (
            str(item["source_event"]),
            item["source_start"] if isinstance(item["source_start"], int) else 10**12,
            item["source_ref"],
            str(item["target_event"]),
            item["target_start"] if isinstance(item["target_start"], int) else 10**12,
            item["target_ref"],
            item["relation_id"],
        )

    relations.sort(key=order_key)
    union_find = UnionFind()
    edge_decisions: list[dict[str, Any]] = []
    accepted_edges: list[dict[str, Any]] = []

    for processing_order, relation_item in enumerate(relations):
        source_ref = relation_item["source_ref"]
        target_ref = relation_item["target_ref"]
        union_find.add(source_ref)
        union_find.add(target_ref)
        source_candidate = candidates_by_ref.get(source_ref)
        target_candidate = candidates_by_ref.get(target_ref)
        edge = {
            "processing_order": processing_order,
            "relation_id": relation_item["relation_id"],
            "kind": relation_item["kind"],
            "source": source_ref,
            "target": target_ref,
            "source_event_id": relation_item["source_event"],
            "target_event_id": relation_item["target_event"],
            "injection_id": relation_item.get("injection_id"),
            "dedup_threshold_permille": threshold,
        }

        if source_candidate is None or target_candidate is None:
            edge.update(
                status="fault",
                accepted=False,
                business_code="E_SCHEMA_INVALID",
                stage_code="STAGE_REFERENCE_ENDPOINT_MISSING",
                reason="relation_endpoint_has_no_reviewed_argument",
            )
        else:
            source_id = source_candidate["artifact_id"]
            target_id = target_candidate["artifact_id"]
            source_score = Fraction(
                source_candidate["quantized_score_numerator"],
                source_candidate["quantized_score_denominator"],
            )
            target_score = Fraction(
                target_candidate["quantized_score_numerator"],
                target_candidate["quantized_score_denominator"],
            )
            edge["source_score_fixed"] = source_candidate["score_fixed"]
            edge["target_score_fixed"] = target_candidate["score_fixed"]
            edge["edge_score_permille"] = int(min(source_score, target_score) * 1000)
            edge["argument_cache_keys"] = [
                source_candidate["cache_key"],
                target_candidate["cache_key"],
            ]

            if source_id in suppressed_artifact_ids or target_id in suppressed_artifact_ids:
                edge.update(
                    status="rejected",
                    accepted=False,
                    business_code="E_REF_MUTEX_SUPPRESSED",
                    reason="endpoint_was_suppressed_by_mutex_decision",
                )
            elif not source_candidate.get("eligible") or not target_candidate.get("eligible"):
                edge.update(
                    status="rejected",
                    accepted=False,
                    business_code="E_REF_INELIGIBLE",
                    reason="endpoint_failed_fragment_or_neighborhood_rule",
                )
            elif source_id not in active_artifact_ids or target_id not in active_artifact_ids:
                edge.update(
                    status="rejected",
                    accepted=False,
                    business_code="E_REF_BELOW_ACCEPT_THRESHOLD",
                    reason="endpoint_not_in_accepted_decision_set",
                )
            elif min(source_score, target_score) < Fraction(threshold, 1000):
                edge.update(
                    status="rejected",
                    accepted=False,
                    business_code="E_DEDUP_BELOW_THRESHOLD",
                    reason="minimum_endpoint_confidence_below_dedup_threshold",
                )
            else:
                root = union_find.union(source_ref, target_ref)
                edge.update(
                    status="accepted",
                    accepted=True,
                    business_code="E_REF_MERGE_EDGE_ACCEPTED",
                    reason="explicit_relation_and_threshold_passed",
                    union_root=root,
                )
                accepted_edges.append(edge)
        edge_decisions.append(edge)

    components: dict[str, list[str]] = {}
    known_refs = set(candidates_by_ref) | {
        endpoint
        for edge in edge_decisions
        for endpoint in (edge["source"], edge["target"])
    }
    for node in known_refs:
        root = union_find.find(node)
        components.setdefault(root, []).append(node)

    ref_metadata: dict[str, dict[str, Any]] = {}
    for ref, candidate in candidates_by_ref.items():
        event_id, arg_id = ref.split("/", 1)
        ref_metadata[ref] = {
            "event_id": event_id,
            "arg_id": arg_id,
            "start": candidate["start"],
            "end": candidate["end"],
            "artifact_id": candidate["artifact_id"],
            "cache_key": candidate["cache_key"],
        }
    for (event_id, arg_id), raw_arg in raw_arg_map.items():
        ref = ref_key(event_id, arg_id)
        ref_metadata.setdefault(
            ref,
            {
                "event_id": event_id,
                "arg_id": arg_id,
                "start": raw_arg.get("start", 10**12),
                "end": raw_arg.get("end", 10**12),
                "artifact_id": f"arg:{event_id}:{arg_id}",
                "cache_key": None,
            },
        )

    merge_groups: list[dict[str, Any]] = []
    for members in components.values():
        if len(members) < 2:
            continue
        ordered_members = sorted(
            members,
            key=lambda ref: (
                ref_metadata.get(ref, {}).get("event_id", ref.split("/")[0]),
                ref_metadata.get(ref, {}).get("start", 10**12),
                ref_metadata.get(ref, {}).get("end", 10**12),
                ref,
            ),
        )
        member_metadata = [ref_metadata[ref] for ref in ordered_members if ref in ref_metadata]
        source_events = sorted({item["event_id"] for item in member_metadata})
        member_set = set(ordered_members)
        group_edges = [
            edge["relation_id"]
            for edge in accepted_edges
            if edge["source"] in member_set and edge["target"] in member_set
        ]
        argument_keys = sorted(
            item["cache_key"] for item in member_metadata if item.get("cache_key")
        )
        relation_fps = sorted(
            fingerprint(
                {
                    "relation_id": edge["relation_id"],
                    "kind": edge["kind"],
                    "source": edge["source"],
                    "target": edge["target"],
                },
                16,
            )
            for edge in edge_decisions
            if edge["relation_id"] in group_edges
        )
        rule_fp = fingerprint(
            {"dedup_threshold_permille": threshold, "rule": "explicit_relation_union_v1"},
            16,
        )
        artifact_fingerprint = fingerprint(
            {"members": ordered_members, "argument_keys": argument_keys}, 16
        )
        cache_key = fingerprint(
            {
                "artifact": "reference_merge",
                "members": ordered_members,
                "argument_cache_keys": argument_keys,
                "relation_fingerprints": relation_fps,
                "merge_rule_fingerprint": rule_fp,
            }
        )
        merge_groups.append(
            {
                "artifact_id": f"merge:{artifact_fingerprint}",
                "cache_key": cache_key,
                "stable_content_id": artifact_fingerprint,
                "members": ordered_members,
                "arguments": member_metadata,
                "argument_ids": [item["artifact_id"] for item in member_metadata],
                "argument_cache_keys": argument_keys,
                "source_event_ids": source_events,
                "relation_ids": sorted(group_edges),
                "relation_fingerprints": relation_fps,
                "dedup_threshold_permille": threshold,
                "rule_pack_fingerprint": rule_fp,
                "basis": [
                    "stable_order:event_id,arg_start,arg_id",
                    "union_find_over_accepted_explicit_relations",
                    f"dedup_threshold_permille={threshold}",
                ],
            }
        )

    merge_groups.sort(
        key=lambda group: (
            group["arguments"][0]["event_id"] if group["arguments"] else "",
            group["arguments"][0]["start"] if group["arguments"] else 10**12,
            group["artifact_id"],
        )
    )
    for index, group in enumerate(merge_groups, 1):
        group["merge_id"] = f"M{index:03d}"

    return {
        "artifact_id": "references:all",
        "edge_decisions": edge_decisions,
        "merge_groups": merge_groups,
        "merge_count": len(merge_groups),
        "accepted_edge_count": len(accepted_edges),
        "rejected_edge_count": sum(1 for edge in edge_decisions if not edge["accepted"]),
    }
