from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .arbiter import ArbitrationResult, arbitrate
from .ledger import Ledger
from .spec import PipelineSpec


@dataclass(frozen=True)
class CanonicalCopy:
    digest: str
    references: tuple[str, ...]
    canonical_node: str


def canonical_copies(
    spec: PipelineSpec, summaries: list[dict[str, Any]]
) -> dict[str, CanonicalCopy]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in summaries:
        by_key.setdefault(row["artifact_key"], []).append(row)
    result: dict[str, CanonicalCopy] = {}
    for artifact in sorted(spec.artifacts, key=lambda item: item.key):
        rows = sorted(by_key.get(artifact.key, []), key=lambda row: row["node_id"])
        if not rows:
            result[artifact.key] = CanonicalCopy(
                digest="", references=artifact.references, canonical_node=""
            )
            continue
        row = rows[0]
        result[artifact.key] = CanonicalCopy(
            digest=row["digest"],
            references=tuple(sorted(row["references"])),
            canonical_node=row["node_id"],
        )
    return result


def per_node_digest_map(summaries: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in sorted(summaries, key=lambda item: (item["node_id"], item["artifact_key"])):
        result.setdefault(row["node_id"], {})[row["artifact_key"]] = row["digest"]
    return result


def valid_source_map(ledger: Ledger) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for source in ledger.sources(valid_only=True):
        result.setdefault(source["artifact_key"], []).append(source["node_id"])
    return {key: sorted(set(nodes)) for key, nodes in result.items()}


def equivalent_source_nodes(
    spec: PipelineSpec,
    summaries: list[dict[str, Any]],
    artifact_key: str,
    canonical: CanonicalCopy,
) -> list[str]:
    return sorted(
        row["node_id"]
        for row in summaries
        if row["artifact_key"] == artifact_key
        and row["digest"] == canonical.digest
        and tuple(sorted(row["references"])) == canonical.references
    )


def arbitration_for(
    spec: PipelineSpec,
    ledger: Ledger,
    canonical: CanonicalCopy,
    artifact_key: str,
) -> ArbitrationResult:
    summaries = ledger.summaries()
    digest_map = per_node_digest_map(summaries)
    per_node = {
        node_id: digest_map.get(node_id, {}).get(artifact_key, "")
        for node_id in sorted(spec.node_by_id)
    }
    return arbitrate(
        artifact_key,
        canonical.digest,
        ledger.nodes(),
        per_node,
    )


def reuse_plan(
    spec: PipelineSpec,
    ledger: Ledger,
) -> dict[str, dict[str, Any]]:
    summaries = ledger.summaries()
    canonical = canonical_copies(spec, summaries)
    valid_sources = valid_source_map(ledger)
    decisions: dict[str, dict[str, Any]] = {}
    for layer in spec.topological_layers():
        for key in layer:
            artifact = spec.artifact_by_key[key]
            source_nodes = sorted(
                node_id
                for node_id in valid_sources.get(key, [])
                if canonical[key].digest
                and _source_matches(ledger, node_id, key, canonical[key])
            )
            dependencies_reusable = all(
                decisions[reference]["reusable"] for reference in artifact.references
            )
            reusable = bool(source_nodes) and dependencies_reusable
            decisions[key] = {
                "artifact_key": key,
                "canonical_digest": canonical[key].digest,
                "references": list(artifact.references),
                "source_nodes": source_nodes,
                "reusable": reusable,
                "dependencies_reusable": dependencies_reusable,
            }
    return decisions


def _source_matches(
    ledger: Ledger, node_id: str, artifact_key: str, canonical: CanonicalCopy
) -> bool:
    for row in ledger.sources(valid_only=True):
        if (
            row["node_id"] == node_id
            and row["artifact_key"] == artifact_key
            and row["digest"] == canonical.digest
            and tuple(sorted(row["references"])) == canonical.references
        ):
            return True
    return False


def build_decisions(spec: PipelineSpec, ledger: Ledger) -> list[dict[str, Any]]:
    summaries = ledger.summaries()
    canonical = canonical_copies(spec, summaries)
    valid_sources = valid_source_map(ledger)
    decisions: dict[str, dict[str, Any]] = {}
    for layer in spec.topological_layers():
        for key in layer:
            artifact = spec.artifact_by_key[key]
            source_nodes = sorted(
                node_id
                for node_id in valid_sources.get(key, [])
                if _source_matches(ledger, node_id, key, canonical[key])
            )
            dependencies_reusable = all(
                decisions[reference]["reusable"] for reference in artifact.references
            )
            reusable = bool(source_nodes) and dependencies_reusable
            arbitration = arbitration_for(spec, ledger, canonical[key], key)
            decisions[key] = {
                "artifact_key": key,
                "canonical_digest": canonical[key].digest,
                "references": list(artifact.references),
                "source_nodes": source_nodes,
                "reusable": reusable,
                "dependencies_reusable": dependencies_reusable,
                "selected_node": arbitration.selected_node,
                "verdicts": [verdict.to_dict() for verdict in arbitration.verdicts],
            }
    return [decisions[key] for key in sorted(decisions)]
