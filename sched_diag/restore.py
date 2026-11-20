"""Whole-session snapshot restoration and consistency replay."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_REPLAY
from .manifest import load_manifests
from .persistence import verify_snapshot_payload
from .scheduler import FILTER_ORDER, SCORE_SCALE, TRUNCATION, WorkbenchSession
from .util import canonical_json, first_deep_difference, sha256_digest


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _materials_from_payload(payload: dict[str, Any]) -> tuple[list[Any], list[Any], Any]:
    content = payload["content"]
    required = ("nodes", "pods", "plugins")
    missing = [key for key in required if key not in content]
    if missing:
        raise BusinessError(
            E_REPLAY,
            "snapshot content is missing manifest sections",
            details={"missing": missing},
        )
    try:
        with tempfile.TemporaryDirectory(prefix="k8s-diag-restore-") as temporary:
            root = Path(temporary)
            nodes_path = root / "nodes.json"
            pods_path = root / "pods.json"
            plugins_path = root / "plugins.json"
            _write_json(nodes_path, {"schemaVersion": 1, "nodes": content["nodes"]})
            _write_json(pods_path, {"schemaVersion": 1, "pods": content["pods"]})
            _write_json(plugins_path, {"schemaVersion": 1, **content["plugins"]})
            return load_manifests(nodes_path, pods_path, plugins_path)
    except BusinessError as exc:
        raise BusinessError(
            E_REPLAY,
            "snapshot manifests cannot be revalidated",
            details={"cause": exc.to_dict()},
        ) from None


def session_from_payload(
    payload: Any,
    *,
    source: str = "snapshot",
    verify_digest: bool = True,
) -> WorkbenchSession:
    if verify_digest:
        verify_snapshot_payload(payload, source=source)
    if not isinstance(payload, dict):
        raise BusinessError(E_REPLAY, "snapshot payload is not an object", source)
    nodes, pods, plugins = _materials_from_payload(payload)
    session = WorkbenchSession(nodes, pods, plugins)
    versions = payload["versions"]
    state = payload["state"]

    plugin_content_version = int(payload["content"]["plugins"]["version"])
    if plugin_content_version != versions["plugin"]:
        raise BusinessError(
            E_REPLAY,
            "plugin weight version is inconsistent with the snapshot version block",
            details={
                "basis": "plugin_weight_version_mismatch",
                "contentPluginVersion": plugin_content_version,
                "snapshotPluginVersion": versions["plugin"],
            },
        )
    session.plugins = plugins
    session.node_version = int(versions["node"])
    session.pod_version = int(versions["pod"])
    session.plugin_version = int(versions["plugin"])
    session.graph_version = int(versions["graph"])
    session.fact_revisions = {
        str(key): int(value) for key, value in versions.get("factRevisions", {}).items()
    }
    for fact_id in (node_id for node_id, kind in session.graph.nodes.items() if kind == "fact"):
        session.fact_revisions.setdefault(fact_id, 1)

    pod_ids = [pod.pod_id for pod in session.pods]
    expected_ids = set(pod_ids)
    results: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(state["results"]):
        if not isinstance(result, dict) or not isinstance(result.get("pod"), str):
            raise BusinessError(E_REPLAY, "snapshot contains a malformed pod result", details={"index": index})
        pod_id = result["pod"]
        if pod_id not in expected_ids:
            raise BusinessError(E_REPLAY, f"snapshot result refers to unknown pod '{pod_id}'", details={"pod": pod_id})
        if pod_id in results:
            raise BusinessError(E_REPLAY, f"snapshot contains duplicate result for pod '{pod_id}'", details={"pod": pod_id})
        version = result.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise BusinessError(
                E_REPLAY,
                f"snapshot result for pod '{pod_id}' has no valid conclusion version",
                details={"pod": pod_id, "basis": "missing_conclusion_version"},
            )
        results[pod_id] = result
    if set(results) != expected_ids:
        raise BusinessError(
            E_REPLAY,
            "snapshot does not contain a conclusion for every pod",
            details={"missingPods": sorted(expected_ids - set(results)), "extraPods": sorted(set(results) - expected_ids)},
        )
    session.results = {pod_id: results[pod_id] for pod_id in pod_ids}

    conclusion_revisions = versions.get("conclusionRevisions", {})
    for pod_id in pod_ids:
        recorded = conclusion_revisions.get(pod_id)
        if isinstance(recorded, bool) or not isinstance(recorded, int):
            raise BusinessError(
                E_REPLAY,
                f"snapshot conclusion version for pod '{pod_id}' is missing",
                details={"pod": pod_id, "basis": "missing_conclusion_version"},
            )
        if recorded != results[pod_id]["version"]:
            raise BusinessError(
                E_REPLAY,
                f"snapshot conclusion version for pod '{pod_id}' is inconsistent",
                details={
                    "pod": pod_id,
                    "basis": "conclusion_version_mismatch",
                    "versionBlock": recorded,
                    "result": results[pod_id]["version"],
                },
            )

    stats: dict[str, dict[str, Any]] = {}
    for index, (stat_id, value) in enumerate(sorted(state["stats"].items()), start=1):
        stats[stat_id] = {"value": value, "revision": index}
    session.stats = stats
    session.last_change = state["lastChange"]
    session.last_invalidation = state["lastInvalidation"]
    session.last_recomputed_pods = list(state["lastInvalidation"].get("pods", [])) if isinstance(state["lastInvalidation"], dict) else []
    session.last_recomputed_stats = list(state["lastInvalidation"].get("stats", [])) if isinstance(state["lastInvalidation"], dict) else []
    session._unaffected_check = None
    return session


def _changed_entries(old_content: dict[str, Any] | None, new_content: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    old_nodes = {node["name"]: node for node in (old_content or {}).get("nodes", [])}
    new_nodes = {node["name"]: node for node in new_content.get("nodes", [])}
    changed_nodes = sorted(
        name
        for name in set(old_nodes) | set(new_nodes)
        if canonical_json(old_nodes.get(name)) != canonical_json(new_nodes.get(name))
    )
    old_pods = {pod["id"]: pod for pod in (old_content or {}).get("pods", [])}
    new_pods = {pod["id"]: pod for pod in new_content.get("pods", [])}
    changed_pods = sorted(
        pod_id
        for pod_id in set(old_pods) | set(new_pods)
        if canonical_json(old_pods.get(pod_id)) != canonical_json(new_pods.get(pod_id))
    )
    old_order = [pod["id"] for pod in (old_content or {}).get("pods", [])]
    new_order = [pod["id"] for pod in new_content.get("pods", [])]
    return changed_nodes, changed_pods, old_order != new_order


def restore_payload(
    payload: Any,
    *,
    current_session: WorkbenchSession | None = None,
    source: str = "snapshot",
) -> tuple[WorkbenchSession, dict[str, Any]]:
    """Replace the complete session from a snapshot, never merge incrementally."""
    verify_snapshot_payload(payload, source=source)
    restored = session_from_payload(payload, source=source, verify_digest=False)
    snapshot_content = payload["content"]
    snapshot_versions = {
        key: int(payload["versions"][key]) for key in ("node", "pod", "plugin", "graph")
    }
    current_content = current_session.content_payload() if current_session is not None else None
    current_versions = current_session.versions() if current_session is not None else None
    current_digest = current_session.content_digest() if current_session is not None else None
    snapshot_digest = payload["contentDigest"]
    versions_match = current_versions == snapshot_versions
    digest_match = current_digest == snapshot_digest
    changed_nodes, changed_pods, queue_order_changed = _changed_entries(current_content, snapshot_content)
    graph_signature_match = payload["state"]["graphSignature"] == restored.graph.signature()
    report = {
        "ok": True,
        "source": source,
        "snapshot": payload.get("name"),
        "revision": payload.get("revision"),
        "replaceRequired": not (versions_match and digest_match),
        "replaced": True,
        "versionsMatch": versions_match,
        "digestMatch": digest_match,
        "currentVersions": current_versions,
        "snapshotVersions": snapshot_versions,
        "currentDigest": current_digest,
        "snapshotDigest": snapshot_digest,
        "changedNodes": changed_nodes,
        "changedPods": changed_pods,
        "podQueueOrderChanged": queue_order_changed,
        "graphSignatureMatch": graph_signature_match,
    }
    return restored, report


def _expected_invalidation(payload: dict[str, Any], source: str) -> dict[str, Any]:
    last_change = payload["state"].get("lastChange")
    expected = payload["state"].get("lastInvalidation")
    if not isinstance(last_change, dict) or not isinstance(last_change.get("roots"), list):
        raise BusinessError(
            E_REPLAY,
            "snapshot cannot reproduce invalidation because last-change roots are missing",
            source + "#/state/lastChange",
            {"basis": "missing_invalidation_roots"},
        )
    if not isinstance(expected, dict):
        raise BusinessError(
            E_REPLAY,
            "snapshot cannot reproduce invalidation because the expected set is missing",
            source + "#/state/lastInvalidation",
            {"basis": "missing_expected_invalidation"},
        )
    return {"lastChange": last_change, "expected": expected}


def _invalidation_mapping(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "roots": report.get("roots", []),
        "pods": report.get("pods", []),
        "stats": report.get("stats", []),
        "allIds": report.get("all_ids", []),
        "steps": [
            {
                "id": step.get("id"),
                "kind": step.get("kind"),
                "triggeredBy": step.get("triggered_by", []),
            }
            for step in report.get("steps", [])
        ],
    }


def _compare_all_results(expected: dict[str, dict[str, Any]], actual: WorkbenchSession, pod_order: list[str]) -> None:
    for pod_id in pod_order:
        expected_content = {key: value for key, value in expected[pod_id].items() if key != "version"}
        diff = first_deep_difference(expected_content, actual.result_content(pod_id), "")
        if diff is not None:
            raise BusinessError(
                E_REPLAY,
                f"snapshot replay differs at pod {pod_id}, field {diff['field']}",
                details={"pod": pod_id, "basis": "replay_result_mismatch", **diff},
            )


def _compare_stats(expected_stats: dict[str, Any], actual: WorkbenchSession, pods: list[Any]) -> None:
    actual_stats = {stat_id: cache["value"] for stat_id, cache in actual.stats.items()}
    for stat_id in sorted(set(expected_stats) | set(actual_stats)):
        diff = first_deep_difference(expected_stats.get(stat_id), actual_stats.get(stat_id), "")
        if diff is not None:
            pod_id = _pod_for_stat(stat_id, pods)
            raise BusinessError(
                E_REPLAY,
                f"snapshot replay differs at pod {pod_id}, statistic {stat_id}, field {diff['field']}",
                details={
                    "pod": pod_id,
                    "stat": stat_id,
                    "basis": "replay_stat_mismatch",
                    **diff,
                },
            )


def _pod_for_stat(stat_id: str, pods: list[Any]) -> str:
    for pod in pods:
        if f":{pod.pod_id}:" in stat_id:
            return pod.pod_id
    return pods[0].pod_id if pods else "<none>"


def replay_payload(payload: Any, *, source: str = "snapshot") -> dict[str, Any]:
    """Replay snapshot invalidation on an isolated clone and compare every field."""
    verify_snapshot_payload(payload, source=source)
    session = session_from_payload(payload, source=source, verify_digest=False)
    expected_signature = payload["state"]["graphSignature"]
    actual_signature = session.graph.signature()
    if expected_signature != actual_signature:
        raise BusinessError(
            E_REPLAY,
            "dependency graph cannot be reproduced from the snapshot version",
            details={
                "basis": "dependency_graph_signature_mismatch",
                "expected": expected_signature,
                "actual": actual_signature,
            },
        )
    expected_plugin_weights = payload["content"]["plugins"]["weights"]
    for pod in session.pods:
        result = session.results[pod.pod_id]
        if result.get("plugin_weights") != dict(sorted(expected_plugin_weights.items())):
            raise BusinessError(
                E_REPLAY,
                f"plugin weight basis is inconsistent for pod {pod.pod_id}",
                details={
                    "pod": pod.pod_id,
                    "basis": "plugin_weight_basis_mismatch",
                    "expected": expected_plugin_weights,
                    "actual": result.get("plugin_weights"),
                },
            )
        if result.get("score_scale") != SCORE_SCALE or result.get("truncation") != TRUNCATION:
            raise BusinessError(
                E_REPLAY,
                f"fixed-point score basis is inconsistent for pod {pod.pod_id}",
                details={"pod": pod.pod_id, "basis": "score_basis_mismatch"},
            )
        if result.get("filter_order") != list(FILTER_ORDER):
            raise BusinessError(
                E_REPLAY,
                f"filter order is inconsistent for pod {pod.pod_id}",
                details={"pod": pod.pod_id, "basis": "filter_order_mismatch"},
            )

    invalidation_inputs = _expected_invalidation(payload, source)
    last_change = invalidation_inputs["lastChange"]
    expected_invalidation = invalidation_inputs["expected"]
    source_label = str(last_change.get("source", "snapshot-replay"))
    derived = session.graph.invalidation_report(last_change["roots"], source_label)
    expected_map = _invalidation_mapping(expected_invalidation)
    actual_map = _invalidation_mapping(derived)
    diff = first_deep_difference(expected_map, actual_map, "")
    if diff is not None:
        raise BusinessError(
            E_REPLAY,
            f"snapshot invalidation set differs at {diff['field']}",
            details={"pod": "<invalidation>", "basis": "invalidation_mismatch", **diff},
        )

    expected_results = {result["pod"]: result for result in payload["state"]["results"]}
    pod_order = [pod.pod_id for pod in session.pods]

    fresh = WorkbenchSession.from_data(
        [session.nodes[name] for name in sorted(session.nodes)], session.pods, session.plugins
    )
    fresh.full_schedule()
    _compare_all_results(expected_results, fresh, pod_order)
    _compare_stats(payload["state"]["stats"], fresh, fresh.pods)

    replay_session = session_from_payload(payload, source=source, verify_digest=False)
    affected_pods = set(derived["pods"])
    affected_stats = set(derived["stats"])
    for stat_id in affected_stats:
        replay_session.stats.pop(stat_id, None)
    for pod_id in affected_pods:
        replay_session.results.pop(pod_id, None)
    for pod in replay_session.pods:
        if pod.pod_id in affected_pods:
            replay_session._recompute_pod(pod)
    _compare_all_results(expected_results, replay_session, pod_order)

    return {
        "ok": True,
        "source": source,
        "snapshot": payload.get("name"),
        "revision": payload.get("revision"),
        "graphSignature": actual_signature,
        "invalidation": {
            "source": source_label,
            "roots": derived["roots"],
            "pods": derived["pods"],
            "stats": derived["stats"],
            "steps": len(derived["steps"]),
        },
        "replayedAffectedPods": [pod_id for pod_id in pod_order if pod_id in affected_pods],
        "comparedPods": pod_order,
        "fullRecomputeMatched": True,
        "contentDigest": sha256_digest(payload["content"]),
    }
