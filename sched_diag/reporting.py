"""Stable human-readable diagnostic rendering shared by CLI and demo."""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable

from .scheduler import FILTER_ORDER, WorkbenchSession
from .util import canonical_json


PrintFn = Callable[[str], None]


def _print(printer: PrintFn, text: str = "") -> None:
    printer(text)


def format_score_line(node: str, score: dict[str, Any], error_bound: int) -> str:
    raw = score["raw"]
    contribution = score["weighted_contrib_milli"]
    raw_text = ",".join(f"{name}={raw[name]}" for name in sorted(raw))
    contribution_text = ",".join(
        f"{name}={contribution[name]}" for name in sorted(contribution)
    )
    return (
        f"    node={node} total_milli={score['total_milli']} error_bound_milli={error_bound} "
        f"raw[{raw_text}] contrib_milli[{contribution_text}]"
    )


def pod_lines(session: WorkbenchSession, pod_id: str, *, include_all_scores: bool = True) -> list[str]:
    result = session.results[pod_id]
    selected = result.get("selected")
    lines = [
        (
            f"  pod={pod_id} version={result['version']} selected={selected or '<none>'} "
            f"candidates={','.join(result['candidates']) or '<none>'} "
            f"score_milli={result.get('selected_score_milli')} error_bound_milli={result.get('error_bound_milli')}"
        )
    ]
    for filtered in result["filtered"]:
        lines.append(
            f"    reject node={filtered['node']} order={filtered['stage']} "
            f"stage={filtered['stage_name']} reason={filtered['reason']}"
        )
    nodes = result["scores"] if include_all_scores else (
        {selected: result["scores"][selected]} if selected in result["scores"] else {}
    )
    for node in sorted(nodes):
        lines.append(format_score_line(node, nodes[node], result["error_bound_milli"]))
    if selected is None:
        lines.append(f"    unschedulable={result['unschedulable_reason']}")
    else:
        lines.append(
            f"    tie_break={canonical_json(result['tie_break'])} truncation={result['truncation']}"
        )
    return lines


def session_lines(
    session: WorkbenchSession,
    *,
    title: str,
    include_all_scores: bool = True,
) -> list[str]:
    lines = [
        title,
        (
            f"  versions node={session.node_version} pod={session.pod_version} "
            f"plugin={session.plugin_version} graph={session.graph_version} "
            f"digest={session.content_digest()}"
        ),
        (
            f"  graph nodes={len(session.graph.nodes)} "
            f"edges={sum(len(v) for v in session.graph.edges.values())} "
            f"signature={session.graph.signature()}"
        ),
        "  FILTER_ORDER=" + " > ".join(FILTER_ORDER),
    ]
    for pod in session.pods:
        lines.extend(pod_lines(session, pod.pod_id, include_all_scores=include_all_scores))
    return lines


def invalidation_lines(report: dict[str, Any], *, max_steps: int = 24) -> list[str]:
    lines = [
        (
            f"  invalidation source={report.get('source')} roots={len(report.get('roots', []))} "
            f"stats={len(report.get('stats', []))} pods={','.join(report.get('pods', [])) or '<none>'}"
        )
    ]
    steps = report.get("steps", [])
    for step in steps[:max_steps]:
        triggers = ",".join(step.get("triggered_by", []))
        lines.append(
            f"    step={step.get('step')} id={step.get('id')} kind={step.get('kind')} trigger={triggers}"
        )
    if len(steps) > max_steps:
        lines.append(f"    ... {len(steps) - max_steps} more stable-topology steps")
    return lines


def print_lines(lines: Iterable[str], printer: PrintFn = print) -> None:
    for line in lines:
        printer(line)


def metrics_summary(session: WorkbenchSession) -> dict[str, Any]:
    return {
        "versions": session.versions(),
        "digest": session.content_digest(),
        "graphSignature": session.graph.signature(),
        "pods": [
            {
                "pod": pod.pod_id,
                "selected": session.results[pod.pod_id]["selected"],
                "candidates": session.results[pod.pod_id]["candidates"],
                "version": session.results[pod.pod_id]["version"],
                "scoreMilli": session.results[pod.pod_id]["selected_score_milli"],
                "errorBoundMilli": session.results[pod.pod_id]["error_bound_milli"],
            }
            for pod in session.pods
        ],
    }


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
