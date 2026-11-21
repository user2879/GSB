"""Tkinter-independent, stably ordered diagnostic view model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .engine import EngineRunResult
from .models import ErrorPropagationResult, IndistinguishabilityResult, PolicySet


@dataclass(frozen=True)
class DiagnosticView:
    topology_lines: tuple[str, ...]
    timeline_rows: tuple[dict[str, Any], ...]
    stage_rows: tuple[dict[str, Any], ...]
    error_heat_rows: tuple[dict[str, Any], ...]
    dominant_stage: str
    error_summary: tuple[str, ...]
    indistinguishability_rows: tuple[dict[str, Any], ...]
    snapshot_rows: tuple[dict[str, Any], ...]

    def as_text(self) -> str:
        lines = ["TOPOLOGY"]
        lines.extend(self.topology_lines)
        lines.append("")
        lines.append("STAGES")
        for row in self.stage_rows:
            lines.append(
                "{stage_id}: pass={passed} queue={queued} drop={dropped} "
                "release={released} bytes={bytes} maxq={maxq}".format(**row)
            )
        lines.append("")
        lines.append("ERRORS")
        lines.extend(self.error_summary)
        lines.append("dominant=" + self.dominant_stage)
        lines.append("")
        lines.append("INDISTINGUISHABILITY")
        if not self.indistinguishability_rows:
            lines.append("none")
        for row in self.indistinguishability_rows:
            lines.append(
                "{indistinguishable} diff={released_bytes_difference} "
                "bound={combined_bytes_error_bound} winner={winner}".format(**row)
            )
        lines.append("")
        lines.append("SNAPSHOTS")
        if not self.snapshot_rows:
            lines.append("none")
        for row in self.snapshot_rows:
            lines.append(
                "{filename} version={policy_version} scale={scale} cursor={cursor_seq} "
                "digest={content_digest}".format(**row)
            )
        return "\n".join(lines)


def _topology(policy: PolicySet) -> tuple[str, ...]:
    children = policy.children_by_upstream
    lines: list[str] = []
    for root in policy.roots:
        lines.append(f"root {root} (match {policy.by_id[root].match_dst_port_prefix}*)")
    for edge in policy.edges:
        lines.append(f"{edge.upstream} -> {edge.downstream}")
    for stage_id in policy.topological_order():
        if not children[stage_id]:
            lines.append(f"{stage_id} -> <sink>")
    return tuple(lines)


def build_view(
    *,
    policy: PolicySet,
    run: EngineRunResult,
    errors: ErrorPropagationResult,
    indistinguishability: Iterable[IndistinguishabilityResult] = (),
    snapshots: Iterable[Mapping[str, Any]] = (),
) -> DiagnosticView:
    topo_index = {
        stage_id: index for index, stage_id in enumerate(policy.topological_order())
    }
    timeline_rows = tuple(
        {
            "event_seq": event.event_seq,
            "packet_id": event.packet_id,
            "stage_id": event.stage_id,
            "kind": event.kind,
            "decision": event.decision,
            "time_ns": event.time_ns,
            "tokens": event.tokens_after,
            "queue": event.queue_len_after,
            "delay_ns": event.delay_ticks * (10**9 // (10**errors.scale)),
        }
        for event in sorted(
            run.events,
            key=lambda item: (
                item.time_tick,
                item.packet_seq,
                topo_index.get(item.stage_id, len(topo_index)),
                item.kind,
                item.event_seq,
            ),
        )
    )
    stage_rows = tuple(
        {
            "stage_id": stats.stage_id,
            "passed": stats.passed_packets + stats.released_packets,
            "queued": stats.queued_packets,
            "dropped": stats.dropped_packets,
            "released": stats.released_packets,
            "bytes": stats.forwarded_bytes,
            "maxq": stats.max_queue_len,
        }
        for stats in run.summary.stage_stats
    )
    error_heat_rows = tuple(
        {
            "stage_id": term.stage_id,
            "bytes_contribution": term.bytes_contribution,
            "bytes_share": f"{term.bytes_share:.2%}",
            "bytes_share_value": term.bytes_share,
            "heat": "█" * (0 if term.bytes_share <= 0 else max(1, round(term.bytes_share * 10))),
            "delay_contribution_ns": term.delay_contribution_ns,
            "delay_share": f"{term.delay_share:.2%}",
            "dominant": "YES" if term.stage_id == errors.dominant_stage else "",
        }
        for term in errors.terms
    )
    error_summary = (
        f"bytes={errors.final_released_bytes} +/- {errors.bytes_error_bound}",
        f"avg_delay_ns={errors.average_delay_ns} +/- {errors.average_delay_error_bound_ns}",
        f"scale={errors.scale} tick_ns={errors.tick_ns} threshold={errors.dominance_threshold:.2f}",
    )
    indistinguishable_rows = tuple(
        {
            "indistinguishable": (
                "YES" if item.indistinguishable else "NO"
            ),
            "released_bytes_difference": item.released_bytes_difference,
            "combined_bytes_error_bound": item.combined_bytes_error_bound,
            "winner": item.winner,
            "stable_order": ",".join(item.stable_order),
        }
        for item in indistinguishability
    )
    snapshot_rows = tuple(
        {
            "filename": str(item.get("filename", "")),
            "policy_version": str(item.get("policy_version", "")),
            "scale": int(item.get("scale", 0)),
            "cursor_seq": int(item.get("cursor_seq", 0)),
            "content_digest": str(item.get("content_digest", "")),
        }
        for item in sorted(
            snapshots,
            key=lambda value: (
                str(value.get("policy_version", "")),
                str(value.get("content_digest", "")),
                str(value.get("filename", "")),
            ),
        )
    )
    return DiagnosticView(
        topology_lines=_topology(policy),
        timeline_rows=timeline_rows,
        stage_rows=stage_rows,
        error_heat_rows=error_heat_rows,
        dominant_stage=errors.dominant_stage,
        error_summary=error_summary,
        indistinguishability_rows=indistinguishable_rows,
        snapshot_rows=snapshot_rows,
    )
