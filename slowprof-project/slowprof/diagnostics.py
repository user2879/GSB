"""Interactive diagnostics: stable text model plus an optional Tkinter view.

The rendering pipeline is strictly read-only: it reads ledger/snapshot state
and never writes back, so refreshing cannot alter the ledger. When Tkinter is
available (e.g. a standard Windows Python install verified with
``python -m tkinter``) :func:`launch_gui` opens a window; otherwise
:func:`render_text` produces the same, deterministically ordered report on the
console.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .fixedpoint import ShareReport
from .folding import STACK_SEP, FoldResult
from .ledger import BatchRecord, Ledger
from .ranking import FrameContext, FrameRank
from .samples import Interval
from .snapshot import read_snapshot


def tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    return True


def _share_lookup(report: Optional[ShareReport]
                  ) -> Dict[str, Dict[str, object]]:
    if report is None:
        return {}
    table: Dict[str, Dict[str, object]] = {}
    for share in report.self_shares:
        table.setdefault(share.frame, {})["self"] = share
    for share in report.cumulative_shares:
        table.setdefault(share.frame, {})["cumulative"] = share
    return table


def render_text(*, interval: Interval, result: FoldResult,
                ranks: Sequence[FrameRank],
                contexts: Dict[str, FrameContext],
                report: Optional[ShareReport],
                batches: Sequence[BatchRecord],
                max_stacks: int = 15) -> str:
    """Deterministic, line-stable textual diagnostics report."""

    lines: List[str] = []
    lines.append("=== slowprof diagnostics ===")
    lines.append(f"observation_window : {interval.canonical()}")
    lines.append(
        f"aggregation_path   : {result.path_used} "
        f"(stacks<{result.unique_stack_threshold}, "
        f"weight<{result.total_weight_threshold})"
    )
    lines.append(f"path_reason        : {result.path_reason}")
    lines.append(
        f"totals             : rows_folded_unique_stacks="
        f"{result.unique_stack_count} total_weight={result.total_weight}")

    lines.append("")
    lines.append("-- folded stacks (top by weight, then path) --")
    stacks = sorted(
        result.stacks, key=lambda e: (-e.weight, e.frames))[:max_stacks]
    for entry in stacks:
        lines.append(
            f"{entry.weight:>8}  {STACK_SEP.join(entry.frames)}  [leaf={entry.leaf}]"
        )

    lines.append("")
    lines.append("-- hotspot frame ranking (self desc, cumulative desc, name) --")
    share_table = _share_lookup(report)
    for rank in ranks:
        suffix = ""
        info = share_table.get(rank.frame)
        if info and "self" in info:
            share = info["self"]
            suffix = (
                f"  self_share={share.quantized}/{share.error_bound_den} "
                f"(err<{share.error_bound_num}/{share.error_bound_den})"
                + (" [ERROR-AFFECTED]" if share.error_affected else "")
            )
        lines.append(
            f"#{rank.rank:<2} {rank.frame:<28} self={rank.self_weight:<8} "
            f"cum={rank.cumulative_weight:<8}{suffix}"
        )

    lines.append("")
    lines.append("-- call context (direct callers / callees) --")
    for rank in ranks:
        ctx = contexts.get(rank.frame)
        if ctx is None:
            continue
        callers = ", ".join(
            f"{e.counterpart}(w={e.edge_weight},n={e.call_count})"
            for e in ctx.callers) or "<root>"
        callees = ", ".join(
            f"{e.counterpart}(w={e.edge_weight},n={e.call_count})"
            for e in ctx.callees) or "<leaf>"
        lines.append(f"{rank.frame}:")
        lines.append(f"    callers: {callers}")
        lines.append(f"    callees: {callees}")

    lines.append("")
    lines.append("-- fixed-point error annotations --")
    if report is not None:
        lines.append(
            f"scale={report.scale} total_weight={report.total_weight}")
        lines.append(
            f"per_frame_error_bound=1/{report.scale}; "
            f"top{len(report.self_shares)}_aggregate_error_bound="
            f"{report.aggregate_error_bound_num}/"
            f"{report.aggregate_error_bound_den}"
        )
        lines.append(
            f"top_k_contains_true_top_k="
            f"{report.top_k_contains_true_top_k} "
            f"basis={report.top_k_basis}"
        )
        if report.error_affected_pairs:
            for left, right, kind in report.error_affected_pairs:
                lines.append(
                    f"error-affected ordering: {left} vs {right} ({kind}) "
                    "decided by stable exact-arithmetic tie-break"
                )
        else:
            lines.append("no quantization-indistinguishable adjacent frames")

    lines.append("")
    lines.append("-- ledger batches (committed only) --")
    for batch in batches:
        lines.append(
            f"batch#{batch.batch_seq} lines={batch.start_line}.."
            f"{batch.end_line} samples={batch.sample_count} "
            f"weight+={batch.weight_added} "
            f"committed_total={batch.committed_total_weight}"
        )
    return "\n".join(lines)


def launch_gui(*, title: str, text_report: str) -> bool:
    """Open a Tkinter window showing the stable report.

    Returns ``True`` if a window was opened, ``False`` when Tkinter is
    unavailable (caller should use the text fallback). Read-only: no ledger
    write occurs here.
    """

    if not tkinter_available():
        return False
    import tkinter as tk  # local import keeps headless import cheap

    root = tk.Tk()
    root.title(title)
    root.geometry("980x720")
    header = tk.Label(
        root, text=title, font=("Courier", 12, "bold"), anchor="w")
    header.pack(fill="x", padx=8, pady=(8, 2))
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=8, pady=8)
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")
    text_widget = tk.Text(
        frame, wrap="none", font=("Courier", 10),
        yscrollcommand=scrollbar.set)
    text_widget.insert("1.0", text_report)
    text_widget.configure(state="disabled")  # render is read-only
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=text_widget.yview)
    root.mainloop()
    return True


def render_from_ledger(config, session) -> str:
    """Convenience adapter building the report from a finished session."""

    ledger = Ledger(config.ledger_path)
    try:
        batches = ledger.list_batches()
    finally:
        ledger.close()
    assert session.result is not None and session.report is not None
    return render_text(
        interval=session.interval,
        result=session.result,
        ranks=session.ranks,
        contexts=session.contexts,
        report=session.report,
        batches=batches,
    )


def render_from_snapshot(snapshot_dir: str, ledger_path: str,
                         interval: "Interval") -> str:
    """Read-only diagnostics built entirely from a persisted snapshot.

    No write is issued against the ledger or snapshot directory, so repeated
    refreshes produce the identical report without mutating any state.
    """

    from .errors import LedgerError
    from .folding import FrameMetric, FoldResult, StackEntry
    from .fixedpoint import ShareReport
    from .fixedpoint import FrameShare
    from .ranking import ContextEdge, FrameContext, FrameRank

    try:
        record = read_snapshot(snapshot_dir)
    except ValueError as exc:  # corrupt/tampered transport digest
        raise LedgerError(str(exc), field="snapshot") from exc
    if record is None:
        raise LedgerError(
            "no snapshot found; run 'profile' before 'diagnose'",
            field="snapshot_dir")

    result_payload = record["result"]
    result = FoldResult(
        stacks=tuple(
            StackEntry(
                frames=tuple(item["frames"]),
                weight=item["weight"],
                leaf=item["leaf"],
            )
            for item in result_payload["stacks"]
        ),
        frame_metrics=tuple(
            FrameMetric(
                frame=item["frame"],
                self_weight=item["self_weight"],
                cumulative_weight=item["cumulative_weight"],
            )
            for item in result_payload["frame_metrics"]
        ),
        total_weight=result_payload["total_weight"],
        unique_stack_count=result_payload["unique_stack_count"],
        path_used=result_payload["path_used"],
        path_reason=result_payload["path_reason"],
        unique_stack_threshold=result_payload[
            "unique_stack_threshold"],
        total_weight_threshold=result_payload[
            "total_weight_threshold"],
    )
    ranks = [
        FrameRank(
            rank=item["rank"],
            frame=item["frame"],
            self_weight=item["self_weight"],
            cumulative_weight=item["cumulative_weight"],
        )
        for item in record["ranks"]
    ]

    def _edges(payload):
        return tuple(
            ContextEdge(
                counterpart=item["counterpart"],
                edge_weight=item["edge_weight"],
                call_count=item["call_count"],
            )
            for item in payload
        )

    contexts = {
        frame: FrameContext(
            frame=frame,
            callers=_edges(payload["callers"]),
            callees=_edges(payload["callees"]),
        )
        for frame, payload in record["contexts"].items()
    }

    def _shares(payload):
        return tuple(FrameShare(**item) for item in payload)

    share_payload = record["share_report"]
    agg_num, agg_den = share_payload["aggregate_error_bound"]
    report = ShareReport(
        scale=share_payload["scale"],
        total_weight=share_payload["total_weight"],
        self_shares=_shares(share_payload["self_shares"]),
        cumulative_shares=_shares(share_payload["cumulative_shares"]),
        aggregate_error_bound_num=agg_num,
        aggregate_error_bound_den=agg_den,
        top_k_contains_true_top_k=share_payload[
            "top_k_contains_true_top_k"],
        top_k_basis=share_payload["top_k_basis"],
        error_affected_pairs=tuple(
            tuple(item) for item in share_payload["error_affected_pairs"]
        ),
    )

    ledger = Ledger(ledger_path)
    try:
        batches = ledger.list_batches()  # SELECT only, no writes
    finally:
        ledger.close()

    return render_text(
        interval=interval,
        result=result,
        ranks=ranks,
        contexts=contexts,
        report=report,
        batches=batches,
    )
