"""Tk diagnostic workbench plus a deterministic text renderer.

Tkinter is imported lazily so validation, replay, and headless diagnostics do
not require a display or the Tk shared libraries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def render_view_text(view: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("SERVERLESS CONCURRENCY LEASE WORKBENCH")
    lines.append(
        "version={version} ledger_batch={batch} cursor={cursor} clock_ms={clock} scale={scale}".format(
            version=view.get("declaration_version"),
            batch=view.get("ledger_batch") or "-",
            cursor=view.get("ledger_cursor", view.get("clock_ms")),
            clock=view.get("clock_ms"),
            scale=view.get("scale"),
        )
    )
    lines.append(
        "global_used={used}/{budget} recovery={status}".format(
            used=view.get("global_used"),
            budget=view.get("global_budget"),
            status=view.get("recovery_status", "UNKNOWN"),
        )
    )
    if view.get("unknown_lease_ids"):
        lines.append(
            "UNKNOWN_RECOMPUTE_REQUIRED lease_ids="
            + ",".join(sorted(view["unknown_lease_ids"]))
        )
    lines.append("")

    lines.append("FUNCTIONS")
    lines.append(
        "function   prio  used  cap  borrowed  lent  queue  min  max  lease_ms"
    )
    for item in sorted(view.get("functions", []), key=lambda row: row["function_id"]):
        lines.append(
            "{fid:<10} {prio:>4} {used:>5} {cap:>4} {borrowed:>9} {lent:>5} "
            "{queue:>6} {minimum:>4} {maximum:>4} {lease:>9}".format(
                fid=item["function_id"],
                prio=item["priority"],
                used=item["used_concurrency"],
                cap=item["effective_capacity"],
                borrowed=item["borrowed_concurrency"],
                lent=item["lent_concurrency"],
                queue=item["queue_length"],
                minimum=item["min_concurrency"],
                maximum=item["max_concurrency"],
                lease=item["lease_ms"],
            )
        )
    lines.append("")

    lines.append("ACTIVE LEASES")
    lines.append("lease_id          event  function   concurrency  grant_ms  return_deadline_ms")
    for lease in sorted(view.get("leases", []), key=lambda row: row["lease_id"]):
        lines.append(
            "{lease:<17} {event:>5}  {function:<9} {concurrency:>11}  {grant:>8}  {deadline:>18}".format(
                lease=lease["lease_id"],
                event=lease["event_seq"],
                function=lease["function_id"],
                concurrency=lease["concurrency"],
                grant=lease["grant_ms"],
                deadline=lease["return_deadline_ms"],
            )
        )
    lines.append("")

    lines.append("BORROWING")
    lines.append("loan_id                 lender      borrower    amount  return_deadline_ms  status")
    for loan in sorted(view.get("loans", []), key=lambda row: row["loan_id"]):
        lines.append(
            "{loan:<23} {lender:<11} {borrower:<11} {amount:>6}  {deadline:>18}  {status}".format(
                loan=loan["loan_id"],
                lender=loan["lender_id"],
                borrower=loan["borrower_id"],
                amount=loan["amount"],
                deadline=loan["return_deadline_ms"],
                status=loan["status"],
            )
        )
    lines.append("")

    lines.append("QUEUE")
    lines.append("queue_id      seq  event  function   demand  enqueued_ms  not_before_ms  source")
    for entry in sorted(view.get("queue", []), key=lambda row: row["queue_seq"]):
        lines.append(
            "{qid:<13} {qseq:>3}  {event:>5}  {function:<9} {demand:>6}  {enqueued:>11}  {not_before:>13}  {source}".format(
                qid=entry["queue_id"],
                qseq=entry["queue_seq"],
                event=entry["event_seq"],
                function=entry["function_id"],
                demand=entry["demand"],
                enqueued=entry["enqueued_ms"],
                not_before=entry["not_before_ms"],
                source=entry["source"],
            )
        )
    lines.append("")

    lines.append("PREEMPTIONS")
    lines.append("preemption_id          beneficiary  victim_lease      occupied  suspend_at  status")
    for item in sorted(
        view.get("preemptions", []), key=lambda row: row["preemption_id"]
    ):
        lines.append(
            "{pid:<22} {event:>11}  {victim:<17} {occupied:>8}  {suspend:>10}  {status}".format(
                pid=item["preemption_id"],
                event=item["beneficiary_event_seq"],
                victim=item["victim_lease_id"],
                occupied=item["occupied_concurrency"],
                suspend=item["suspend_at_ms"],
                status=item["status"],
            )
        )
    lines.append("")

    lines.append("FIXED-POINT WATERMARKS (FLOOR_TOWARD_ZERO)")
    lines.append(
        "scope     function  used  capacity  ratio_fp  scale  compare_err_fp  borrow_err_fp  dominant  affected"
    )
    for item in sorted(
        view.get("watermarks", []),
        key=lambda row: (0 if row.get("scope") == "GLOBAL" else 1, row.get("function_id") or ""),
    ):
        lines.append(
            "{scope:<9} {function:<9} {used:>4}  {capacity:>8}  {ratio:>8}  {scale:>5}  {comparison:>14}  {borrow:>13}  {dominant:<30} {affected}".format(
                scope=item["scope"],
                function=item.get("function_id") or "-",
                used=item["used_concurrency"],
                capacity=item["capacity"],
                ratio=item["ratio_fp"],
                scale=item["ratio_scale"],
                comparison=item["total_comparison_error_fp"],
                borrow=item["borrow_return_error_fp"],
                dominant=item["dominant_error_segment"],
                affected="YES"
                if item["comparison_affected_by_error"]
                else "NO",
            )
        )
    lines.append("")

    lines.append("STALE INVALIDATIONS")
    invalidations = view.get("ledger_invalidations") or view.get("invalidations", [])
    for item in sorted(
        invalidations,
        key=lambda row: (
            row.get("new_version", 0),
            row.get("kind", ""),
            row.get("object_id", ""),
        ),
    ):
        lines.append(
            "v{old}->v{new} {kind:<10} {object_id:<22} event={event} reason={reason}".format(
                old=item.get("old_version", "?"),
                new=item.get("new_version", "?"),
                kind=item.get("kind", "LEASE"),
                object_id=item.get("object_id", item.get("lease_id", "-")),
                event=item.get("event_seq", "-"),
                reason=item.get("reason", "-"),
            )
        )
    if not invalidations:
        lines.append("(none)")
    lines.append("")
    lines.append("view_digest=" + str(view.get("view_digest", "-")))
    return "\n".join(lines)


class WorkbenchTkApp:
    def __init__(
        self,
        *,
        view_loader: Callable[[], dict[str, Any]],
        refresh_ms: int = 1000,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.view_loader = view_loader
        self.root = tk.Tk()
        self.root.title("Serverless Concurrency Lease Workbench")
        self.root.geometry("1280x820")
        self.root.minsize(960, 640)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.header_var = tk.StringVar()
        header = ttk.Label(
            self.root,
            textvariable=self.header_var,
            font=("TkFixedFont", 11, "bold"),
            anchor="w",
            padding=8,
        )
        header.grid(row=0, column=0, sticky="ew")
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self._build_tables()
        controls = ttk.Frame(self.root, padding=6)
        controls.grid(row=2, column=0, sticky="ew")
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=12)
        self.refresh_ms = refresh_ms
        self.refresh()
        self.root.after(self.refresh_ms, self._tick)

    def _build_tables(self) -> None:
        ttk = self.ttk
        self.function_tree = ttk.Treeview(
            self.notebook,
            columns=("used", "cap", "borrowed", "lent", "queue", "ratio", "affected"),
            show="tree headings",
        )
        for key, label in (
            ("#0", "function"),
            ("used", "used"),
            ("cap", "capacity"),
            ("borrowed", "borrowed"),
            ("lent", "lent"),
            ("queue", "queue"),
            ("ratio", "ratio_fp"),
            ("affected", "error affected"),
        ):
            self.function_tree.heading(key, text=label)
        self.function_tree.column("#0", width=160, anchor="w")
        self.notebook.add(self.function_tree, text="Functions")

        self.lease_tree = ttk.Treeview(
            self.notebook,
            columns=("event", "function", "concurrency", "grant", "deadline"),
            show="tree headings",
        )
        for key, label in (
            ("#0", "lease_id"),
            ("event", "event"),
            ("function", "function"),
            ("concurrency", "concurrency"),
            ("grant", "grant_ms"),
            ("deadline", "return_deadline_ms"),
        ):
            self.lease_tree.heading(key, text=label)
        self.lease_tree.column("#0", width=210, anchor="w")
        self.notebook.add(self.lease_tree, text="Leases")

        self.queue_tree = ttk.Treeview(
            self.notebook,
            columns=("event", "function", "demand", "enqueued", "not_before", "source"),
            show="tree headings",
        )
        for key, label in (
            ("#0", "queue_id"),
            ("event", "event"),
            ("function", "function"),
            ("demand", "demand"),
            ("enqueued", "enqueued_ms"),
            ("not_before", "not_before_ms"),
            ("source", "source"),
        ):
            self.queue_tree.heading(key, text=label)
        self.queue_tree.column("#0", width=150, anchor="w")
        self.notebook.add(self.queue_tree, text="Queue")

        self.text = self.tk.Text(
            self.notebook,
            wrap="none",
            font=("TkFixedFont", 10),
        )
        self.notebook.add(self.text, text="Borrow / Preemption / Watermarks")

    def _tick(self) -> None:
        self.refresh()
        self.root.after(self.refresh_ms, self._tick)

    def refresh(self) -> None:
        try:
            view = self.view_loader()
        except Exception as exc:
            self.status_var.set(f"Refresh failed: {type(exc).__name__}: {exc}")
            return
        self.header_var.set(
            "declaration v{version} | ledger {batch} | cursor {cursor} | clock {clock} ms | "
            "global {used}/{budget} | recovery {recovery} | digest {digest}".format(
                version=view.get("declaration_version"),
                batch=view.get("ledger_batch") or "-",
                cursor=view.get("ledger_cursor"),
                clock=view.get("clock_ms"),
                used=view.get("global_used"),
                budget=view.get("global_budget"),
                recovery=view.get("recovery_status"),
                digest=str(view.get("view_digest", ""))[:12],
            )
        )
        self.function_tree.delete(*self.function_tree.get_children())
        watermarks = {
            item["function_id"]: item
            for item in view.get("watermarks", [])
            if item.get("scope") == "FUNCTION"
        }
        for item in sorted(view.get("functions", []), key=lambda row: row["function_id"]):
            watermark = watermarks.get(item["function_id"], {})
            self.function_tree.insert(
                "",
                "end",
                iid="fn-" + item["function_id"],
                text=item["function_id"],
                values=(
                    item["used_concurrency"],
                    item["effective_capacity"],
                    item["borrowed_concurrency"],
                    item["lent_concurrency"],
                    item["queue_length"],
                    watermark.get("ratio_fp", "-"),
                    "YES" if watermark.get("comparison_affected_by_error") else "NO",
                ),
            )
        self.lease_tree.delete(*self.lease_tree.get_children())
        for item in sorted(view.get("leases", []), key=lambda row: row["lease_id"]):
            self.lease_tree.insert(
                "",
                "end",
                iid="lease-" + item["lease_id"],
                text=item["lease_id"],
                values=(
                    item["event_seq"],
                    item["function_id"],
                    item["concurrency"],
                    item["grant_ms"],
                    item["return_deadline_ms"],
                ),
            )
        self.queue_tree.delete(*self.queue_tree.get_children())
        for item in sorted(view.get("queue", []), key=lambda row: row["queue_seq"]):
            self.queue_tree.insert(
                "",
                "end",
                iid="queue-" + item["queue_id"],
                text=item["queue_id"],
                values=(
                    item["event_seq"],
                    item["function_id"],
                    item["demand"],
                    item["enqueued_ms"],
                    item["not_before_ms"],
                    item["source"],
                ),
            )
        detail = []
        detail.append("BORROWING")
        for item in sorted(view.get("loans", []), key=lambda row: row["loan_id"]):
            detail.append(
                "{loan} {lender}->{borrower} amount={amount} deadline={deadline}".format(
                    loan=item["loan_id"],
                    lender=item["lender_id"],
                    borrower=item["borrower_id"],
                    amount=item["amount"],
                    deadline=item["return_deadline_ms"],
                )
            )
        detail.append("")
        detail.append("PREEMPTIONS")
        for item in sorted(
            view.get("preemptions", []), key=lambda row: row["preemption_id"]
        ):
            detail.append(
                "{pid} event={event} victim={victim} occupied={occupied} "
                "suspend_at={suspend} status={status}".format(
                    pid=item["preemption_id"],
                    event=item["beneficiary_event_seq"],
                    victim=item["victim_lease_id"],
                    occupied=item["occupied_concurrency"],
                    suspend=item["suspend_at_ms"],
                    status=item["status"],
                )
            )
        detail.append("")
        detail.append("WATERMARKS")
        for item in sorted(
            view.get("watermarks", []),
            key=lambda row: (
                0 if row.get("scope") == "GLOBAL" else 1,
                row.get("function_id") or "",
            ),
        ):
            detail.append(
                "{scope}:{function} used={used} capacity={capacity} ratio_fp={ratio}/"
                "{scale} comparison_error_fp={comparison} borrow_delay_error_fp={borrow} "
                "dominant={dominant} affected={affected} rounding={rounding}".format(
                    scope=item["scope"],
                    function=item.get("function_id") or "-",
                    used=item["used_concurrency"],
                    capacity=item["capacity"],
                    ratio=item["ratio_fp"],
                    scale=item["ratio_scale"],
                    comparison=item["total_comparison_error_fp"],
                    borrow=item["borrow_return_error_fp"],
                    dominant=item["dominant_error_segment"],
                    affected=item["comparison_affected_by_error"],
                    rounding=item["rounding"],
                )
            )
        detail.append("")
        detail.append("STALE INVALIDATIONS")
        invalidations = view.get("ledger_invalidations") or view.get("invalidations", [])
        for item in sorted(
            invalidations,
            key=lambda row: (
                row.get("new_version", 0),
                row.get("kind", ""),
                row.get("object_id", ""),
            ),
        ):
            detail.append(
                "v{old}->v{new} {kind} {object} event={event} reason={reason}".format(
                    old=item.get("old_version"),
                    new=item.get("new_version"),
                    kind=item.get("kind", "LEASE"),
                    object=item.get("object_id", item.get("lease_id")),
                    event=item.get("event_seq"),
                    reason=item.get("reason"),
                )
            )
        if not invalidations:
            detail.append("(none)")
        if view.get("unknown_lease_ids"):
            detail.append("")
            detail.append("UNKNOWN_RECOMPUTE_REQUIRED")
            detail.extend(sorted(view["unknown_lease_ids"]))
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(detail))
        self.text.configure(state="disabled")
        self.status_var.set(
            "ledger unchanged by refresh | queues={queues} leases={leases} unknown={unknown}".format(
                queues=len(view.get("queue", [])),
                leases=len(view.get("leases", [])),
                unknown=len(view.get("unknown_lease_ids", [])),
            )
        )

    def run(self) -> None:
        self.root.mainloop()


def launch_ui(view_loader: Callable[[], dict[str, Any]]) -> None:
    try:
        import tkinter
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable in this Python installation. On Windows, "
            "rerun the official Python installer and enable 'tcl/tk and IDLE'. "
            "Use --headless for deterministic text rendering."
        ) from exc
    app = WorkbenchTkApp(view_loader=view_loader)
    app.run()


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8", newline="\n")
