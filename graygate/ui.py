from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

from .errors import GatewayError
from .service import WorkbenchService
from .snapshot import Snapshot

try:  # Tk is optional for headless core tests and CI containers.
    import tkinter as tk
    from tkinter import messagebox, ttk

    TK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on machines without Tk
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    TK_AVAILABLE = False


class WorkbenchApp:
    def __init__(self, root: Any, spec_path: str, db_path: str):
        self.root = root
        self.service = WorkbenchService(spec_path, db_path)
        self._active_rebuild: str | None = None
        self._busy = False
        self._selected_snapshot: Snapshot | None = None
        self.root.title("Gray Snapshot Rebuild Workbench")
        self.root.geometry("1280x820")
        self._build_layout()
        self._refresh_dashboard()
        if self.service.ledger.recovered_unknown or self.service.ledger.list_snapshots(limit=1):
            self.root.after(250, self._recover_replay)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(
            header,
            text=f"spec={self.service.spec.source_path}  db={self.service.db_path}",
        ).pack(side="left")
        self.status_var = tk.StringVar(value="ready")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(6, 6))
        ttk.Button(buttons, text="重建快照", command=self._start_rebuild).pack(
            side="left"
        )
        ttk.Button(buttons, text="刷新查询", command=self._refresh_dashboard).pack(
            side="left", padx=4
        )
        ttk.Button(buttons, text="权威校验", command=self._check_consistency).pack(
            side="left", padx=4
        )
        ttk.Button(buttons, text="恢复重放", command=self._recover_replay).pack(
            side="left", padx=4
        )

        summary_frame = ttk.LabelFrame(outer, text="当前快照 / 交互延迟 / 一致性")
        summary_frame.pack(fill="x", pady=(0, 6))
        self.summary_var = tk.StringVar(value="no snapshot")
        ttk.Label(
            summary_frame,
            textvariable=self.summary_var,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=6, pady=4)

        stats_frame = ttk.LabelFrame(outer, text="规则 × 阶段灰度统计")
        stats_frame.pack(fill="both", expand=True, pady=(0, 6))
        columns = (
            "rule",
            "stage",
            "weight",
            "stable",
            "canary",
            "ratio_bp",
            "stable_us",
            "canary_us",
            "stable_max",
            "canary_max",
        )
        self.stats_tree = ttk.Treeview(
            stats_frame, columns=columns, show="headings", height=10
        )
        headings = {
            "rule": "规则",
            "stage": "阶段",
            "weight": "灰度权重万分比",
            "stable": "稳定命中",
            "canary": "灰度命中",
            "ratio_bp": "实际灰度占比(bp)",
            "stable_us": "稳定总耗时(us)",
            "canary_us": "灰度总耗时(us)",
            "stable_max": "稳定最大(us)",
            "canary_max": "灰度最大(us)",
        }
        widths = {
            "rule": 150,
            "stage": 90,
            "weight": 120,
            "stable": 90,
            "canary": 90,
            "ratio_bp": 140,
            "stable_us": 120,
            "canary_us": 120,
            "stable_max": 110,
            "canary_max": 110,
        }
        for column in columns:
            self.stats_tree.heading(column, text=headings[column])
            self.stats_tree.column(column, width=widths[column], anchor="center")
        self.stats_tree.pack(fill="both", expand=True, side="left")
        stats_scroll = ttk.Scrollbar(
            stats_frame, orient="vertical", command=self.stats_tree.yview
        )
        stats_scroll.pack(side="right", fill="y")
        self.stats_tree.configure(yscrollcommand=stats_scroll.set)

        lower = ttk.Frame(outer)
        lower.pack(fill="both", expand=True)
        snapshots_frame = ttk.LabelFrame(lower, text="快照列表")
        snapshots_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.snapshot_tree = ttk.Treeview(
            snapshots_frame,
            columns=("id", "source", "observed", "excluded", "elapsed"),
            show="headings",
            height=8,
        )
        for column, text, width in (
            ("id", "快照", 270),
            ("source", "账本版本", 90),
            ("observed", "观察上界", 90),
            ("excluded", "排除命中", 90),
            ("elapsed", "重建ms", 80),
        ):
            self.snapshot_tree.heading(column, text=text)
            self.snapshot_tree.column(column, width=width, anchor="center")
        self.snapshot_tree.pack(fill="both", expand=True)
        self.snapshot_tree.bind("<<TreeviewSelect>>", self._select_snapshot)

        rebuilds_frame = ttk.LabelFrame(lower, text="重建轨迹")
        rebuilds_frame.pack(side="left", fill="both", expand=True, padx=4)
        self.rebuild_tree = ttk.Treeview(
            rebuilds_frame,
            columns=("id", "source", "status", "elapsed", "excluded"),
            show="headings",
            height=8,
        )
        for column, text, width in (
            ("id", "重建", 240),
            ("source", "源版本", 80),
            ("status", "状态", 90),
            ("elapsed", "时限ms", 80),
            ("excluded", "排除", 70),
        ):
            self.rebuild_tree.heading(column, text=text)
            self.rebuild_tree.column(column, width=width, anchor="center")
        self.rebuild_tree.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(lower, text="诊断")
        log_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.log = tk.Text(log_frame, width=52, height=10, wrap="none")
        self.log.pack(fill="both", expand=True)

    def _selected_snapshot_or_current(self) -> Snapshot | None:
        if self._selected_snapshot is not None:
            return self._selected_snapshot
        return self.service.current_snapshot

    def _populate_stats(self, snapshot: Snapshot | None) -> None:
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        if snapshot is None:
            return
        for row in snapshot.stats:
            self.stats_tree.insert(
                "",
                "end",
                values=(
                    row["rule_id"],
                    row["stage_id"],
                    row["stage_weight_bp"],
                    row["stable_hits"],
                    row["canary_hits"],
                    row["canary_ratio_bp"],
                    row["stable_total_us"],
                    row["canary_total_us"],
                    row["stable_max_us"],
                    row["canary_max_us"],
                ),
            )

    def _populate_lists(self, dashboard: dict[str, Any]) -> None:
        self.snapshot_tree.delete(*self.snapshot_tree.get_children())
        for row in dashboard["snapshots"]:
            self.snapshot_tree.insert(
                "",
                "end",
                iid=row["snapshot_id"],
                values=(
                    row["snapshot_id"],
                    row["source_ledger_version"],
                    row["observed_ledger_version"],
                    row["excluded_hits"],
                    row["rebuild_elapsed_ms"],
                ),
            )
        self.rebuild_tree.delete(*self.rebuild_tree.get_children())
        for row in dashboard["rebuilds"]:
            self.rebuild_tree.insert(
                "",
                "end",
                values=(
                    row["rebuild_id"],
                    row["source_ledger_version"],
                    row["status"],
                    row["elapsed_ms"] if row["elapsed_ms"] is not None else "-",
                    row["excluded_hits"] if row["excluded_hits"] is not None else "-",
                ),
            )

    def _summary_text(
        self, dashboard: dict[str, Any], snapshot: Snapshot | None
    ) -> str:
        lines = [f"ledger_version={dashboard['ledger_version']}"]
        if snapshot is None:
            lines.append("snapshot=none")
        else:
            lines.append(
                "snapshot={} source={} observed={} excluded={}".format(
                    snapshot.snapshot_id,
                    snapshot.source_ledger_version,
                    snapshot.observed_ledger_version,
                    snapshot.excluded_hits,
                )
            )
            lines.append(
                "rebuild_elapsed_ms={} budget_ms={} exceeded={} aggregation={}".format(
                    snapshot.rebuild_elapsed_ms,
                    snapshot.budget_ms,
                    snapshot.latency_exceeded,
                    snapshot.aggregation_path,
                )
            )
        rebuild_latency = dashboard["latency"]["rebuild"]["summary"]
        query_latency = dashboard["latency"]["ui_query"]["summary"]
        lines.append(
            "rebuild_latency mean={}/{}ms p95={}ms max={}ms err<={} scaled".format(
                rebuild_latency["mean_scaled_ms"],
                rebuild_latency["scale"],
                rebuild_latency["p95_ms"],
                rebuild_latency["max_ms"],
                rebuild_latency["quantization_error_upper_bound_scaled_ms"],
            )
        )
        lines.append(
            "query_latency mean={}/{}ms p95={}ms max={}ms err<={} scaled".format(
                query_latency["mean_scaled_ms"],
                query_latency["scale"],
                query_latency["p95_ms"],
                query_latency["max_ms"],
                query_latency["quantization_error_upper_bound_scaled_ms"],
            )
        )
        if snapshot is not None:
            authoritative = snapshot.body
            lines.append(
                "snapshot_digest={} stats_rows={}".format(
                    snapshot.content_digest, len(authoritative["stats"])
                )
            )
        snapshot_summary = dashboard.get("snapshot") or {}
        authoritative_check = snapshot_summary.get("last_authoritative_check")
        replay_check = snapshot_summary.get("last_replay_check")
        if authoritative_check:
            lines.append(
                "authoritative_check={} first={}/{}/{}".format(
                    authoritative_check["code"],
                    authoritative_check["rule_id"],
                    authoritative_check["stage_id"],
                    authoritative_check["field_name"],
                )
            )
        else:
            lines.append("authoritative_check=none")
        if replay_check:
            lines.append(
                "replay_check={} first={}/{}/{}".format(
                    replay_check["code"],
                    replay_check["rule_id"],
                    replay_check["stage_id"],
                    replay_check["field_name"],
                )
            )
        else:
            lines.append("replay_check=none")
        for error in dashboard["errors"]:
            lines.append(
                "{} over_by_ms={} scope={}".format(
                    error["code"], error["over_by_ms"], error["impact_scope"]
                )
            )
        return "\n".join(lines)

    def _refresh_dashboard(self) -> None:
        if self._busy:
            return
        try:
            dashboard = self.service.dashboard()
            snapshot = self._selected_snapshot_or_current()
            self._populate_stats(snapshot)
            self._populate_lists(dashboard)
            self.summary_var.set(self._summary_text(dashboard, snapshot))
            if dashboard["errors"]:
                self._log(
                    "E_LATENCY "
                    + dashboard["errors"][0]["impact_scope"]
                    + " over_by_ms="
                    + str(dashboard["errors"][0]["over_by_ms"])
                )
        except GatewayError as exc:
            self._log(str(exc))
        except Exception as exc:  # pragma: no cover - UI safety net
            self._log(f"E_UI {exc}")

    def _start_rebuild(self) -> None:
        if self._busy:
            return
        try:
            self._active_rebuild = self.service.start_rebuild()
            self._busy = True
            self.status_var.set(f"rebuild running: {self._active_rebuild}")
            self._log(f"rebuild started {self._active_rebuild}")
            self.root.after(100, self._poll_rebuild)
        except GatewayError as exc:
            self._log(str(exc))

    def _poll_rebuild(self) -> None:
        if self._active_rebuild is None:
            self._busy = False
            return
        info = self.service.ledger.get_rebuild(self._active_rebuild)
        if info is None:
            self._busy = False
            self._log("E_REBUILD missing trajectory")
            return
        self.status_var.set(
            "rebuild {} {}".format(info["rebuild_id"], info["status"])
        )
        if info["status"] == "RUNNING":
            self.root.after(100, self._poll_rebuild)
            return
        self._busy = False
        self._active_rebuild = None
        self._selected_snapshot = None
        self._log(
            "rebuild complete elapsed_ms={} excluded={}".format(
                info.get("elapsed_ms"), info.get("excluded_hits")
            )
        )
        self._refresh_dashboard()

    def _check_consistency(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.status_var.set("authoritative check running")

        def worker() -> None:
            try:
                result = self.service.check_consistency()
                self.root.after(0, lambda: self._finish_check(result.canonical()))
            except Exception as exc:  # pragma: no cover - UI safety net
                self.root.after(0, lambda: self._finish_check({"code": "E_UI", "message": str(exc)}))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_check(self, result: dict[str, Any]) -> None:
        self._busy = False
        self.status_var.set(result.get("code", "checked"))
        self._log(
            "{} {} first={}/{}/{} digest={}".format(
                result.get("code"),
                result.get("message"),
                result.get("first_mismatch", {}).get("rule_id"),
                result.get("first_mismatch", {}).get("stage_id"),
                result.get("first_mismatch", {}).get("field"),
                result.get("stats_digest"),
            )
        )
        self._refresh_dashboard()

    def _recover_replay(self) -> None:
        if self._busy:
            return
        try:
            report = self.service.recover_and_replay()
            for item in report["replays"]:
                self._log(
                    "replay snapshot={} code={} first={}/{}/{}".format(
                        item["snapshot_id"],
                        item["code"],
                        item["first_mismatch"]["rule_id"],
                        item["first_mismatch"]["stage_id"],
                        item["first_mismatch"]["field"],
                    )
                )
            self._log(
                "unknown_rebuilds={} exit_report_claimed=false".format(
                    len(report["unknown_rebuilds"])
                )
            )
            self._refresh_dashboard()
        except Exception as exc:
            self._log(f"E_REPLAY {exc}")

    def _select_snapshot(self, _event: Any = None) -> None:
        selection = self.snapshot_tree.selection()
        if not selection:
            return
        row = self.service.ledger.get_snapshot(selection[0])
        if row is None:
            return
        self._selected_snapshot = Snapshot.from_row(row)
        self._populate_stats(self._selected_snapshot)
        self.summary_var.set(
            "selected snapshot={} source={} excluded={} digest={}".format(
                self._selected_snapshot.snapshot_id,
                self._selected_snapshot.source_ledger_version,
                self._selected_snapshot.excluded_hits,
                self._selected_snapshot.content_digest,
            )
        )

    def _log(self, message: str) -> None:
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def close(self) -> None:
        self.service.close()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gray snapshot diagnostics workbench")
    parser.add_argument(
        "--spec", default="samples/gateway_spec.json", help="gateway JSON declaration"
    )
    parser.add_argument(
        "--db", default="runtime/workbench.db", help="local SQLite ledger path"
    )
    args = parser.parse_args(argv)
    if not TK_AVAILABLE:
        print(
            "E_TK: Python Tkinter is unavailable. On Windows reinstall Python 3.11 "
            "with the tcl/tk component; on Linux install python3-tk.",
            file=sys.stderr,
        )
        return 2
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(
            f"E_TK_DISPLAY: Tkinter is installed but no display can be opened: {exc}",
            file=sys.stderr,
        )
        return 2
    app = WorkbenchApp(root, args.spec, args.db)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
