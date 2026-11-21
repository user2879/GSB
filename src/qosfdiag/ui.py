"""Tkinter diagnostics UI, imported without requiring Tk at package import time."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine import CascadeEngine, EngineRunResult
from .error_propagation import compare_configurations, propagate_fixed_errors
from .errors import ReplayError
from .generator import DeterministicUdpGenerator
from .ledger import DecisionLedger
from .models import ErrorPropagationResult, IndistinguishabilityResult, PolicySet
from .policy import load_policy
from .replay import verify_snapshot_replay
from .samples import (
    three_stage_generator,
    three_stage_near_equal_policy,
    three_stage_policy,
)
from .snapshot import SnapshotDocument, SnapshotManager
from .transport import InMemoryTransport
from .ui_model import DiagnosticView, build_view


DEMO_SCALE = 3


def launch_ui(
    *,
    policy_path: str | Path | None = None,
    generator_path: str | Path | None = None,
    workdir: str | Path = "artifacts/ui",
) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable. On Windows use the official Python installer "
            "with 'tcl/tk and IDLE'; on Debian/Ubuntu install python3-tk."
        ) from exc
    try:
        root = tk.Tk()
    except Exception as exc:
        raise RuntimeError(
            "Tkinter could not open a display. Run this command in a desktop session."
        ) from exc
    app = DiagnosticUi(
        root=root,
        tk=tk,
        ttk=ttk,
        filedialog=filedialog,
        messagebox=messagebox,
        policy_path=policy_path,
        generator_path=generator_path,
        workdir=Path(workdir),
    )
    root.mainloop()
    del app


class DiagnosticUi:
    def __init__(
        self,
        *,
        root: Any,
        tk: Any,
        ttk: Any,
        filedialog: Any,
        messagebox: Any,
        policy_path: str | Path | None,
        generator_path: str | Path | None,
        workdir: Path,
    ) -> None:
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.policy_path = Path(policy_path) if policy_path else None
        self.generator_path = Path(generator_path) if generator_path else None
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.snapshot_manager = SnapshotManager(self.workdir / "snapshots")
        self.current_policy: PolicySet | None = None
        self.current_generator: DeterministicUdpGenerator | None = None
        self.current_run: EngineRunResult | None = None
        self.current_errors: ErrorPropagationResult | None = None
        self.current_ledger_path = self.workdir / "ui-ledger.sqlite"
        self.current_snapshot: SnapshotDocument | None = None
        self._build()

    def _build(self) -> None:
        self.root.title("QoS Fixed-Point Diagnostic")
        self.root.geometry("1280x800")
        toolbar = self.ttk.Frame(self.root)
        toolbar.pack(side="top", fill="x", padx=6, pady=4)
        buttons = (
            ("Load policy", self.load_policy),
            ("Load generator", self.load_generator),
            ("Run", self.run_diagnostic),
            ("Save snapshot", self.save_snapshot),
            ("Replay selected", self.replay_selected),
            ("Refresh snapshots", self.refresh_snapshots),
        )
        for label, command in buttons:
            self.ttk.Button(toolbar, text=label, command=command).pack(
                side="left", padx=3
            )
        self.status = self.tk.StringVar(value="Ready")
        self.ttk.Label(toolbar, textvariable=self.status).pack(side="right", padx=8)

        notebook = self.ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=6, pady=4)
        self.notebook = notebook

        overview = self.ttk.Frame(notebook)
        notebook.add(overview, text="Overview")
        self._build_overview(overview)

        timeline = self.ttk.Frame(notebook)
        notebook.add(timeline, text="Timeline")
        self._build_timeline(timeline)

        heat = self.ttk.Frame(notebook)
        notebook.add(heat, text="Error heat")
        self._build_heat(heat)

        snapshots = self.ttk.Frame(notebook)
        notebook.add(snapshots, text="Snapshots")
        self._build_snapshots(snapshots)
        self.refresh_snapshots()

    def _tree(
        self,
        parent: Any,
        columns: tuple[str, ...],
        *,
        height: int = 18,
    ) -> Any:
        frame = self.ttk.Frame(parent)
        tree = self.ttk.Treeview(frame, columns=columns, show="headings", height=height)
        y_scroll = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = self.ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        for column in columns:
            tree.heading(column, text=column.replace("_", " ").title())
            width = 110 if column in {"stage_id", "decision", "kind"} else 130
            tree.column(column, width=width, anchor="w", stretch=True)
        return frame, tree

    def _build_overview(self, parent: Any) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=2)
        parent.grid_rowconfigure(1, weight=3)
        parent.grid_rowconfigure(2, weight=2)
        self.topology_text = self.tk.Text(parent, height=8, wrap="none")
        self.topology_text.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        stats_frame, self.stats_tree = self._tree(
            parent,
            ("stage_id", "passed", "queued", "dropped", "released", "bytes", "maxq"),
        )
        stats_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.summary_text = self.tk.Text(parent, height=12, wrap="word")
        self.summary_text.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.indist_tree_frame, self.indist_tree = self._tree(
            parent,
            (
                "indistinguishable",
                "released_bytes_difference",
                "combined_bytes_error_bound",
                "winner",
            ),
            height=5,
        )
        self.indist_tree_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)

    def _build_timeline(self, parent: Any) -> None:
        frame, self.timeline_tree = self._tree(
            parent,
            (
                "event_seq",
                "packet_id",
                "stage_id",
                "kind",
                "decision",
                "time_ns",
                "tokens",
                "queue",
                "delay_ns",
            ),
            height=30,
        )
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.timeline_tree.tag_configure("PASS", background="#e8f5e9")
        self.timeline_tree.tag_configure("RELEASE", background="#e3f2fd")
        self.timeline_tree.tag_configure("QUEUE", background="#fff8e1")
        self.timeline_tree.tag_configure("DROP", background="#ffebee")

    def _build_heat(self, parent: Any) -> None:
        frame, self.heat_tree = self._tree(
            parent,
            (
                "stage_id",
                "bytes_contribution",
                "bytes_share",
                "heat",
                "delay_contribution_ns",
                "delay_share",
                "dominant",
            ),
            height=18,
        )
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.heat_tree.tag_configure("heat_high", background="#ff8a65")
        self.heat_tree.tag_configure("heat_mid", background="#ffcc80")
        self.heat_tree.tag_configure("heat_low", background="#fff9c4")
        self.heat_tree.tag_configure("dominant", background="#ffd54f")

    def _build_snapshots(self, parent: Any) -> None:
        frame, self.snapshot_tree = self._tree(
            parent,
            ("filename", "policy_version", "scale", "cursor_seq", "content_digest"),
            height=28,
        )
        frame.pack(fill="both", expand=True, padx=4, pady=4)

    def load_policy(self) -> None:
        filename = self.filedialog.askopenfilename(
            title="Select policy JSON", filetypes=(("JSON", "*.json"), ("All", "*.*"))
        )
        if filename:
            self.policy_path = Path(filename)
            self.status.set(f"Policy selected: {self.policy_path.name}")

    def load_generator(self) -> None:
        filename = self.filedialog.askopenfilename(
            title="Select generator JSON", filetypes=(("JSON", "*.json"), ("All", "*.*"))
        )
        if filename:
            self.generator_path = Path(filename)
            self.status.set(f"Generator selected: {self.generator_path.name}")

    def _clean_ledger(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.current_ledger_path) + suffix)
            if path.exists():
                path.unlink()

    def _near_equal_comparison(
        self, policy: PolicySet, generator: DeterministicUdpGenerator
    ) -> IndistinguishabilityResult | None:
        if self.current_errors is None:
            return None
        if policy.policy_version != "three-stage-v1":
            return None
        near_policy = load_policy(three_stage_near_equal_policy())
        with CascadeEngine(
            near_policy,
            generator,
            scale=DEMO_SCALE,
            transport=InMemoryTransport([item.shaper_id for item in near_policy.shapers]),
        ) as engine:
            near_run = engine.run()
        near_errors = propagate_fixed_errors(
            near_policy,
            near_run.summary,
            near_run.events,
            scale=DEMO_SCALE,
            generator_digest=generator.sequence_digest,
        )
        return compare_configurations(self.current_errors, near_errors)

    def run_diagnostic(self) -> None:
        try:
            policy = (
                load_policy(self.policy_path)
                if self.policy_path is not None
                else load_policy(three_stage_policy())
            )
            generator = (
                DeterministicUdpGenerator(self.generator_path)
                if self.generator_path is not None
                else DeterministicUdpGenerator(three_stage_generator())
            )
            self._clean_ledger()
            with DecisionLedger(self.current_ledger_path) as ledger:
                with CascadeEngine(
                    policy,
                    generator,
                    scale=DEMO_SCALE,
                    use_loopback=True,
                    base_port=54500,
                ) as engine:
                    run = engine.run(ledger=ledger)
                errors = propagate_fixed_errors(
                    policy,
                    run.summary,
                    run.events,
                    scale=DEMO_SCALE,
                    generator_digest=generator.sequence_digest,
                )
            self.current_policy = policy
            self.current_generator = generator
            self.current_run = run
            self.current_errors = errors
            comparison = self._near_equal_comparison(policy, generator)
            self.current_snapshot = self._save_current_snapshot()
            view = build_view(
                policy=policy,
                run=run,
                errors=errors,
                indistinguishability=() if comparison is None else (comparison,),
                snapshots=self.snapshot_manager.list_snapshots(),
            )
            self.render(view)
            self.status.set(
                f"Run complete: {run.summary.final_released_packets}/"
                f"{run.summary.total_generated_packets} released"
            )
        except Exception as exc:
            self.messagebox.showerror("QoS diagnostic error", str(exc))
            self.status.set("Run failed")

    def _save_current_snapshot(self) -> SnapshotDocument | None:
        if (
            self.current_policy is None
            or self.current_generator is None
            or self.current_errors is None
        ):
            return None
        with DecisionLedger(self.current_ledger_path) as ledger:
            ledger.initialize(
                policy=self.current_policy,
                scale=DEMO_SCALE,
                generator=self.current_generator,
            )
            filename = f"{self.current_policy.policy_version}.json"
            return self.snapshot_manager.save(
                filename=filename,
                policy=self.current_policy,
                scale=DEMO_SCALE,
                generator=self.current_generator,
                ledger=ledger,
                errors=self.current_errors,
            )

    def save_snapshot(self) -> None:
        try:
            self.current_snapshot = self._save_current_snapshot()
            self.refresh_snapshots()
            self.status.set("Snapshot saved")
        except Exception as exc:
            self.messagebox.showerror("Snapshot error", str(exc))

    def replay_selected(self) -> None:
        selection = self.snapshot_tree.selection()
        if not selection:
            self.messagebox.showinfo("Replay", "Select a snapshot row first")
            return
        filename = self.snapshot_tree.item(selection[0], "values")[0]
        path = self.snapshot_manager.directory / str(filename)
        try:
            result = verify_snapshot_replay(
                path,
                current_policy_version=(
                    None if self.current_policy is None else self.current_policy.policy_version
                ),
                current_scale=DEMO_SCALE,
                current_generator_digest=(
                    None
                    if self.current_generator is None
                    else self.current_generator.sequence_digest
                ),
            )
            self.messagebox.showinfo(
                "Replay",
                f"REPRODUCIBLE\nEvents compared: {result.events_compared}\n"
                f"Released bytes: {result.summary.final_released_bytes} +/- "
                f"{result.errors.bytes_error_bound}",
            )
            self.status.set("Replay verified")
        except ReplayError as exc:
            classification = exc.details.get("classification", "UNKNOWN")
            self.messagebox.showerror(
                f"E_REPLAY ({classification})", str(exc)
            )
            self.status.set("Replay mismatch")
        except Exception as exc:
            self.messagebox.showerror("Replay error", str(exc))

    def refresh_snapshots(self) -> None:
        for item in self.snapshot_tree.get_children():
            self.snapshot_tree.delete(item)
        for row in self.snapshot_manager.list_snapshots():
            self.snapshot_tree.insert(
                "",
                "end",
                values=(
                    row["filename"],
                    row["policy_version"],
                    row["scale"],
                    row["cursor_seq"],
                    row["content_digest"],
                ),
            )

    def _set_text(self, widget: Any, lines: list[str]) -> None:
        widget.delete("1.0", "end")
        widget.insert("1.0", "\n".join(lines))

    def render(self, view: DiagnosticView) -> None:
        self._set_text(self.topology_text, list(view.topology_lines))
        summary_lines = list(view.error_summary)
        summary_lines.append(f"dominant_stage={view.dominant_stage}")
        self._set_text(self.summary_text, summary_lines)
        for tree in (self.stats_tree, self.timeline_tree, self.heat_tree):
            for item in tree.get_children():
                tree.delete(item)
        for row in view.stage_rows:
            self.stats_tree.insert(
                "",
                "end",
                values=(
                    row["stage_id"],
                    row["passed"],
                    row["queued"],
                    row["dropped"],
                    row["released"],
                    row["bytes"],
                    row["maxq"],
                ),
            )
        for row in view.timeline_rows:
            self.timeline_tree.insert(
                "",
                "end",
                values=(
                    row["event_seq"],
                    row["packet_id"],
                    row["stage_id"],
                    row["kind"],
                    row["decision"],
                    row["time_ns"],
                    row["tokens"],
                    row["queue"],
                    row["delay_ns"],
                ),
                tags=(row["decision"],),
            )
        for row in view.error_heat_rows:
            self.heat_tree.insert(
                "",
                "end",
                values=(
                    row["stage_id"],
                    row["bytes_contribution"],
                    row["bytes_share"],
                    row["heat"],
                    row["delay_contribution_ns"],
                    row["delay_share"],
                    row["dominant"],
                ),
                tags=(
                    ("dominant",) if row["dominant"] else ()
                )
                + (
                    ("heat_high",)
                    if row["bytes_share_value"] >= 0.50
                    else ("heat_mid",)
                    if row["bytes_share_value"] >= 0.25
                    else ("heat_low",)
                ),
            )
        for item in self.indist_tree.get_children():
            self.indist_tree.delete(item)
        for row in view.indistinguishability_rows:
            self.indist_tree.insert(
                "",
                "end",
                values=(
                    row["indistinguishable"],
                    row["released_bytes_difference"],
                    row["combined_bytes_error_bound"],
                    row["winner"],
                ),
            )
        self.refresh_snapshots()
