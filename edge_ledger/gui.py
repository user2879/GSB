from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .engine import BuildResult, build_workspace
from .report import compare_quick_view
from .workspace import WorkspaceEditor


class DiagnosticApp:
    """Tk diagnostics backed directly by the same authoritative build result."""

    def __init__(
        self,
        master: tk.Tk,
        workspace_root: Path,
        *,
        auto_build: bool = True,
    ) -> None:
        self.master = master
        self.workspace_root = Path(workspace_root)
        self.master.title("Edge Ledger Diagnostics")
        self.master.geometry("1280x820")
        self.result: BuildResult | None = None
        self.selected_template_key: tuple[str, int] | None = None
        self.threshold_var = tk.StringVar(value="720")
        self.status_var = tk.StringVar(value="Not built")
        self.quick_match_var = tk.StringVar(value="Quick view: pending")
        self._build_ui()
        if auto_build:
            self.rebuild()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.master, padding=(8, 6))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Refresh / Build", command=self.rebuild).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Dedupe threshold permille:").pack(side=tk.LEFT, padx=(16, 4))
        self.threshold_spin = ttk.Spinbox(
            toolbar,
            from_=0,
            to=1000,
            increment=1,
            width=8,
            textvariable=self.threshold_var,
        )
        self.threshold_spin.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Apply Threshold", command=self.apply_threshold).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(toolbar, text="Write Report", command=self.write_report).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Label(toolbar, textvariable=self.quick_match_var).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        self.template_tab = ttk.Frame(self.notebook)
        self.sample_tab = ttk.Frame(self.notebook)
        self.ledger_tab = ttk.Frame(self.notebook)
        self.detail_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.template_tab, text="Templates")
        self.notebook.add(self.sample_tab, text="Samples")
        self.notebook.add(self.ledger_tab, text="Ledger / Queue")
        self.notebook.add(self.detail_tab, text="Editor / Details")
        self._build_template_tab()
        self._build_sample_tab()
        self._build_ledger_tab()
        self._build_detail_tab()
        footer = ttk.Frame(self.master, padding=(8, 4))
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(footer, textvariable=self.status_var).pack(side=tk.LEFT)

    def _build_template_tab(self) -> None:
        paned = ttk.PanedWindow(self.template_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)
        columns = ("template", "revision", "parent", "path", "slots", "fingerprint")
        self.template_tree = ttk.Treeview(left, columns=columns, show="headings")
        for column, heading, width in (
            ("template", "Template", 150),
            ("revision", "Rev", 55),
            ("parent", "Parent", 65),
            ("path", "Revision path", 100),
            ("slots", "Effective slots", 320),
            ("fingerprint", "Chain fingerprint", 260),
        ):
            self.template_tree.heading(column, text=heading)
            self.template_tree.column(column, width=width, anchor=tk.W)
        self.template_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.template_tree.yview)
        scroll.pack(fill=tk.Y, side=tk.RIGHT)
        self.template_tree.configure(yscrollcommand=scroll.set)
        self.template_tree.bind("<<TreeviewSelect>>", self._on_template_select)
        ttk.Label(right, text="Selected effective template").pack(anchor=tk.W)
        self.template_detail = tk.Text(right, wrap=tk.WORD, height=28)
        self.template_detail.pack(fill=tk.BOTH, expand=True)

    def _build_sample_tab(self) -> None:
        columns = (
            "sample",
            "shard",
            "template",
            "status",
            "score",
            "residual",
            "cluster",
            "dedupe",
            "render",
        )
        self.sample_tree = ttk.Treeview(self.sample_tab, columns=columns, show="headings")
        for column, heading, width in (
            ("sample", "Sample", 150),
            ("shard", "Shard", 110),
            ("template", "Template revision", 160),
            ("status", "Status", 90),
            ("score", "Quality", 90),
            ("residual", "Residual", 80),
            ("cluster", "Cluster", 140),
            ("dedupe", "Dedupe", 80),
            ("render", "Render digest", 330),
        ):
            self.sample_tree.heading(column, text=heading)
            self.sample_tree.column(column, width=width, anchor=tk.W)
        y_scroll = ttk.Scrollbar(self.sample_tab, orient=tk.VERTICAL, command=self.sample_tree.yview)
        x_scroll = ttk.Scrollbar(self.sample_tab, orient=tk.HORIZONTAL, command=self.sample_tree.xview)
        self.sample_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.sample_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        y_scroll.pack(fill=tk.Y, side=tk.RIGHT)
        x_scroll.pack(fill=tk.X, side=tk.BOTTOM)
        self.sample_tree.bind("<<TreeviewSelect>>", self._on_sample_select)

    def _build_ledger_tab(self) -> None:
        upper = ttk.Frame(self.ledger_tab)
        upper.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(upper)
        right = ttk.Frame(upper)
        left.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        right.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
        self.failure_tree = ttk.Treeview(
            left,
            columns=("stage", "code", "target", "injection", "message"),
            show="headings",
        )
        for column, heading, width in (
            ("stage", "Stage", 110),
            ("code", "Code", 160),
            ("target", "Target", 190),
            ("injection", "Injection", 150),
            ("message", "Message", 400),
        ):
            self.failure_tree.heading(column, text=heading)
            self.failure_tree.column(column, width=width, anchor=tk.W)
        self.failure_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Pending queue / cache ledger").pack(anchor=tk.W)
        self.ledger_tree = ttk.Treeview(
            right,
            columns=("kind", "entity", "action", "detail"),
            show="headings",
        )
        for column, heading, width in (
            ("kind", "Kind", 100),
            ("entity", "Entity", 180),
            ("action", "Action", 100),
            ("detail", "Detail", 600),
        ):
            self.ledger_tree.heading(column, text=heading)
            self.ledger_tree.column(column, width=width, anchor=tk.W)
        self.ledger_tree.pack(fill=tk.BOTH, expand=True)
        self.code_summary = tk.Text(self.ledger_tab, height=5, wrap=tk.WORD)
        self.code_summary.pack(fill=tk.X)

    def _build_detail_tab(self) -> None:
        editor = ttk.LabelFrame(self.detail_tab, text="Edit selected template revision")
        editor.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(editor, text="Instruction template:").grid(row=0, column=0, sticky=tk.NW)
        self.instruction_edit = tk.Text(editor, height=4, width=100)
        self.instruction_edit.grid(row=0, column=1, sticky=tk.EW, padx=6, pady=3)
        ttk.Label(editor, text="Answer template:").grid(row=1, column=0, sticky=tk.NW)
        self.answer_edit = tk.Text(editor, height=4, width=100)
        self.answer_edit.grid(row=1, column=1, sticky=tk.EW, padx=6, pady=3)
        ttk.Label(editor, text="Slot declarations (JSON):").grid(row=2, column=0, sticky=tk.NW)
        self.slots_edit = tk.Text(editor, height=4, width=100)
        self.slots_edit.grid(row=2, column=1, sticky=tk.EW, padx=6, pady=3)
        ttk.Button(editor, text="Save Revision + Rebuild", command=self.save_template_edit).grid(
            row=3, column=1, sticky=tk.E, padx=6, pady=4
        )
        editor.columnconfigure(1, weight=1)
        self.detail_text = tk.Text(self.detail_tab, wrap=tk.WORD)
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

    def rebuild(self) -> None:
        try:
            self.result = build_workspace(self.workspace_root)
        except Exception as exc:
            messagebox.showerror("Build failed", str(exc), parent=self.master)
            self.status_var.set(f"Build failed: {exc}")
            return
        self.threshold_var.set(str(self._threshold()))
        self.refresh()

    def _threshold(self) -> int:
        if self.result is None:
            return 720
        # The result does not need to carry raw config; read the one used by stats.
        config_path = self.workspace_root / "config" / "pipeline.yaml"
        try:
            import yaml

            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            return int(payload.get("dedupe", {}).get("threshold_permille", 720))
        except Exception:
            return 720

    def refresh(self) -> None:
        if self.result is None:
            return
        for tree in (self.template_tree, self.sample_tree, self.failure_tree, self.ledger_tree):
            for item in tree.get_children():
                tree.delete(item)
        result = self.result.result
        for template in result["templates"]:
            slots = ", ".join(
                f"{slot['name']}({slot['source_revision']})" for slot in template["slots"]
            )
            self.template_tree.insert(
                "",
                tk.END,
                iid=f"{template['template_id']}@{template['revision']}",
                values=(
                    template["template_id"],
                    template["revision"],
                    template.get("parent_revision", ""),
                    ",".join(str(item) for item in template.get("revision_path", [])),
                    slots,
                    template.get("chain_fingerprint", ""),
                ),
            )
        for sample in result["samples"]:
            self.sample_tree.insert(
                "",
                tk.END,
                iid=sample["sample_id"],
                values=(
                    sample["sample_id"],
                    sample["source_shard"],
                    f"{sample['template_id']}@{sample['template_revision']}",
                    sample["status"],
                    sample.get("quality_score", ""),
                    sample.get("quality_residual", ""),
                    sample.get("cluster_id", ""),
                    sample.get("dedupe_status", ""),
                    sample.get("render_digest", ""),
                ),
            )
        for failure in result["failures"]:
            self.failure_tree.insert(
                "",
                tk.END,
                values=(
                    failure["stage"],
                    failure["code"],
                    f"{failure['target_kind']}:{failure['target_id']}",
                    failure.get("injection_id") or "",
                    failure["message"],
                ),
            )
        for pending in result["pending_queue"]:
            self.ledger_tree.insert(
                "",
                tk.END,
                values=("pending", pending["sample_id"], pending["code"], pending["sort_key"]),
            )
        for entry in result["execution"]["cache_ledger"]:
            self.ledger_tree.insert(
                "",
                tk.END,
                values=(entry["stage"], entry["entity_id"], entry["action"], entry["basis"]),
            )
        code_text = (
            f"Recovery: {json.dumps(result['execution']['recovery'], ensure_ascii=False)}\n"
            f"Business codes: {json.dumps(result['stats']['code_counts'], ensure_ascii=False)}\n"
            f"Snapshot generation: {self.result.snapshot_generation}\n"
            f"Runtime: {self.result.runtime_seconds * 1000:.2f} ms\n"
            f"Render points: {result['stats']['total']['render_points_consumed']}\n"
        )
        self.code_summary.delete("1.0", tk.END)
        self.code_summary.insert("1.0", code_text)
        comparison = compare_quick_view(result)
        self.quick_match_var.set(
            "Quick view: MATCH" if comparison["matches"] else "Quick view: DIFFERENT"
        )
        self.status_var.set(
            f"Built generation {self.result.snapshot_generation}; "
            f"{len(result['samples'])} sample rows; {len(result['failures'])} failures"
        )
        self.threshold_var.set(str(self._threshold()))

    def _on_template_select(self, _event: Any = None) -> None:
        selection = self.template_tree.selection()
        if not selection:
            return
        key = selection[0]
        try:
            template_id, revision_text = key.rsplit("@", 1)
            self.selected_template_key = (template_id, int(revision_text))
        except ValueError:
            return
        detail = next(
            (
                item
                for item in self.result.result["templates"]
                if item["template_id"] == template_id
                and item["revision"] == int(revision_text)
            ),
            None,
        ) if self.result else None
        if detail is None:
            return
        self.template_detail.delete("1.0", tk.END)
        self.template_detail.insert(
            "1.0", json.dumps(detail, ensure_ascii=False, sort_keys=True, indent=2)
        )
        raw = self._raw_revision(template_id, int(revision_text))
        if raw is not None:
            self.instruction_edit.delete("1.0", tk.END)
            self.answer_edit.delete("1.0", tk.END)
            self.slots_edit.delete("1.0", tk.END)
            self.instruction_edit.insert("1.0", raw.get("instruction_template") or raw.get("instruction") or "")
            self.answer_edit.insert("1.0", raw.get("answer_template") or raw.get("answer") or "")
            self.slots_edit.insert(
                "1.0", json.dumps(raw.get("slots", []), ensure_ascii=False, indent=2)
            )

    def _raw_revision(self, template_id: str, revision: int) -> dict[str, Any] | None:
        editor = WorkspaceEditor.from_root(self.workspace_root)
        for item in editor.templates:
            if item.get("template_id") == template_id and item.get("revision") == revision:
                return item
        return None

    def _on_sample_select(self, _event: Any = None) -> None:
        selection = self.sample_tree.selection()
        if not selection or self.result is None:
            return
        sample_id = selection[0]
        sample = next(
            (item for item in self.result.result["samples"] if item["sample_id"] == sample_id),
            None,
        )
        self.detail_text.delete("1.0", tk.END)
        if sample is not None:
            self.detail_text.insert(
                "1.0", json.dumps(sample, ensure_ascii=False, sort_keys=True, indent=2)
            )

    def apply_threshold(self) -> None:
        try:
            value = int(self.threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid threshold", "Threshold must be an integer", parent=self.master)
            return
        if not 0 <= value <= 1000:
            messagebox.showerror("Invalid threshold", "Threshold must be between 0 and 1000", parent=self.master)
            return
        editor = WorkspaceEditor.from_root(self.workspace_root)
        editor.config.setdefault("dedupe", {})["threshold_permille"] = value
        editor.save()
        self.rebuild()

    def save_template_edit(self) -> None:
        if self.selected_template_key is None:
            messagebox.showinfo("No selection", "Select a template revision first", parent=self.master)
            return
        template_id, revision = self.selected_template_key
        try:
            slots = json.loads(self.slots_edit.get("1.0", tk.END).strip() or "[]")
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid slots", str(exc), parent=self.master)
            return
        editor = WorkspaceEditor.from_root(self.workspace_root)
        found = None
        for item in editor.templates:
            if item.get("template_id") == template_id and item.get("revision") == revision:
                found = item
                break
        if found is None:
            messagebox.showerror("Missing revision", "Selected revision is no longer present", parent=self.master)
            return
        found["instruction_template"] = self.instruction_edit.get("1.0", tk.END).rstrip("\n")
        found["answer_template"] = self.answer_edit.get("1.0", tk.END).rstrip("\n")
        found["slots"] = slots
        editor.save()
        self.rebuild()

    def write_report(self) -> None:
        if self.result is None:
            self.rebuild()
        from .report import write_reports

        paths = write_reports(self.result)
        messagebox.showinfo("Report written", "\n".join(str(path) for path in paths.values()), parent=self.master)


def launch_gui(root: Path) -> None:
    master = tk.Tk()
    app = DiagnosticApp(master, root)
    master.mainloop()
