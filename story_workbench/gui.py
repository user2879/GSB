"""Tkinter desktop timeline workbench reading immutable snapshots only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .spec import load_spec
from .timeline import NarrativeSession, Snapshot, TimelineResult

try:  # Keep the core importable on headless build machines.
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError:  # pragma: no cover - platform dependent
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    ScrolledText = None  # type: ignore[assignment]


def _fmt(value: float) -> str:
    return repr(float(value))


def _format_snapshot(snapshot: Snapshot) -> str:
    lines = [
        f"版本戳  {snapshot.version_stamp}",
        f"内容哈希 {snapshot.content_hash}",
        f"状态哈希 {snapshot.state_hash}",
        f"节点     {snapshot.node_id} — {snapshot.node_title}",
        f"状态     {snapshot.status}",
        f"结局     {snapshot.ending or '—'}",
        f"边序列   {' -> '.join(snapshot.edge_history) if snapshot.edge_history else '（起点）'}",
        "",
        "状态向量（声明顺序）",
    ]
    for index, component in enumerate(snapshot.components):
        chain = snapshot.causal_chains[index]
        edges = " -> ".join(chain.edge_ids()) if chain.edge_ids() else "初值"
        lines.append(f"  {component:<16} {_fmt(snapshot.state[index]):<22} 来源: {edges}")
    lines.extend(["", "边界判定清单"])
    if snapshot.boundary_judgments:
        for item in snapshot.boundary_judgments:
            lines.append(
                f"  {item['edge_id']} / {item['condition_id']}: value={_fmt(item['value'])}, "
                f"threshold={_fmt(item['threshold'])}, signed_margin={_fmt(item['signed_margin'])}"
            )
    else:
        lines.append("  无")
    return "\n".join(lines)


def _format_branches(snapshot: Snapshot) -> str:
    lines = ["后续分支与条件裕度", ""]
    if not snapshot.branches:
        lines.append("  （无出边）")
        return "\n".join(lines)
    for branch in snapshot.branches:
        lines.append(
            f"[{branch.status.upper():<11}] {branch.edge_id}  {branch.label} -> {branch.target_node}"
        )
        if not branch.conditions:
            lines.append("    no conditions")
        for condition in branch.conditions:
            boundary_mark = "  BOUNDARY" if condition.boundary else ""
            lines.append(
                f"    {condition.component} {condition.op} {_fmt(condition.threshold)} | "
                f"value={_fmt(condition.value)} signed_margin={_fmt(condition.signed_margin)} "
                f"status={condition.status}{boundary_mark}"
            )
        lines.append("")
    return "\n".join(lines)


def _format_causality(snapshot: Snapshot) -> str:
    lines = ["逐分量因果链", ""]
    for chain in snapshot.causal_chains:
        lines.append(f"{chain.component}: initial={_fmt(chain.initial)} final={_fmt(chain.value)}")
        if not chain.contributions:
            lines.append("  （无效果贡献）")
        for contribution in chain.contributions:
            lines.append(
                f"  {contribution.edge_id}[{contribution.edge_index}] / {contribution.effect_id}: "
                f"{_fmt(contribution.before)} + {_fmt(contribution.delta)} = {_fmt(contribution.after)}"
            )
        lines.append("")
    return "\n".join(lines)


def _format_conflicts(snapshot: Snapshot) -> str:
    lines = ["冲突裁决账本", ""]
    if snapshot.arbitrations:
        for arbitration in snapshot.arbitrations:
            lines.append(
                f"实际边效果: {arbitration.edge_id} / {arbitration.target} | "
                f"覆盖 {arbitration.overwritten_effect_id} ({_fmt(arbitration.overwritten_value)}) -> "
                f"生效 {arbitration.winner_effect_id} ({_fmt(arbitration.effective_value)}) | "
                f"reason={arbitration.reason}"
            )
    else:
        lines.append("实际边效果：无覆盖")
    lines.extend(["", "同节点出边叠加投影"])
    projected = snapshot.outgoing_projection.get("arbitrations", [])
    if not projected:
        lines.append("  无覆盖")
    for arbitration in projected:
        lines.append(
            f"  {arbitration['target']}: 覆盖 {arbitration['overwritten_effect_id']} -> "
            f"生效 {arbitration['winner_effect_id']} ({arbitration['reason']})"
        )
    lines.extend(["", "投影状态（仅诊断，不写入互斥分支的时间轴）"])
    projected_state = snapshot.outgoing_projection.get("projected_state", {})
    for component in snapshot.components:
        if component in projected_state:
            lines.append(f"  {component}: {_fmt(projected_state[component])}")
    return "\n".join(lines)


def launch(spec_path: str | Path, sequence_id: str | None = None) -> int:
    if tk is None:
        raise WorkbenchError(
            "E_GUI_UNAVAILABLE",
            "Tkinter is unavailable in this Python environment; install the Windows official Python package or python3-tk",
            details={"hint": "python -m tkinter"},
        )

    root = tk.Tk()
    root.title("叙事分支与事件因果工作台")
    root.geometry("1520x900")

    state: dict[str, Any] = {"path": Path(spec_path), "result": None, "updating_selection": False}

    toolbar = ttk.Frame(root, padding=(8, 6))
    toolbar.pack(fill="x")
    path_var = tk.StringVar(value=str(state["path"]))
    sequence_var = tk.StringVar(value=sequence_id or "")
    position_var = tk.StringVar(value="")

    path_label = ttk.Label(toolbar, textvariable=path_var, width=50)
    path_label.pack(side="left", padx=(0, 8))

    sequence_box = ttk.Combobox(toolbar, textvariable=sequence_var, state="readonly", width=24)
    sequence_box.pack(side="left", padx=(0, 8))

    tree_pane = ttk.Panedwindow(root, orient="horizontal")
    tree_pane.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    left = ttk.Frame(tree_pane)
    middle = ttk.Frame(tree_pane)
    right = ttk.Frame(tree_pane)
    tree_pane.add(left, weight=1)
    tree_pane.add(middle, weight=3)
    tree_pane.add(right, weight=3)

    timeline_list = tk.Listbox(left, exportselection=False, font=("Consolas", 10))
    timeline_list.pack(fill="both", expand=True)

    middle_tabs = ttk.Notebook(middle)
    middle_tabs.pack(fill="both", expand=True)
    state_tab = ttk.Frame(middle_tabs)
    branches_tab = ttk.Frame(middle_tabs)
    middle_tabs.add(state_tab, text="状态 / 边界")
    middle_tabs.add(branches_tab, text="分支 / 裕度")
    state_text = ScrolledText(state_tab, wrap="word", font=("Consolas", 10))
    state_text.pack(fill="both", expand=True)
    branches_text = ScrolledText(branches_tab, wrap="word", font=("Consolas", 10))
    branches_text.pack(fill="both", expand=True)

    right_tabs = ttk.Notebook(right)
    right_tabs.pack(fill="both", expand=True)
    causal_tab = ttk.Frame(right_tabs)
    conflict_tab = ttk.Frame(right_tabs)
    right_tabs.add(causal_tab, text="因果链")
    right_tabs.add(conflict_tab, text="裁决")
    causal_text = ScrolledText(causal_tab, wrap="word", font=("Consolas", 10))
    causal_text.pack(fill="both", expand=True)
    conflict_text = ScrolledText(conflict_tab, wrap="word", font=("Consolas", 10))
    conflict_text.pack(fill="both", expand=True)

    def show_index(index: int) -> None:
        if state["updating_selection"]:
            return
        result: TimelineResult | None = state["result"]
        if result is None or not result.snapshots:
            return
        state["updating_selection"] = True
        try:
            index = max(0, min(index, len(result.snapshots) - 1))
            timeline_list.selection_clear(0, "end")
            timeline_list.selection_set(index)
            timeline_list.activate(index)
        finally:
            state["updating_selection"] = False
        snapshot = result.snapshots[index]
        position_var.set(
            f"快照 {index + 1}/{len(result.snapshots)}  {snapshot.version_stamp}  结局={result.ending or '—'}"
        )
        state_text.configure(state="normal")
        state_text.delete("1.0", "end")
        state_text.insert("1.0", _format_snapshot(snapshot))
        state_text.configure(state="disabled")
        branches_text.configure(state="normal")
        branches_text.delete("1.0", "end")
        branches_text.insert("1.0", _format_branches(snapshot))
        branches_text.configure(state="disabled")
        causal_text.configure(state="normal")
        causal_text.delete("1.0", "end")
        causal_text.insert("1.0", _format_causality(snapshot))
        causal_text.configure(state="disabled")
        conflict_text.configure(state="normal")
        conflict_text.delete("1.0", "end")
        conflict_text.insert("1.0", _format_conflicts(snapshot))
        conflict_text.configure(state="disabled")

    def rebuild_timeline(selected_sequence: str | None = None) -> None:
        try:
            spec = load_spec(state["path"])
        except WorkbenchError as exc:
            messagebox.showerror("规格读取失败", exc.message)
            return
        sequence_ids = [sample.id for sample in spec.samples]
        sequence_box["values"] = sequence_ids
        chosen = selected_sequence or sequence_var.get() or (sequence_ids[0] if sequence_ids else "")
        if chosen not in sequence_ids:
            chosen = sequence_ids[0] if sequence_ids else ""
        sequence_var.set(chosen)
        if not chosen:
            messagebox.showerror("缺少样本", "该规格没有选择序列样本。")
            return
        try:
            result = NarrativeSession(spec, chosen).run()
        except WorkbenchError as exc:
            messagebox.showerror("重放失败", exc.message)
            return
        state["result"] = result
        timeline_list.delete(0, "end")
        for snapshot in result.snapshots:
            timeline_list.insert(
                "end",
                f"v{snapshot.state_version:04d}  {snapshot.node_id:<16} {snapshot.status}",
            )
        show_index(0)

    def open_file() -> None:
        selected = filedialog.askopenfilename(
            title="打开叙事规格",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if selected:
            state["path"] = Path(selected)
            path_var.set(selected)
            rebuild_timeline()

    def step(delta: int) -> None:
        selection = timeline_list.curselection()
        current = selection[0] if selection else 0
        show_index(current + delta)

    ttk.Button(toolbar, text="打开 YAML", command=open_file).pack(side="left", padx=(0, 8))
    ttk.Button(toolbar, text="上一个", command=lambda: step(-1)).pack(side="left", padx=(0, 4))
    ttk.Button(toolbar, text="下一个", command=lambda: step(1)).pack(side="left", padx=(0, 12))
    ttk.Label(toolbar, textvariable=position_var).pack(side="left")
    sequence_box.bind("<<ComboboxSelected>>", lambda _event: rebuild_timeline(sequence_var.get()))
    timeline_list.bind("<<ListboxSelect>>", lambda _event: show_index(timeline_list.curselection()[0]) if timeline_list.curselection() else None)

    rebuild_timeline(sequence_id)
    root.mainloop()
    return 0
