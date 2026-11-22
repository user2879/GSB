"""Headless diagnostic view model and optional Tkinter renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bitmap import bitmap_overview, popcount, set_bits
from .errors import DiagnosticError, E_STATE, E_UI
from .store import Session


def default_target(session: Session) -> str | None:
    rows = session.snapshot_rows()
    if not rows:
        return None
    session_file = session.root / "session.json"
    if session_file.exists():
        try:
            declaration = json.loads(session_file.read_text(encoding="utf-8"))
            target = declaration.get("rollback_target")
            if isinstance(target, str) and any(row["snapshot_id"] == target for row in rows):
                return target
        except (OSError, json.JSONDecodeError):
            pass
    return str(rows[-1]["snapshot_id"])


def build_diagnostic_view(
    session: Session,
    *,
    target: str | None = None,
    threshold: float | str | None = None,
) -> dict[str, Any]:
    target = target or default_target(session)
    snapshot_items: list[dict[str, Any]] = []
    for row in session.snapshot_rows():
        snapshot_id = str(row["snapshot_id"])
        bitmap = bytes(row["bitmap"])
        refs = session.snapshot_blocks(snapshot_id)
        snapshot_items.append(
            {
                "snapshot_id": snapshot_id,
                "parent_snapshot_id": row["parent_snapshot_id"],
                "created_seq": int(row["created_seq"]),
                "created_at": row["created_at"],
                "snapshot_version": int(row["snapshot_version"]),
                "dirty_blocks": popcount(bitmap),
                "block_count": int(row["block_count"]),
                "reference_count": len(refs),
                "bitmap_hex": bitmap.hex(),
                "bitmap_bits": list(set_bits(bitmap, int(row["block_count"]))),
                "bitmap_overview": bitmap_overview(bitmap, int(row["block_count"])),
                "checkpoint_sha256": f"sha256:{row['checkpoint_digest']}",
                "state": row["state"],
            }
        )
    path_result: dict[str, Any]
    if target is None:
        path_result = {"ok": False, "code": E_STATE, "message": "no snapshot is available"}
    else:
        try:
            path_result = {"ok": True, **session.plan_rollback(target, threshold=threshold).as_dict()}
        except DiagnosticError as error:
            path_result = error.to_dict()
    temp_items: list[dict[str, Any]] = []
    for operation in session.operation_rows():
        marker = session.tmp_dir / f"{operation['operation_id']}.json"
        temp_items.append(
            {
                "operation_id": operation["operation_id"],
                "kind": operation["kind"],
                "target_snapshot": operation["target_snapshot"],
                "phase": operation["phase"],
                "status": operation["status"],
                "requires_rerun": operation["requires_rerun"],
                "temp_path": operation["temp_path"],
                "temp_exists": bool(
                    operation["temp_path"] and (session.root / operation["temp_path"]).exists()
                ),
                "marker_exists": marker.exists(),
                "created_at": operation["created_at"],
                "updated_at": operation["updated_at"],
            }
        )
    temp_items.sort(key=lambda item: (item["created_at"], item["operation_id"]))
    try:
        audit = session.latest_audit(strict=False)
        audit["available"] = True
    except DiagnosticError as error:
        audit = {"available": False, "error": error.to_dict()}
    manifest: dict[str, Any] = session.manifest_consistency()
    manifest["backups"] = session.list_manifest_backups()
    events = session.logger.read(limit=20)
    return {
        "session": {
            "root": str(session.root),
            "block_size": session.block_size,
            "block_count": session.block_count,
            "state_version": session.state_version,
            "threshold": str(session.threshold),
            "manifest_format": session.meta().get("manifest_format", "m3"),
        },
        "target": target,
        "snapshots": snapshot_items,
        "rollback_path": path_result,
        "temp_area": temp_items,
        "audit": audit,
        "manifest": manifest,
        "events": events,
    }


def render_diagnostic_text(view: dict[str, Any]) -> str:
    lines: list[str] = []
    session = view["session"]
    lines.append(
        "SESSION "
        f"root={session['root']} block_size={session['block_size']} "
        f"block_count={session['block_count']} state_version={session['state_version']} "
        f"threshold={session['threshold']} manifest={session['manifest_format']}"
    )
    lines.append("")
    lines.append("SNAPSHOTS")
    if not view["snapshots"]:
        lines.append("  <none>")
    for item in view["snapshots"]:
        lines.append(
            f"  {item['created_seq']:06d} {item['snapshot_id']} "
            f"parent={item['parent_snapshot_id']} version={item['snapshot_version']} "
            f"dirty={item['dirty_blocks']}/{item['block_count']} refs={item['reference_count']} "
            f"checkpoint={item['checkpoint_sha256']}"
        )
    lines.append("")
    lines.append("BITMAP OVERVIEW")
    if not view["snapshots"]:
        lines.append("  <none>")
    for item in view["snapshots"]:
        lines.append(f"  [{item['snapshot_id']}] hex={item['bitmap_hex'] or '-'}")
        for row in item["bitmap_overview"]:
            lines.append(f"    {row}")
    lines.append("")
    lines.append("ROLLBACK PATH")
    path = view["rollback_path"]
    if path.get("ok"):
        lines.append(
            f"  target={view['target']} selected={path['selected_path']} "
            f"threshold={path['threshold']} ratio={path['ratio']} "
            f"dirty={path['dirty_blocks']}/{path['total_blocks']} "
            f"snapshots={path['snapshot_count']} chain_depth={path['chain_depth']} "
            f"boundary={path['boundary_hit']}"
        )
        lines.append(f"  reason={path['reason']}")
        lines.append(f"  tie_rule={path['tie_rule']}")
    else:
        lines.append(f"  unavailable code={path.get('code')} field={path.get('field')} message={path.get('message')}")
    lines.append("")
    lines.append("TEMP AREA")
    if not view["temp_area"]:
        lines.append("  <empty>")
    for item in view["temp_area"]:
        lines.append(
            f"  {item['operation_id']} phase={item['phase']} status={item['status']} "
            f"target={item['target_snapshot']} temp_exists={item['temp_exists']} "
            f"marker_exists={item['marker_exists']} requires_rerun={item['requires_rerun']}"
        )
    lines.append("")
    lines.append("AUDIT")
    audit = view["audit"]
    if audit.get("available"):
        lines.append(
            f"  audit_version={audit.get('audit_version')} status={audit.get('status')} "
            f"state_version={audit.get('state_version')} version_current={audit.get('version_current')} "
            f"issues={audit.get('issue_count', 0)}"
        )
        for snapshot in audit.get("snapshots", []):
            lines.append(
                f"  snapshot={snapshot['snapshot_id']} ok={snapshot['ok']} "
                f"bits={snapshot['bitmap_bits']} refs={snapshot['reference_count']} "
                f"view={snapshot.get('full_view_count', 0)}"
            )
        for issue in audit.get("issues", []):
            lines.append(
                f"  issue code={issue.get('code')} snapshot={issue.get('snapshot_id')} "
                f"field={issue.get('field')} message={issue.get('message')}"
            )
    else:
        error = audit.get("error", {})
        lines.append(f"  unavailable code={error.get('code')} message={error.get('message')}")
    lines.append("")
    lines.append("MANIFEST")
    manifest = view["manifest"]
    if manifest.get("consistent") is not None:
        lines.append(
            f"  consistent={manifest.get('consistent')} "
            f"file_format={manifest.get('file_source_format')} "
            f"file_digest={manifest.get('file_manifest_digest')} "
            f"database_digest={manifest.get('database_manifest_digest')}"
        )
    else:
        error = manifest.get("error", {})
        lines.append(f"  conflict code={error.get('code')} field={error.get('field')} message={error.get('message')}")
    for backup in manifest.get("backups", []):
        lines.append(
            f"  backup={backup['name']} readonly={backup['readonly']} sha256={backup['sha256']}"
        )
    lines.append("")
    lines.append("RECENT EVENTS")
    if not view["events"]:
        lines.append("  <none>")
    for event in view["events"]:
        lines.append(
            f"  {event.get('ts')} event={event.get('event')} code={event.get('code')} "
            f"operation={event.get('operation')} snapshot={event.get('snapshot_id')}"
        )
    return "\n".join(lines) + "\n"


def run_gui(
    session_path: str | Path,
    *,
    target: str | None = None,
    threshold: float | str | None = None,
) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        raise DiagnosticError(
            E_UI,
            "tkinter is unavailable in this Python installation",
            field="tkinter",
            details={"reason": str(exc)},
        )

    session = Session.open(session_path)
    root = tk.Tk()
    root.title("snapdiag - 快照与回滚诊断")
    root.geometry("1180x760")
    toolbar = ttk.Frame(root, padding=(8, 8, 8, 4))
    toolbar.pack(fill=tk.X)
    ttk.Label(toolbar, text=f"会话: {session.root}").pack(side=tk.LEFT)
    status = tk.StringVar(value="就绪")
    ttk.Label(toolbar, textvariable=status).pack(side=tk.LEFT, padx=16)
    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
    left = ttk.Frame(paned)
    right = ttk.Frame(paned)
    paned.add(left, weight=2)
    paned.add(right, weight=3)
    columns = ("parent", "version", "dirty", "refs")
    tree = ttk.Treeview(left, columns=columns, show="tree headings", height=24)
    tree.heading("#0", text="快照")
    tree.heading("parent", text="父快照")
    tree.heading("version", text="版本")
    tree.heading("dirty", text="变更块")
    tree.heading("refs", text="引用数")
    tree.column("#0", width=150, anchor=tk.W)
    for column in columns:
        tree.column(column, width=80, anchor=tk.CENTER)
    scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text = tk.Text(right, wrap=tk.NONE, font=("Consolas", 9))
    text_scroll_y = ttk.Scrollbar(right, orient=tk.VERTICAL, command=text.yview)
    text_scroll_x = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=text.xview)
    text.configure(yscrollcommand=text_scroll_y.set, xscrollcommand=text_scroll_x.set)
    text.grid(row=0, column=0, sticky="nsew")
    text_scroll_y.grid(row=0, column=1, sticky="ns")
    text_scroll_x.grid(row=1, column=0, sticky="ew")
    right.rowconfigure(0, weight=1)
    right.columnconfigure(0, weight=1)

    def refresh() -> None:
        try:
            view = build_diagnostic_view(session, target=target, threshold=threshold)
        except Exception as exc:
            status.set(f"刷新失败: {exc}")
            messagebox.showerror("诊断失败", str(exc), parent=root)
            return
        for item in tree.get_children():
            tree.delete(item)
        for snapshot in view["snapshots"]:
            tree.insert(
                "",
                tk.END,
                iid=snapshot["snapshot_id"],
                text=snapshot["snapshot_id"],
                values=(
                    snapshot["parent_snapshot_id"] or "-",
                    snapshot["snapshot_version"],
                    f"{snapshot['dirty_blocks']}/{snapshot['block_count']}",
                    snapshot["reference_count"],
                ),
            )
        text.configure(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        text.insert("1.0", render_diagnostic_text(view))
        text.configure(state=tk.DISABLED)
        status.set(f"已刷新 target={view['target']} state_version={session.state_version}")

    refresh_button = ttk.Button(toolbar, text="刷新", command=refresh)
    refresh_button.pack(side=tk.RIGHT)
    refresh()
    root.mainloop()

