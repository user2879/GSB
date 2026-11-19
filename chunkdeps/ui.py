"""Tkinter diagnostic console for the local chunk-dependency workbench."""

from __future__ import annotations

import json
import queue
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from .client import WorkbenchClient
from .errors import BusinessError

try:  # tkinter is expected on the Windows target but optional in headless CI.
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except Exception:  # noqa: BLE001 - environment capability probe
    tk = None
    filedialog = None
    messagebox = None
    scrolledtext = None
    ttk = None


if ttk is not None:

    class WorkbenchUI:
        BG = "#101820"
        PANEL = "#17232e"
        PANEL_ALT = "#1d2c38"
        ACCENT = "#36c2a9"
        ACCENT_DARK = "#218c7a"
        WARN = "#f2b84b"
        ERROR = "#ef6b73"
        TEXT = "#e8f0f4"
        MUTED = "#91a5b2"

        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("ChunkDeps Workbench · 对象分片与元数据依赖诊断台")
            self.root.geometry("1280x840")
            self.root.minsize(1080, 700)
            self.root.configure(bg=self.BG)
            self.events: queue.Queue[tuple[str, Any, Any]] = queue.Queue()
            self.session_map: dict[str, dict[str, Any]] = {}
            self.object_map: dict[str, dict[str, Any]] = {}
            self._build_style()
            self._build_layout()
            self.root.after(120, self._drain_events)
            self.root.bind("<F5>", lambda _event: self.refresh_all())

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        def _build_style(self) -> None:
            style = ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            font = ("Microsoft YaHei UI", 10)
            style.configure(".", font=font, background=self.BG, foreground=self.TEXT)
            style.configure("TFrame", background=self.BG)
            style.configure("Panel.TFrame", background=self.PANEL)
            style.configure("TLabel", background=self.BG, foreground=self.TEXT)
            style.configure("Muted.TLabel", background=self.BG, foreground=self.MUTED)
            style.configure("Panel.TLabel", background=self.PANEL, foreground=self.TEXT)
            style.configure("PanelMuted.TLabel", background=self.PANEL, foreground=self.MUTED)
            style.configure(
                "Title.TLabel",
                background=self.BG,
                foreground=self.TEXT,
                font=("Microsoft YaHei UI", 19, "bold"),
            )
            style.configure(
                "Subtitle.TLabel",
                background=self.BG,
                foreground=self.MUTED,
                font=("Microsoft YaHei UI", 10),
            )
            style.configure(
                "CardValue.TLabel",
                background=self.PANEL_ALT,
                foreground=self.ACCENT,
                font=("Consolas", 16, "bold"),
            )
            style.configure(
                "CardLabel.TLabel",
                background=self.PANEL_ALT,
                foreground=self.MUTED,
                font=("Microsoft YaHei UI", 9),
            )
            style.configure(
                "Accent.TButton",
                background=self.ACCENT,
                foreground="#07110f",
                borderwidth=0,
                padding=(14, 9),
                font=("Microsoft YaHei UI", 10, "bold"),
            )
            style.map(
                "Accent.TButton",
                background=[("active", "#54d9c2"), ("disabled", "#49625f")],
            )
            style.configure(
                "Secondary.TButton",
                background=self.PANEL_ALT,
                foreground=self.TEXT,
                borderwidth=0,
                padding=(10, 7),
            )
            style.map("Secondary.TButton", background=[("active", "#2c4050")])
            style.configure(
                "Treeview",
                background=self.PANEL,
                fieldbackground=self.PANEL,
                foreground=self.TEXT,
                rowheight=27,
                borderwidth=0,
            )
            style.configure(
                "Treeview.Heading",
                background=self.PANEL_ALT,
                foreground=self.MUTED,
                borderwidth=0,
                font=("Microsoft YaHei UI", 9, "bold"),
            )
            style.map("Treeview", background=[("selected", self.ACCENT_DARK)])
            style.configure(
                "TNotebook", background=self.BG, borderwidth=0, tabmargins=(0, 0, 0, 0)
            )
            style.configure(
                "TNotebook.Tab",
                background=self.PANEL,
                foreground=self.MUTED,
                padding=(18, 10),
                borderwidth=0,
            )
            style.map(
                "TNotebook.Tab",
                background=[("selected", self.PANEL_ALT)],
                foreground=[("selected", self.TEXT)],
            )
            style.configure("TEntry", fieldbackground="#0d171e", foreground=self.TEXT, padding=7)
            style.configure("TCombobox", fieldbackground="#0d171e", foreground=self.TEXT, padding=5)
            style.configure("TLabelframe", background=self.PANEL, borderwidth=0)
            style.configure(
                "TLabelframe.Label", background=self.PANEL, foreground=self.MUTED
            )

        def _build_layout(self) -> None:
            outer = ttk.Frame(self.root, padding=(20, 16, 20, 12))
            outer.pack(fill="both", expand=True)

            header = ttk.Frame(outer)
            header.pack(fill="x", pady=(0, 14))
            title_box = ttk.Frame(header)
            title_box.pack(side="left", fill="x", expand=True)
            ttk.Label(title_box, text="ChunkDeps Workbench", style="Title.TLabel").pack(anchor="w")
            ttk.Label(
                title_box,
                text="确定性分片 · SQLite 元数据 · 增量依赖失效 · 审计与快照",
                style="Subtitle.TLabel",
            ).pack(anchor="w", pady=(2, 0))

            connection = ttk.Frame(header, style="Panel.TFrame", padding=10)
            connection.pack(side="right")
            ttk.Label(connection, text="HOST", style="PanelMuted.TLabel").grid(row=0, column=0, sticky="w")
            self.host_var = tk.StringVar(value="127.0.0.1")
            ttk.Entry(connection, textvariable=self.host_var, width=13).grid(row=1, column=0, padx=(0, 6))
            ttk.Label(connection, text="PORT", style="PanelMuted.TLabel").grid(row=0, column=1, sticky="w")
            self.port_var = tk.StringVar(value="8765")
            ttk.Entry(connection, textvariable=self.port_var, width=7).grid(row=1, column=1, padx=(0, 10))
            ttk.Button(connection, text="连接", style="Accent.TButton", command=self.connect).grid(row=1, column=2)
            self.conn_var = tk.StringVar(value="未连接")
            self.conn_label = tk.Label(
                connection,
                textvariable=self.conn_var,
                bg=self.PANEL,
                fg=self.WARN,
                font=("Microsoft YaHei UI", 9),
            )
            self.conn_label.grid(row=2, column=0, columnspan=3, sticky="w", pady=(7, 0))

            cards = ttk.Frame(outer)
            cards.pack(fill="x", pady=(0, 14))
            self.card_vars = {
                "sessions": tk.StringVar(value="—"),
                "objects": tk.StringVar(value="—"),
                "versions": tk.StringVar(value="—"),
                "graph": tk.StringVar(value="—"),
                "chunks": tk.StringVar(value="—"),
            }
            specs = [
                ("sessions", "上传会话"),
                ("objects", "可见对象"),
                ("versions", "元数据版本"),
                ("chunks", "登记分片"),
                ("graph", "依赖图版本"),
            ]
            for index, (key, label) in enumerate(specs):
                card = ttk.Frame(cards, style="Panel.TFrame", padding=(14, 9))
                card.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 10, 0))
                cards.columnconfigure(index, weight=1)
                ttk.Label(card, textvariable=self.card_vars[key], style="CardValue.TLabel").pack(anchor="w")
                ttk.Label(card, text=label, style="CardLabel.TLabel").pack(anchor="w")

            body = ttk.Panedwindow(outer, orient="horizontal")
            body.pack(fill="both", expand=True)
            left = ttk.Frame(body, style="Panel.TFrame", padding=14, width=355)
            right = ttk.Frame(body)
            body.add(left, weight=0)
            body.add(right, weight=1)

            self._build_upload_panel(left)
            self._build_tabs(right)

            footer = ttk.Frame(outer)
            footer.pack(fill="x", pady=(10, 0))
            self.status_var = tk.StringVar(value="Ready")
            ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").pack(side="left")
            ttk.Label(
                footer,
                text="F5 刷新 · 所有网络请求仅访问本机回环地址",
                style="Muted.TLabel",
            ).pack(side="right")

        def _build_upload_panel(self, parent: ttk.Frame) -> None:
            ttk.Label(parent, text="上传会话", style="Panel.TLabel", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
            ttk.Label(
                parent,
                text="按固定整数分片切分样本并逐片写入本地服务",
                style="PanelMuted.TLabel",
            ).pack(anchor="w", pady=(2, 12))

            self.source_var = tk.StringVar()
            self.key_var = tk.StringVar(value="demo/windows-sample.bin")
            self.chunk_size_var = tk.StringVar(value="1024")
            self.tags_var = tk.StringVar(value='{"team":"storage","case":"demo"}')
            self.concurrency_var = tk.StringVar(value="4")
            self.expected_version_var = tk.StringVar()
            self.commit_after_var = tk.BooleanVar(value=True)

            self._field(parent, "样本文件", self.source_var)
            file_row = ttk.Frame(parent, style="Panel.TFrame")
            file_row.pack(fill="x", pady=(0, 9))
            ttk.Entry(file_row, textvariable=self.source_var).pack(side="left", fill="x", expand=True)
            ttk.Button(file_row, text="浏览", style="Secondary.TButton", command=self.browse_file).pack(side="left", padx=(6, 0))
            self._field(parent, "对象键", self.key_var)
            size_row = ttk.Frame(parent, style="Panel.TFrame")
            size_row.pack(fill="x", pady=(0, 9))
            size_row.columnconfigure(0, weight=1)
            size_row.columnconfigure(1, weight=1)
            left = ttk.Frame(size_row, style="Panel.TFrame")
            left.grid(row=0, column=0, sticky="ew", padx=(0, 5))
            right = ttk.Frame(size_row, style="Panel.TFrame")
            right.grid(row=0, column=1, sticky="ew", padx=(5, 0))
            self._field(left, "分片大小 (bytes)", self.chunk_size_var)
            self._field(right, "并发上限", self.concurrency_var)
            self._field(parent, "标签 JSON", self.tags_var)
            self._field(parent, "预期基础版本 (可选)", self.expected_version_var)

            ttk.Checkbutton(
                parent,
                text="上传完成后立即提交",
                variable=self.commit_after_var,
            ).pack(anchor="w", pady=(4, 10))
            ttk.Button(
                parent,
                text="切分、上传并提交",
                style="Accent.TButton",
                command=self.upload_file,
            ).pack(fill="x", pady=(0, 8))

            ttk.Separator(parent).pack(fill="x", pady=10)
            ttk.Label(parent, text="会话操作", style="Panel.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
            ttk.Button(parent, text="断点续传所选会话", style="Secondary.TButton", command=self.resume_selected).pack(fill="x", pady=3)
            ttk.Button(parent, text="中止所选会话（保留分片）", style="Secondary.TButton", command=self.abort_selected).pack(fill="x", pady=3)
            ttk.Button(parent, text="校验并提交所选会话", style="Secondary.TButton", command=self.commit_selected).pack(fill="x", pady=3)

            ttk.Separator(parent).pack(fill="x", pady=10)
            ttk.Label(parent, text="诊断操作", style="Panel.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
            row = ttk.Frame(parent, style="Panel.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Button(row, text="审计", style="Secondary.TButton", command=self.audit).pack(side="left", fill="x", expand=True, padx=(0, 4))
            ttk.Button(row, text="依赖图", style="Secondary.TButton", command=self.show_graph).pack(side="left", fill="x", expand=True, padx=(4, 0))
            row2 = ttk.Frame(parent, style="Panel.TFrame")
            row2.pack(fill="x", pady=2)
            ttk.Button(row2, text="保存快照", style="Secondary.TButton", command=self.snapshot_save).pack(side="left", fill="x", expand=True, padx=(0, 4))
            ttk.Button(row2, text="整体恢复", style="Secondary.TButton", command=self.snapshot_restore).pack(side="left", fill="x", expand=True, padx=(4, 0))
            ttk.Button(parent, text="恢复分类 / 强制终止检查", style="Secondary.TButton", command=self.recover).pack(fill="x", pady=(5, 3))

            self.progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
            self.progress.pack(fill="x", pady=(14, 0))

        def _field(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
            ttk.Label(parent, text=label, style="PanelMuted.TLabel").pack(anchor="w", pady=(4, 2))
            ttk.Entry(parent, textvariable=variable).pack(fill="x")

        def _build_tabs(self, parent: ttk.Frame) -> None:
            notebook = ttk.Notebook(parent)
            notebook.pack(fill="both", expand=True)

            sessions = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
            notebook.add(sessions, text="会话")
            controls = ttk.Frame(sessions, style="Panel.TFrame")
            controls.pack(fill="x", pady=(0, 8))
            ttk.Button(controls, text="刷新全部", style="Secondary.TButton", command=self.refresh_all).pack(side="left")
            ttk.Button(controls, text="刷新会话", style="Secondary.TButton", command=self.refresh_sessions).pack(side="left", padx=6)
            self.session_tree = self._tree(
                sessions,
                ("session", "key", "state", "bitmap", "registered", "version"),
                {"session": 170, "key": 250, "state": 90, "bitmap": 240, "registered": 80, "version": 60},
            )
            self.session_tree.bind("<<TreeviewSelect>>", lambda _event: self._select_session())

            objects = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
            notebook.add(objects, text="对象版本")
            object_controls = ttk.Frame(objects, style="Panel.TFrame")
            object_controls.pack(fill="x", pady=(0, 8))
            ttk.Button(object_controls, text="刷新对象", style="Secondary.TButton", command=self.refresh_objects).pack(side="left")
            ttk.Button(object_controls, text="删除所选对象", style="Secondary.TButton", command=self.delete_selected).pack(side="left", padx=6)
            self.object_tree = self._tree(
                objects,
                ("key", "version", "status", "metadata", "content", "chunks"),
                {"key": 270, "version": 65, "status": 80, "metadata": 300, "content": 300, "chunks": 70},
            )

            views = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
            notebook.add(views, text="前缀视图")
            ttk.Button(views, text="刷新视图", style="Secondary.TButton", command=self.refresh_views).pack(anchor="w", pady=(0, 8))
            self.view_tree = self._tree(
                views,
                ("prefix", "count", "digest", "version"),
                {"prefix": 220, "count": 90, "digest": 470, "version": 90},
            )

            deps = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
            notebook.add(deps, text="依赖失效")
            ttk.Button(deps, text="加载最近失效", style="Secondary.TButton", command=self.refresh_invalidations).pack(anchor="w", pady=(0, 8))
            self.dep_text = self._text(deps)

            output = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
            notebook.add(output, text="JSON 输出")
            self.output_text = self._text(output)

        def _tree(
            self,
            parent: ttk.Frame,
            columns: tuple[str, ...],
            widths: dict[str, int],
        ) -> ttk.Treeview:
            frame = ttk.Frame(parent, style="Panel.TFrame")
            frame.pack(fill="both", expand=True)
            tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
            for column in columns:
                tree.heading(column, text=column.upper())
                tree.column(column, width=widths[column], anchor="w", stretch=True)
            y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            x_scroll.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            return tree

        def _text(self, parent: ttk.Frame) -> tk.Text:
            frame = ttk.Frame(parent, style="Panel.TFrame")
            frame.pack(fill="both", expand=True)
            widget = tk.Text(
                frame,
                bg="#0b1319",
                fg=self.TEXT,
                insertbackground=self.ACCENT,
                selectbackground=self.ACCENT_DARK,
                font=("Cascadia Mono", 9),
                borderwidth=0,
                padx=12,
                pady=10,
                wrap="none",
            )
            scroll_y = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
            scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=widget.xview)
            widget.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
            widget.grid(row=0, column=0, sticky="nsew")
            scroll_y.grid(row=0, column=1, sticky="ns")
            scroll_x.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            return widget

        # ------------------------------------------------------------------
        # Async IPC
        # ------------------------------------------------------------------
        def _client(self) -> WorkbenchClient:
            try:
                port = int(self.port_var.get())
            except ValueError as exc:
                raise BusinessError("E_INPUT", "端口必须是整数") from exc
            return WorkbenchClient(self.host_var.get().strip(), port, timeout=120.0)

        def _run(
            self,
            label: str,
            work: Callable[[], dict[str, Any]],
            on_success: Callable[[dict[str, Any]], None] | None = None,
        ) -> None:
            self.status_var.set(label)
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)

            def worker() -> None:
                try:
                    result = work()
                except Exception as exc:  # noqa: BLE001 - surface to UI
                    self.events.put(("error", exc, None))
                else:
                    self.events.put(("success", result, on_success))

            threading.Thread(target=worker, daemon=True).start()

        def _drain_events(self) -> None:
            try:
                while True:
                    kind, payload, callback = self.events.get_nowait()
                    if kind == "progress":
                        self.status_var.set(str(payload))
                        continue
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    if kind == "success":
                        self.status_var.set("完成")
                        self._show_json(payload)
                        if callback is not None:
                            callback(payload)
                    else:
                        exc = payload
                        self.status_var.set("操作失败")
                        if isinstance(exc, BusinessError):
                            message = f"[{exc.code}] {exc.message}\n\n{json.dumps(exc.details, ensure_ascii=False, indent=2)}"
                        else:
                            message = "".join(traceback.format_exception(exc))
                        self._show_json({"ok": False, "error": str(exc)})
                        messagebox.showerror("操作失败", message, parent=self.root)
            except queue.Empty:
                pass
            self.root.after(120, self._drain_events)

        def _show_json(self, value: Any) -> None:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", text)

        # ------------------------------------------------------------------
        # Actions
        # ------------------------------------------------------------------
        def browse_file(self) -> None:
            selected = filedialog.askopenfilename(parent=self.root, title="选择确定性样本文件")
            if selected:
                self.source_var.set(selected)

        def connect(self) -> None:
            def work() -> dict[str, Any]:
                return self._client().ping()

            def connected(result: dict[str, Any]) -> None:
                self.conn_var.set(f"已连接 :{result.get('port')} · protocol {result.get('protocol')}")
                self.conn_label.configure(fg=self.ACCENT)
                self.refresh_all()

            self._run("正在连接本地服务…", work, connected)

        def upload_file(self) -> None:
            source = Path(self.source_var.get().strip())
            if not source.is_file():
                messagebox.showwarning("缺少样本", "请选择存在的本地样本文件。", parent=self.root)
                return
            try:
                chunk_size = int(self.chunk_size_var.get())
                concurrency = int(self.concurrency_var.get())
                tags = json.loads(self.tags_var.get() or "{}")
                expected = self.expected_version_var.get().strip()
                expected_version = int(expected) if expected else None
            except (ValueError, json.JSONDecodeError) as exc:
                messagebox.showwarning("输入无效", str(exc), parent=self.root)
                return
            commit = self.commit_after_var.get()
            client = self._client()
            key = self.key_var.get().strip()

            def progress(item: dict[str, Any]) -> None:
                self.events.put(("progress", f"已上传分片 {item.get('chunk_index')}", None))

            def work() -> dict[str, Any]:
                return client.upload_file(
                    object_key=key,
                    source=source,
                    chunk_size=chunk_size,
                    tags=tags,
                    concurrency=concurrency,
                    expected_base_version=expected_version,
                    commit=commit,
                    progress=progress,
                )

            def done(result: dict[str, Any]) -> None:
                session = result.get("created_session", {})
                if session.get("session_id"):
                    self.session_map[session["session_id"]] = session
                self.refresh_all()

            self._run("正在切分并上传…", work, done)

        def _selected_session_id(self, *, quiet: bool = False) -> str | None:
            selection = self.session_tree.selection()
            if not selection:
                if not quiet:
                    messagebox.showinfo("未选择", "请先在会话表中选择一个会话。", parent=self.root)
                return None
            selection_id = selection[0]
            session = self.session_map.get(selection_id)
            if session is None:
                return None
            return session["session_id"]

        def _select_session(self) -> None:
            session_id = self._selected_session_id(quiet=True)
            if session_id:
                self.status_var.set(f"已选择会话 {session_id}")

        def resume_selected(self) -> None:
            session_id = self._selected_session_id()
            if not session_id:
                return
            client = self._client()
            self._run(
                "正在按位图恢复会话…",
                lambda: client.request("resume_session", {"session_id": session_id}),
                lambda _result: self.refresh_all(),
            )

        def abort_selected(self) -> None:
            session_id = self._selected_session_id()
            if not session_id:
                return
            if not messagebox.askyesno("确认中止", "中止会话但保留已上传分片？", parent=self.root):
                return
            client = self._client()
            self._run(
                "正在中止会话…",
                lambda: client.request(
                    "abort_session", {"session_id": session_id, "reason": "Tkinter operator abort"}
                ),
                lambda _result: self.refresh_all(),
            )

        def commit_selected(self) -> None:
            session_id = self._selected_session_id()
            if not session_id:
                return
            client = self._client()
            self._run(
                "正在逐片校验并原子提交…",
                lambda: client.request("commit_session", {"session_id": session_id}),
                lambda _result: self.refresh_all(),
            )

        def delete_selected(self) -> None:
            selection = self.object_tree.selection()
            if not selection:
                messagebox.showinfo("未选择", "请先选择一个可见对象。", parent=self.root)
                return
            obj = self.object_map.get(selection[0])
            if not obj:
                return
            if obj.get("status") != "active":
                messagebox.showinfo(
                    "只读版本",
                    "只有活动版本可以删除；过期版本保留为只读视图。",
                    parent=self.root,
                )
                return
            if not messagebox.askyesno(
                "确认删除", f"删除对象 {obj['object_key']} 的活动版本？", parent=self.root
            ):
                return
            client = self._client()
            self._run(
                "正在删除并传播失效…",
                lambda: client.request("delete_object", {"object_key": obj["object_key"]}),
                lambda _result: self.refresh_all(),
            )

        def refresh_all(self) -> None:
            def work() -> dict[str, Any]:
                client = self._client()
                return {
                    "status": client.request("status"),
                    "sessions": client.request("list_sessions"),
                    "objects": client.request("list_objects"),
                    "views": client.request("list_views"),
                    "invalidations": client.request("invalidations", {"limit": 12}),
                }

            def apply(result: dict[str, Any]) -> None:
                status = result["status"]
                counts = status["counts"]
                self.card_vars["sessions"].set(str(counts["sessions"]))
                self.card_vars["objects"].set(str(counts["active_objects"]))
                self.card_vars["versions"].set(str(counts["object_versions"]))
                self.card_vars["chunks"].set(str(counts["chunks"]))
                self.card_vars["graph"].set(str(status["graph_version"]))
                self._populate_sessions(result["sessions"]["sessions"])
                self._populate_objects(result["objects"])
                self._populate_views(result["views"]["views"])
                self._populate_invalidations(result["invalidations"]["events"])

            self._run("正在刷新工作台…", work, apply)

        def refresh_sessions(self) -> None:
            client = self._client()
            self._run(
                "正在读取会话…",
                lambda: client.request("list_sessions"),
                lambda result: self._populate_sessions(result["sessions"]),
            )

        def refresh_objects(self) -> None:
            client = self._client()
            self._run("正在读取对象版本…", lambda: client.request("list_objects"), self._populate_objects)

        def refresh_views(self) -> None:
            client = self._client()
            self._run(
                "正在刷新前缀视图…",
                lambda: client.request("list_views"),
                lambda result: self._populate_views(result["views"]),
            )

        def refresh_invalidations(self) -> None:
            client = self._client()
            self._run(
                "正在读取依赖失效事件…",
                lambda: client.request("invalidations", {"limit": 30}),
                lambda result: self._populate_invalidations(result["events"]),
            )

        def audit(self) -> None:
            self._run("正在审计分片、对象与元数据…", lambda: self._client().request("audit"), lambda _r: self.refresh_all())

        def show_graph(self) -> None:
            self._run("正在校验依赖图…", lambda: self._client().request("get_graph"))

        def snapshot_save(self) -> None:
            self._run("正在保存整体快照…", lambda: self._client().request("snapshot_save"))

        def snapshot_restore(self) -> None:
            if not messagebox.askyesno(
                "整体恢复",
                "恢复会以快照整体替换当前会话、元数据、依赖图和审计结论，不做增量合并。继续？",
                parent=self.root,
            ):
                return
            self._run(
                "正在比较并整体恢复快照…",
                lambda: self._client().request("snapshot_restore"),
                lambda _result: self.refresh_all(),
            )

        def recover(self) -> None:
            self._run("正在读取恢复分类…", lambda: self._client().request("recovery_status"))

        # ------------------------------------------------------------------
        # Table population
        # ------------------------------------------------------------------
        def _clear(self, tree: ttk.Treeview) -> None:
            for item in tree.get_children():
                tree.delete(item)

        def _populate_sessions(self, sessions: list[dict[str, Any]]) -> None:
            self.session_map = {}
            self._clear(self.session_tree)
            for session in sessions:
                iid = self.session_tree.insert(
                    "",
                    "end",
                    values=(
                        session["session_id"],
                        session["object_key"],
                        session["state"],
                        session["bitmap"],
                        session["registered_count"],
                        session["version"] if session["version"] is not None else "—",
                    ),
                )
                self.session_map[iid] = session

        def _populate_objects(self, result: dict[str, Any]) -> None:
            self.object_map = {}
            self._clear(self.object_tree)
            for obj in result.get("versions", []):
                iid = self.object_tree.insert(
                    "",
                    "end",
                    values=(
                        obj["object_key"],
                        obj["version"],
                        obj["status"],
                        obj["metadata_digest"],
                        obj["content_digest"],
                        obj["chunk_count"],
                    ),
                )
                self.object_map[iid] = obj

        def _populate_views(self, views: list[dict[str, Any]]) -> None:
            self._clear(self.view_tree)
            for view in views:
                self.view_tree.insert(
                    "",
                    "end",
                    values=(
                        view["prefix"],
                        view["count"],
                        view["digest"],
                        view["entry_version"],
                    ),
                )

        def _populate_invalidations(self, events: list[dict[str, Any]]) -> None:
            chunks: list[str] = []
            for event in reversed(events):
                chunks.append(
                    f"● {event['trigger_type']}  {event['event_id']}\n"
                    f"  reason: {event['reason']}\n"
                    f"  graph_version: {event['graph_version']}  affected: {len(event['affected'])}"
                )
                for item in event["affected"]:
                    marker = "DIRECT" if item["direct"] else f"DEPTH {item['depth']}"
                    chunks.append(
                        f"    [{marker}] {item['kind']} {item['ref_id']}  "
                        f"v{item['entry_version_before']}→v{item['entry_version_after']}  "
                        f"{item['digest_before'] or '∅'}→∅"
                    )
                chunks.append("")
            self.dep_text.delete("1.0", "end")
            self.dep_text.insert("1.0", "\n".join(chunks) if chunks else "暂无失效事件")


else:
    WorkbenchUI = None


def main() -> int:
    if tk is None or WorkbenchUI is None:
        print(
            "tkinter is not available in this Python installation. "
            "Install the standard Windows Python build or use the CLI.",
            flush=True,
        )
        return 2
    root = tk.Tk()
    WorkbenchUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
