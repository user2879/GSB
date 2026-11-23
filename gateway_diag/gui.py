"""Tkinter domain-data editor and diagnostic operations console."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .engine import DiagnosticEngine
from .errors import GatewayError
from .ledger import LedgerStore
from .model import canonical_json, load_requests, load_spec, parse_spec_obj


class GatewayApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        spec_path: str | None = None,
        requests_path: str | None = None,
        db_path: str | None = None,
    ) -> None:
        self.root = root
        self.root.title("推理网关路由诊断台")
        self.root.geometry("1380x860")
        self.root.minsize(1080, 680)
        self.spec_path = tk.StringVar(value=spec_path or "")
        self.requests_path = tk.StringVar(value=requests_path or "")
        self.db_path = tk.StringVar(value=db_path or "artifacts/gateway.sqlite3")
        self.status = tk.StringVar(value="就绪")
        self.filter_request = tk.StringVar()
        self._worker: threading.Thread | None = None
        self._queue: queue.Queue[tuple[str, bool, Any]] = queue.Queue()
        self._build_ui()
        if spec_path and Path(spec_path).exists():
            self._load_spec_text(Path(spec_path))
        if requests_path and Path(requests_path).exists():
            self._load_requests_text(Path(requests_path))
        self.root.after(100, self._poll_worker)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Spec").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.spec_path, width=32).pack(side=tk.LEFT, padx=(4, 3))
        ttk.Button(toolbar, text="打开", command=self._choose_spec).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="保存", command=self._save_spec_from_editor).pack(side=tk.LEFT, padx=(3, 12))
        ttk.Label(toolbar, text="Requests").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.requests_path, width=30).pack(side=tk.LEFT, padx=(4, 3))
        ttk.Button(toolbar, text="打开", command=self._choose_requests).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="编辑", command=self._open_requests_editor).pack(side=tk.LEFT, padx=(3, 12))
        ttk.Label(toolbar, text="Ledger").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.db_path, width=28).pack(side=tk.LEFT, padx=(4, 3))
        ttk.Button(toolbar, text="选择", command=self._choose_db).pack(side=tk.LEFT)

        actions = ttk.Frame(self.root, padding=(8, 2))
        actions.pack(fill=tk.X)
        self.action_buttons: list[ttk.Button] = []
        for label, callback in (
            ("校验", self._validate),
            ("执行/续跑", self._run),
            ("快速视图", self._refresh_preview),
            ("权威校验", self._verify_authority),
            ("恢复重放", self._recover),
            ("版本历史", self._refresh_versions),
        ):
            button = ttk.Button(actions, text=label, command=callback)
            button.pack(side=tk.LEFT, padx=(0, 5))
            self.action_buttons.append(button)
        ttk.Label(actions, text="请求过滤").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(actions, textvariable=self.filter_request, width=12).pack(side=tk.LEFT)
        ttk.Button(actions, text="定位", command=self._focus_preview).pack(side=tk.LEFT, padx=4)
        ttk.Label(self.root, textvariable=self.status, anchor=tk.W, padding=(10, 3)).pack(fill=tk.X)

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=2)
        paned.add(right, weight=3)

        editor_header = ttk.Frame(left)
        editor_header.pack(fill=tk.X)
        ttk.Label(editor_header, text="领域规格 JSON").pack(side=tk.LEFT)
        ttk.Button(editor_header, text="+ 后端", command=lambda: self._structured_add("backend")).pack(side=tk.RIGHT)
        ttk.Button(editor_header, text="+ 规则", command=lambda: self._structured_add("rule")).pack(side=tk.RIGHT, padx=4)
        editor_frame = ttk.Frame(left)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        self.spec_editor = tk.Text(editor_frame, wrap=tk.NONE, undo=True, font=("Consolas", 10))
        spec_y = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self.spec_editor.yview)
        spec_x = ttk.Scrollbar(editor_frame, orient=tk.HORIZONTAL, command=self.spec_editor.xview)
        self.spec_editor.configure(yscrollcommand=spec_y.set, xscrollcommand=spec_x.set)
        self.spec_editor.grid(row=0, column=0, sticky="nsew")
        spec_y.grid(row=0, column=1, sticky="ns")
        spec_x.grid(row=1, column=0, sticky="ew")
        editor_frame.rowconfigure(0, weight=1)
        editor_frame.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.backend_tree = self._make_tree_tab(
            "后端",
            ("id", "host", "port", "model_dim", "hidden_dim", "seed", "quantization"),
        )
        self.rule_tree = self._make_tree_tab(
            "规则",
            ("id", "prefix", "stable_backend", "canary_backend", "canary_weight_bp", "batch_size", "version"),
        )
        self.backend_tree.bind("<Double-1>", lambda _event: self._structured_edit("backend"))
        self.rule_tree.bind("<Double-1>", lambda _event: self._structured_edit("rule"))
        for tree, kind in ((self.backend_tree, "backend"), (self.rule_tree, "rule")):
            tree.bind("<Delete>", lambda _event, selected=tree, entity=kind: self._structured_delete(entity))

        preview_frame, self.preview_tree = self._make_tree_tab(
            "快速视图",
            (
                "request_id",
                "rule_id",
                "target",
                "backend_id",
                "bucket",
                "weight_version",
                "batch_id",
                "output_digest",
                "wait_ms",
                "exec_ms",
            ),
        )
        self.preview_tree.bind("<Double-1>", self._show_preview_detail)
        self.version_tree = self._make_tree_tab(
            "版本", ("revision", "format_version", "fingerprint", "activated_at_ms")
        )
        self.version_tree.bind("<<TreeviewSelect>>", lambda _event: self._show_version_diff())
        self.log = tk.Text(self.notebook, wrap=tk.WORD, font=("Consolas", 9))
        self.log.configure(state=tk.DISABLED)
        self.notebook.add(self.log, text="日志")

    def _make_tree_tab(self, title: str, columns: tuple[str, ...]) -> Any:
        frame = ttk.Frame(self.notebook)
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="extended")
        for column in columns:
            tree.heading(column, text=column)
            width = 180 if column in ("output_digest", "batch_id", "fingerprint") else 100
            tree.column(column, width=width, anchor=tk.W, stretch=True)
        scroll_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scroll_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.notebook.add(frame, text=title)
        return tree

    def _log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _choose_spec(self) -> None:
        path = filedialog.askopenfilename(
            title="选择规格 JSON", filetypes=(("JSON", "*.json"), ("All files", "*.*"))
        )
        if path:
            self.spec_path.set(path)
            self._load_spec_text(Path(path))

    def _choose_requests(self) -> None:
        path = filedialog.askopenfilename(
            title="选择请求 NDJSON", filetypes=(("NDJSON", "*.ndjson *.jsonl"), ("All files", "*.*"))
        )
        if path:
            self.requests_path.set(path)
            self._load_requests_text(Path(path))

    def _choose_db(self) -> None:
        path = filedialog.asksaveasfilename(
            title="选择 SQLite 账本",
            defaultextension=".sqlite3",
            filetypes=(("SQLite", "*.sqlite3 *.db"), ("All files", "*.*")),
        )
        if path:
            self.db_path.set(path)

    def _load_spec_text(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
            spec = parse_spec_obj(json.loads(text), source=str(path))
        except Exception as exc:
            messagebox.showerror("无法打开规格", str(exc))
            return
        self._set_editor(text)
        self._populate_structured(spec)
        self.status.set(f"已加载 revision {spec.revision}")

    def _load_requests_text(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("无法打开请求流", str(exc))
            return
        self._requests_editor_text = text
        self.status.set(f"已加载请求流 {path.name}")

    def _set_editor(self, text: str) -> None:
        self.spec_editor.delete("1.0", tk.END)
        self.spec_editor.insert("1.0", text.rstrip() + "\n")

    def _populate_structured(self, spec: Any) -> None:
        self.backend_tree.delete(*self.backend_tree.get_children())
        self.rule_tree.delete(*self.rule_tree.get_children())
        for backend in spec.backends:
            values = [backend.to_dict()[key] for key in self.backend_tree["columns"]]
            self.backend_tree.insert("", tk.END, iid=f"backend:{backend.id}", values=values)
        for rule in spec.rules:
            values = [rule.to_dict()[key] for key in self.rule_tree["columns"]]
            self.rule_tree.insert("", tk.END, iid=f"rule:{rule.id}", values=values)

    def _read_editor_spec(self) -> Any:
        try:
            value = json.loads(self.spec_editor.get("1.0", tk.END))
        except json.JSONDecodeError as exc:
            raise GatewayError(
                f"规格 JSON 无效: {exc.msg}",
                code="E_SPEC",
                entry="spec",
                field=f"line {exc.lineno} column {exc.colno}",
            ) from exc
        return parse_spec_obj(value, source=self.spec_path.get() or "<editor>")

    def _save_spec_from_editor(self) -> Any:
        spec = self._read_editor_spec()
        raw_path = self.spec_path.get().strip()
        if not raw_path:
            raise GatewayError("未选择规格文件", code="E_SPEC", entry="spec", field="file")
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._populate_structured(spec)
        self.status.set(f"规格已保存，revision {spec.revision}")
        return spec

    def _replace_editor_from_dict(self, value: dict[str, Any], *, increment_revision: bool) -> None:
        spec = parse_spec_obj(value)
        normalized = spec.to_dict()
        if increment_revision:
            normalized["revision"] = spec.revision + 1
        self._set_editor(json.dumps(normalized, ensure_ascii=False, indent=2))

    def _structured_add(self, kind: str) -> None:
        try:
            spec = self._read_editor_spec()
        except GatewayError as exc:
            messagebox.showerror("规格无效", json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
            return
        value = spec.to_dict()
        if kind == "backend":
            ports = {backend.port for backend in spec.backends}
            port = 43200
            while port in ports:
                port += 1
            template = {
                "id": f"backend-{len(value['backends']) + 1}",
                "host": "127.0.0.1",
                "port": port,
                "model_dim": 4,
                "hidden_dim": 4,
                "seed": 1,
                "quantization": "int8",
            }
            self._open_object_editor("新增后端", template, lambda obj: self._apply_structured_add(value, "backends", obj))
        else:
            backend_id = spec.backends[0].id
            template = {
                "id": f"rule-{len(value['rules']) + 1}",
                "prefix": f"rule-{len(value['rules']) + 1}:",
                "stable_backend": backend_id,
                "canary_backend": backend_id,
                "canary_weight_bp": 0,
                "batch_size": 2,
                "version": 1,
            }
            self._open_object_editor("新增规则", template, lambda obj: self._apply_structured_add(value, "rules", obj))

    def _apply_structured_add(self, value: dict[str, Any], collection: str, obj: dict[str, Any]) -> None:
        value[collection].append(obj)
        self._replace_editor_from_dict(value, increment_revision=True)
        self.status.set("已新增条目并递增 revision；请保存后执行")

    def _structured_edit(self, kind: str) -> None:
        tree = self.backend_tree if kind == "backend" else self.rule_tree
        selection = tree.selection()
        if not selection:
            return
        item = selection[0]
        entry_id = item.split(":", 1)[1]
        try:
            spec = self._read_editor_spec()
        except GatewayError as exc:
            messagebox.showerror("规格无效", json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
            return
        value = spec.to_dict()
        collection = "backends" if kind == "backend" else "rules"
        current = next(item for item in value[collection] if item["id"] == entry_id)
        self._open_object_editor(
            "编辑" + ("后端" if kind == "backend" else "规则"),
            current,
            lambda obj: self._apply_structured_edit(value, collection, entry_id, obj),
        )

    def _apply_structured_edit(
        self, value: dict[str, Any], collection: str, entry_id: str, obj: dict[str, Any]
    ) -> None:
        for index, current in enumerate(value[collection]):
            if current["id"] != entry_id:
                continue
            if collection == "rules" and canonical_json(current) != canonical_json(obj):
                obj["version"] = max(int(obj.get("version", 1)), int(current.get("version", 1)) + 1)
            value[collection][index] = obj
            break
        self._replace_editor_from_dict(value, increment_revision=True)
        self.status.set("已修改条目、递增 revision 与规则 version；请保存后执行")

    def _structured_delete(self, kind: str) -> None:
        tree = self.backend_tree if kind == "backend" else self.rule_tree
        selection = tree.selection()
        if not selection:
            return
        if not messagebox.askyesno("确认删除", f"删除选中的 {len(selection)} 个条目？"):
            return
        try:
            spec = self._read_editor_spec()
        except GatewayError as exc:
            messagebox.showerror("规格无效", json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
            return
        value = spec.to_dict()
        collection = "backends" if kind == "backend" else "rules"
        ids = {item.split(":", 1)[1] for item in selection}
        value[collection] = [item for item in value[collection] if item["id"] not in ids]
        try:
            self._replace_editor_from_dict(value, increment_revision=True)
        except GatewayError as exc:
            messagebox.showerror("删除后规格无效", json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
            return
        self._populate_structured(parse_spec_obj(value))
        self.status.set("已删除条目并递增 revision；请保存后执行")

    def _open_object_editor(
        self, title: str, value: dict[str, Any], apply: Callable[[dict[str, Any]], None]
    ) -> None:
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("620x520")
        text = tk.Text(window, wrap=tk.NONE, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text.insert("1.0", json.dumps(value, ensure_ascii=False, indent=2))

        def commit() -> None:
            try:
                parsed = json.loads(text.get("1.0", tk.END))
                if not isinstance(parsed, dict):
                    raise ValueError("must be an object")
                apply(parsed)
            except Exception as exc:
                messagebox.showerror(title, str(exc), parent=window)
                return
            window.destroy()

        bar = ttk.Frame(window)
        bar.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(bar, text="应用", command=commit).pack(side=tk.RIGHT)
        ttk.Button(bar, text="取消", command=window.destroy).pack(side=tk.RIGHT, padx=6)

    def _open_requests_editor(self) -> None:
        path = self.requests_path.get()
        if not path:
            messagebox.showwarning("请求流", "请先选择请求流文件")
            return
        window = tk.Toplevel(self.root)
        window.title("NDJSON 请求流编辑")
        window.geometry("900x620")
        text = tk.Text(window, wrap=tk.NONE, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text.insert("1.0", Path(path).read_text(encoding="utf-8"))

        def save() -> None:
            try:
                self._save_spec_from_editor()
                spec = load_spec(self.spec_path.get())
                content = text.get("1.0", tk.END)
                temporary = Path(path + ".tmp")
                temporary.write_text(content, encoding="utf-8")
                load_requests(temporary, spec)
                temporary.replace(path)
            except Exception as exc:
                messagebox.showerror("请求流无效", str(exc), parent=window)
                return
            self._requests_editor_text = text.get("1.0", tk.END)
            self.status.set("请求流已保存并通过强校验")
            window.destroy()

        bar = ttk.Frame(window)
        bar.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(bar, text="校验并保存", command=save).pack(side=tk.RIGHT)
        ttk.Button(bar, text="取消", command=window.destroy).pack(side=tk.RIGHT, padx=6)

    def _validate(self) -> None:
        try:
            self._save_spec_from_editor()
            spec = load_spec(self.spec_path.get())
            requests = load_requests(self.requests_path.get(), spec) if self.requests_path.get() else ()
            self._populate_structured(spec)
            result = {
                "revision": spec.revision,
                "spec_fingerprint": spec.fingerprint,
                "request_count": len(requests),
                "messages": "validation passed",
            }
            self._log(canonical_json(result))
            self.status.set("校验通过")
        except Exception as exc:
            self._show_error(exc)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self.action_buttons:
            button.configure(state=state)

    def _submit(self, label: str, operation: Callable[[], Any], after: Callable[[Any], None] | None = None) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("操作进行中", "请等待当前操作完成")
            return
        self._set_busy(True)
        self.status.set(label + " ...")

        def worker() -> None:
            try:
                value = operation()
                self._queue.put((label, True, (value, after)))
            except Exception as exc:
                self._queue.put((label, False, exc))

        self._worker = threading.Thread(target=worker, name="gui-operation", daemon=True)
        self._worker.start()

    def _poll_worker(self) -> None:
        try:
            label, ok, payload = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_worker)
            return
        self._set_busy(False)
        self._worker = None
        if ok:
            value, after = payload
            self._log(f"{label}: " + canonical_json(value))
            self.status.set(label + " 完成")
            if after is not None:
                after(value)
        else:
            self._show_error(payload)
        self.root.after(100, self._poll_worker)

    def _show_error(self, exc: Exception) -> None:
        if isinstance(exc, GatewayError):
            payload = exc.to_dict()
        else:
            payload = {"ok": False, "code": "E_INTERNAL", "message": str(exc)}
        self._log(canonical_json(payload))
        self.status.set(str(payload.get("code", "ERROR")))
        messagebox.showerror("诊断失败", json.dumps(payload, ensure_ascii=False, indent=2))

    def _run(self) -> None:
        try:
            self._save_spec_from_editor()
        except Exception as exc:
            self._show_error(exc)
            return

        def operation() -> dict[str, Any]:
            spec = load_spec(self.spec_path.get())
            requests = load_requests(self.requests_path.get(), spec)
            with LedgerStore(self.db_path.get()) as store:
                engine = DiagnosticEngine(spec, requests, store)
                return engine.run().to_dict()

        self._submit("执行", operation, lambda _value: self._refresh_preview())

    def _refresh_preview(self) -> None:
        def operation() -> dict[str, Any]:
            with LedgerStore(self.db_path.get()) as store:
                cache = store.load_preview_cache()
                return cache.snapshot.to_dict()

        self._submit("快速视图", operation, self._render_preview)

    def _render_preview(self, result: Any) -> None:
        self.preview_tree.delete(*self.preview_tree.get_children())
        for entry in result.get("entries", []):
            values = [entry.get(column, "") for column in self.preview_tree["columns"]]
            self.preview_tree.insert("", tk.END, iid=f"request:{entry['request_id']}", values=values)
        self.notebook.select(2)
        self.status.set(
            f"revision {result.get('revision')}，{len(result.get('entries', []))} 条请求（单次读取缓存）"
        )

    def _focus_preview(self) -> None:
        raw = self.filter_request.get().strip()
        if not raw:
            return
        try:
            request_id = int(raw)
        except ValueError:
            messagebox.showwarning("过滤", "请求序号必须是整数")
            return
        item = f"request:{request_id}"
        if self.preview_tree.exists(item):
            self.notebook.select(2)
            self.preview_tree.selection_set(item)
            self.preview_tree.focus(item)
            self.preview_tree.see(item)

    def _show_preview_detail(self, _event: Any) -> None:
        selection = self.preview_tree.selection()
        if not selection:
            return
        values = self.preview_tree.item(selection[0], "values")
        detail = {column: value for column, value in zip(self.preview_tree["columns"], values)}
        messagebox.showinfo("请求命中详情", json.dumps(detail, ensure_ascii=False, indent=2))

    def _verify_authority(self) -> None:
        def operation() -> dict[str, Any]:
            spec = load_spec(self.spec_path.get())
            requests = load_requests(self.requests_path.get(), spec)
            with LedgerStore(self.db_path.get()) as store:
                engine = DiagnosticEngine(spec, requests, store)
                authority = engine.verify_authoritative(engine.preview()).to_dict()
                replay = engine.verify_replay().to_dict()
                return {"authority": authority, "replay": replay}

        self._submit("权威与重放校验", operation)

    def _recover(self) -> None:
        def operation() -> dict[str, Any]:
            spec = load_spec(self.spec_path.get())
            requests = load_requests(self.requests_path.get(), spec)
            with LedgerStore(self.db_path.get()) as store:
                return DiagnosticEngine(spec, requests, store).recover().to_dict()

        self._submit("恢复", operation, lambda _value: self._refresh_preview())

    def _refresh_versions(self) -> None:
        def operation() -> dict[str, Any]:
            with LedgerStore(self.db_path.get()) as store:
                history = store.version_history()
                return {
                    "history": history,
                    "diff": store.activation_diff() if history else None,
                    "unknowns": store.recovery_unknown_rows(
                        int(history[-1]["revision"])
                    ) if history else [],
                }

        self._submit("版本历史", operation, self._render_versions)

    def _render_versions(self, result: Any) -> None:
        self.version_tree.delete(*self.version_tree.get_children())
        for row in result.get("history", []):
            values = [
                row.get("revision"),
                row.get("format_version"),
                row.get("fingerprint"),
                row.get("activated_at_ms"),
            ]
            self.version_tree.insert("", tk.END, iid=str(row["revision"]), values=values)
        self.notebook.select(3)
        diff = result.get("diff")
        if diff:
            self._log("当前整体替换差异: " + canonical_json(diff))
        unknowns = result.get("unknowns", [])
        if unknowns:
            self._log("恢复未知批次: " + canonical_json(unknowns))

    def _show_version_diff(self) -> None:
        if not self.version_tree.selection():
            return
        try:
            with LedgerStore(self.db_path.get()) as store:
                revision = int(self.version_tree.selection()[0])
                diff = store.activation_diff(revision)
            self._log(f"revision {revision} 整体替换差异: " + canonical_json(diff))
        except Exception as exc:
            self._show_error(exc)


def launch_gui(
    *,
    spec_path: str | None = None,
    requests_path: str | None = None,
    db_path: str | None = None,
) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise GatewayError(
            "no graphical display is available for Tkinter",
            code="E_UI",
            field="display",
            details={"message": str(exc)},
        ) from exc
    GatewayApp(root, spec_path=spec_path, requests_path=requests_path, db_path=db_path)
    root.mainloop()
