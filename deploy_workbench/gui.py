from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .canonical import atomic_write_json, canonical_json
from .errors import E_ENV, WorkbenchError, log_error
from .service import WorkbenchService


SOURCE_TYPES = ("baseline", "file_override", "runtime_injection")


def launch_gui(
    *,
    spec_path: str | Path,
    state_root: str | Path,
    scale_power_override: int | str | None,
    log_path: str | Path,
) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise WorkbenchError(
            E_ENV,
            "tkinter is unavailable in this Python installation; use the Windows official Python build or run a headless CLI command",
            path="$",
        ) from exc

    service = WorkbenchService(spec_path, state_root, scale_power_override=scale_power_override)
    root = tk.Tk()
    app = WorkbenchApp(
        root=root,
        tk=tk,
        ttk=ttk,
        messagebox=messagebox,
        service=service,
        spec_path=Path(spec_path),
        state_root=Path(state_root),
        scale_power_override=scale_power_override,
        log_path=Path(log_path),
    )
    try:
        root.mainloop()
    finally:
        app.close()
    return 0


class FormDialog:
    def __init__(self, tk: Any, parent: Any, title: str, fields: Iterable[tuple[str, str, str]]) -> None:
        self.result: dict[str, str] | None = None
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()
        self.entries: dict[str, Any] = {}
        for row, (name, label, default) in enumerate(fields):
            tk.Label(self.top, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=10, pady=5)
            entry = tk.Entry(self.top, width=55)
            entry.insert(0, default)
            entry.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
            self.entries[name] = entry
        self.top.columnconfigure(1, weight=1)
        buttons = tk.Frame(self.top)
        buttons.grid(row=len(self.entries), column=0, columnspan=2, pady=10)
        tk.Button(buttons, text="OK", width=12, command=self._accept).pack(side="left", padx=4)
        tk.Button(buttons, text="Cancel", width=12, command=self._cancel).pack(side="left", padx=4)
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)
        self.top.wait_window()

    def _accept(self) -> None:
        self.result = {name: entry.get() for name, entry in self.entries.items()}
        self.top.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.top.destroy()


class WorkbenchApp:
    def __init__(
        self,
        *,
        root: Any,
        tk: Any,
        ttk: Any,
        messagebox: Any,
        service: WorkbenchService,
        spec_path: Path,
        state_root: Path,
        scale_power_override: int | str | None,
        log_path: Path,
    ) -> None:
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.service = service
        self.spec_path = spec_path
        self.state_root = state_root
        self.scale_power_override = scale_power_override
        self.log_path = log_path
        self.raw: dict[str, Any] = json.loads(self.spec_path.read_text(encoding="utf-8"))
        self.selected_environment: str | None = None
        self.selected_source: str | None = None
        self._configure_window()
        self._build_toolbar()
        self._build_body()
        self._refresh_editors()
        self.refresh_view()

    def _configure_window(self) -> None:
        self.root.title("Deployment Consistency Workbench")
        self.root.geometry("1500x900")
        self.root.minsize(1150, 700)
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Treeview", rowheight=24)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_toolbar(self) -> None:
        frame = self.ttk.Frame(self.root, padding=6)
        frame.pack(side="top", fill="x")
        for text, command in (
            ("Reload file", self.reload_file),
            ("Save edits", self.save_edits),
            ("Recalculate", self.recalculate),
            ("Refresh view (read-only)", self.refresh_view),
            ("Replay latest snapshot", self.replay_snapshot),
        ):
            self.ttk.Button(frame, text=text, command=command).pack(side="left", padx=3)
        self.status_var = self.tk.StringVar(value="Ready")
        self.ttk.Label(frame, textvariable=self.status_var, anchor="e").pack(side="right", padx=8)

    def _build_body(self) -> None:
        paned = self.ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = self.ttk.Frame(paned, padding=4)
        right = self.ttk.Frame(paned, padding=4)
        paned.add(left, weight=2)
        paned.add(right, weight=5)
        self._build_editor_tabs(left)
        self._build_view_tabs(right)

    def _tree(self, parent: Any, columns: tuple[str, ...], widths: dict[str, int]) -> tuple[Any, Any]:
        frame = self.ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = self.ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for column in columns:
            tree.heading(column, text=column.replace("_", " ").title())
            tree.column(column, width=widths.get(column, 110), stretch=True, anchor="w")
        vertical = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = self.ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return frame, tree

    def _build_editor_tabs(self, parent: Any) -> None:
        self.editor_notebook = self.ttk.Notebook(parent)
        self.editor_notebook.pack(fill="both", expand=True)

        env_tab = self.ttk.Frame(self.editor_notebook)
        self.editor_notebook.add(env_tab, text="Environments")
        _, self.env_tree = self._tree(
            env_tab,
            ("id", "depth", "available_sources", "source_priority"),
            {"id": 105, "depth": 55, "available_sources": 220, "source_priority": 220},
        )
        env_buttons = self.ttk.Frame(env_tab)
        env_buttons.pack(side="bottom", fill="x", pady=3)
        self.ttk.Button(env_buttons, text="Add", command=self.add_environment).pack(side="left", padx=2)
        self.ttk.Button(env_buttons, text="Edit", command=self.edit_environment).pack(side="left", padx=2)
        self.ttk.Button(env_buttons, text="Delete", command=self.delete_environment).pack(side="left", padx=2)
        self.env_tree.bind("<<TreeviewSelect>>", self._environment_selected)

        source_tab = self.ttk.Frame(self.editor_notebook)
        self.editor_notebook.add(source_tab, text="Sources")
        _, self.source_tree = self._tree(source_tab, ("id", "type"), {"id": 210, "type": 190})
        source_buttons = self.ttk.Frame(source_tab)
        source_buttons.pack(side="bottom", fill="x", pady=3)
        self.ttk.Button(source_buttons, text="Add", command=self.add_source).pack(side="left", padx=2)
        self.ttk.Button(source_buttons, text="Edit type", command=self.edit_source).pack(side="left", padx=2)
        self.ttk.Button(source_buttons, text="Delete", command=self.delete_source).pack(side="left", padx=2)
        self.source_tree.bind("<<TreeviewSelect>>", self._source_selected)

        entries_tab = self.ttk.Frame(self.editor_notebook)
        self.editor_notebook.add(entries_tab, text="Source entries")
        selector = self.ttk.Frame(entries_tab)
        selector.pack(fill="x", pady=3)
        self.ttk.Label(selector, text="Environment").pack(side="left")
        self.entry_env_var = self.tk.StringVar(value="")
        self.entry_env_combo = self.ttk.Combobox(selector, textvariable=self.entry_env_var, state="readonly", width=18)
        self.entry_env_combo.pack(side="left", padx=4)
        self.ttk.Label(selector, text="Source").pack(side="left", padx=(10, 0))
        self.entry_source_var = self.tk.StringVar(value="")
        self.entry_source_combo = self.ttk.Combobox(selector, textvariable=self.entry_source_var, state="readonly", width=22)
        self.entry_source_combo.pack(side="left", padx=4)
        self.entry_env_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_entries())
        self.entry_source_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_entries())
        _, self.entry_tree = self._tree(entries_tab, ("key", "value"), {"key": 220, "value": 160})
        entry_buttons = self.ttk.Frame(entries_tab)
        entry_buttons.pack(side="bottom", fill="x", pady=3)
        self.ttk.Button(entry_buttons, text="Add", command=self.add_entry).pack(side="left", padx=2)
        self.ttk.Button(entry_buttons, text="Edit", command=self.edit_entry).pack(side="left", padx=2)
        self.ttk.Button(entry_buttons, text="Delete", command=self.delete_entry).pack(side="left", padx=2)

    def _build_view_tabs(self, parent: Any) -> None:
        notebook = self.ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        self.view_notebook = notebook

        summary_tab = self.ttk.Frame(notebook)
        notebook.add(summary_tab, text="Summary")
        _, self.summary_tree = self._tree(
            summary_tab,
            ("environment", "depth", "resolved_keys", "unresolved_keys", "chain_digest", "conclusion_digest"),
            {"environment": 110, "depth": 60, "resolved_keys": 100, "unresolved_keys": 100, "chain_digest": 320, "conclusion_digest": 320},
        )

        chains_tab = self.ttk.Frame(notebook)
        notebook.add(chains_tab, text="Source chains")
        _, self.chain_tree = self._tree(
            chains_tab,
            ("environment", "key", "rank", "source", "source_type", "value", "fixed", "binding", "selected"),
            {"environment": 85, "key": 110, "rank": 45, "source": 130, "source_type": 130, "value": 90, "fixed": 80, "binding": 70, "selected": 70},
        )

        values_tab = self.ttk.Frame(notebook)
        notebook.add(values_tab, text="Effective values / heat")
        _, self.value_tree = self._tree(
            values_tab,
            ("environment", "key", "source", "raw", "fixed", "quant_error", "error_bound", "heat"),
            {"environment": 90, "key": 110, "source": 140, "raw": 100, "fixed": 100, "quant_error": 100, "error_bound": 100, "heat": 80},
        )
        self.value_tree.tag_configure("heat_low", background="#dff5e1")
        self.value_tree.tag_configure("heat_medium", background="#fff1c7")
        self.value_tree.tag_configure("heat_high", background="#ffd7d7")

        attribution_tab = self.ttk.Frame(notebook)
        notebook.add(attribution_tab, text="Attribution")
        _, self.attribution_tree = self._tree(
            attribution_tab,
            ("environment", "task", "fixed_result", "error_bound", "dominant", "segment", "contribution", "ratio"),
            {"environment": 90, "task": 130, "fixed_result": 100, "error_bound": 110, "dominant": 180, "segment": 180, "contribution": 130, "ratio": 100},
        )

        versions_tab = self.ttk.Frame(notebook)
        notebook.add(versions_tab, text="Versions / ledger")
        _, self.version_tree = self._tree(
            versions_tab,
            ("run", "status", "environment_v", "priority_v", "scale_v", "current", "stale", "content_digest"),
            {"run": 60, "status": 110, "environment_v": 100, "priority_v": 90, "scale_v": 80, "current": 75, "stale": 65, "content_digest": 340},
        )
        version_buttons = self.ttk.Frame(versions_tab)
        version_buttons.pack(side="bottom", fill="x", pady=3)
        self.ttk.Button(
            version_buttons,
            text="Open selected version (read-only)",
            command=self.open_historical,
        ).pack(side="left", padx=2)

    def _handle_error(self, error: Exception) -> None:
        if isinstance(error, WorkbenchError):
            log_error(self.log_path, error, command="gui")
            report = error.to_report()
            self.status_var.set(f"{report['code']} at {report['path']}")
            self.messagebox.showerror(
                report["code"],
                f"{report['message']}\n\nenvironment: {report.get('environment')}\npath: {report['path']}",
            )
        else:
            self.status_var.set(f"Unexpected error: {type(error).__name__}")
            self.messagebox.showerror("Unexpected error", f"{type(error).__name__}: {error}")

    def _reload_service(self) -> None:
        replacement = WorkbenchService(
            self.spec_path,
            self.state_root,
            scale_power_override=self.scale_power_override,
        )
        self.service.close()
        self.service = replacement

    def reload_file(self) -> None:
        try:
            self.raw = json.loads(self.spec_path.read_text(encoding="utf-8"))
            self._reload_service()
            self._refresh_editors()
            self.refresh_view()
            self.status_var.set("Reloaded declaration file")
        except Exception as error:
            self._handle_error(error)

    def save_edits(self) -> bool:
        try:
            atomic_write_json(self.spec_path, self.raw)
            self._reload_service()
            self.status_var.set("Edits saved; version state refreshed")
            return True
        except Exception as error:
            self._handle_error(error)
            return False

    def recalculate(self) -> None:
        if not self.save_edits():
            return
        try:
            result = self.service.run_all()
            self.status_var.set(
                f"Run {result['run_id']} {result['status']}; affected environments={result['stats'].get('affected_environment_count')}, keys={result['stats'].get('affected_key_count')}"
            )
            self.refresh_view()
        except Exception as error:
            self._handle_error(error)
            try:
                self.refresh_view()
            except Exception:
                pass

    def refresh_view(self) -> None:
        try:
            view = self.service.get_current_view()
            for tree in (self.summary_tree, self.chain_tree, self.value_tree, self.attribution_tree, self.version_tree):
                tree.delete(*tree.get_children())
            environments = sorted(
                view.get("environments", []),
                key=lambda record: (record["payload"].get("depth", 0), record["env_id"]),
            )
            for record in environments:
                payload = record["payload"]
                self.summary_tree.insert(
                    "",
                    "end",
                    values=(
                        record["env_id"],
                        payload.get("depth"),
                        payload.get("resolved_key_count"),
                        payload.get("unresolved_key_count"),
                        payload.get("source_chain_digest"),
                        payload.get("conclusion_digest"),
                    ),
                )
                for key in sorted(payload.get("keys", {})):
                    key_result = payload["keys"][key]
                    for step in key_result.get("chain", []):
                        self.chain_tree.insert(
                            "",
                            "end",
                            values=(
                                record["env_id"],
                                key,
                                step["rank"],
                                step["source_id"],
                                step["source_type"],
                                step["raw_value"],
                                step["value_fixed"],
                                step["binding_satisfied"],
                                step["selected"],
                            ),
                        )
                    if key_result.get("resolved"):
                        heat = key_result.get("heat_level", "none")
                        tags = (f"heat_{heat}",) if heat in {"low", "medium", "high"} else ()
                        self.value_tree.insert(
                            "",
                            "end",
                            values=(
                                record["env_id"],
                                key,
                                key_result["effective_source_id"],
                                key_result["effective_raw"],
                                key_result["effective_fixed"],
                                key_result["quantization_error_abs"],
                                key_result["quantization_error_bound"],
                                heat,
                            ),
                            tags=tags,
                        )
                for task_id in sorted(payload.get("tasks", {})):
                    task = payload["tasks"][task_id]
                    if task.get("status") != "ok":
                        continue
                    dominant = task["attribution"]["dominant_segment"]
                    for segment in task["attribution"]["segments"]:
                        marker = "*" if segment["segment"] == dominant else ""
                        self.attribution_tree.insert(
                            "",
                            "end",
                            values=(
                                record["env_id"],
                                task_id,
                                task["fixed_result"],
                                task["error_bound"],
                                dominant,
                                segment["segment"] + marker,
                                segment["contribution"],
                                segment["ratio"],
                            ),
                        )
            for env_id in sorted(view.get("unknown_environments", [])):
                self.summary_tree.insert(
                    "",
                    "end",
                    values=(env_id, "", "UNKNOWN", "RERUN REQUIRED", "not committed", "not recoverable"),
                )
            for version in self.service.list_versions():
                self.version_tree.insert(
                    "",
                    "end",
                    values=(
                        version["run_id"],
                        version["status"],
                        version["environment_version"],
                        version["priority_version"],
                        version["scale_version"],
                        version["is_current"],
                        version["stale"],
                        version["content_digest"],
                    ),
                )
            if view.get("stale"):
                self.status_var.set(f"Current view: stale ({view.get('stale_reason')})")
            elif view.get("complete"):
                self.status_var.set(
                    f"Current run {view['run']['run_id']} complete; environments={len(environments)}"
                )
            else:
                self.status_var.set("Current view is incomplete; unknown environments require rerun")
        except Exception as error:
            self._handle_error(error)

    def replay_snapshot(self) -> None:
        try:
            result = self.service.replay_snapshot()
            self.messagebox.showinfo("Snapshot replay", json.dumps(result, ensure_ascii=False, indent=2))
            self.refresh_view()
        except Exception as error:
            self._handle_error(error)

    def _refresh_editors(self) -> None:
        for tree in (self.env_tree, self.source_tree, self.entry_tree):
            tree.delete(*tree.get_children())
        environments = sorted(
            [item for item in self.raw.get("environments", []) if isinstance(item, dict)],
            key=lambda item: (item.get("depth", 0), str(item.get("id", ""))),
        )
        for env in environments:
            self.env_tree.insert(
                "",
                "end",
                values=(
                    env.get("id"),
                    env.get("depth"),
                    ",".join(env.get("available_sources", [])),
                    ",".join(env.get("source_priority", [])),
                ),
            )
        sources = sorted(
            [item for item in self.raw.get("sources", []) if isinstance(item, dict)],
            key=lambda item: str(item.get("id", "")),
        )
        for source in sources:
            self.source_tree.insert("", "end", values=(source.get("id"), source.get("type")))
        env_ids = [str(env.get("id")) for env in environments]
        self.entry_env_combo["values"] = env_ids
        if self.selected_environment not in env_ids:
            self.selected_environment = env_ids[0] if env_ids else None
        self.entry_env_var.set(self.selected_environment or "")
        self._refresh_source_choices()

    def _refresh_source_choices(self) -> None:
        env = self._find_environment(self.entry_env_var.get())
        source_ids = sorted(env.get("available_sources", [])) if env else []
        self.entry_source_combo["values"] = source_ids
        if self.selected_source not in source_ids:
            self.selected_source = source_ids[0] if source_ids else None
        self.entry_source_var.set(self.selected_source or "")
        self._refresh_entries()

    def _refresh_entries(self) -> None:
        self.entry_tree.delete(*self.entry_tree.get_children())
        env = self._find_environment(self.entry_env_var.get())
        source_id = self.entry_source_var.get()
        if not env or not source_id:
            return
        entries = env.get("entries", {}).get(source_id, {})
        if not isinstance(entries, dict):
            return
        for key in sorted(entries):
            self.entry_tree.insert("", "end", values=(key, entries[key]))

    def _environment_selected(self, _event: Any = None) -> None:
        selection = self.env_tree.selection()
        if not selection:
            return
        values = self.env_tree.item(selection[0], "values")
        self.selected_environment = str(values[0])
        self.entry_env_var.set(self.selected_environment)
        self.selected_source = None
        self._refresh_source_choices()

    def _source_selected(self, _event: Any = None) -> None:
        selection = self.source_tree.selection()
        if not selection:
            return
        values = self.source_tree.item(selection[0], "values")
        self.selected_source = str(values[0])

    def _find_environment(self, env_id: str | None) -> dict[str, Any] | None:
        for env in self.raw.get("environments", []):
            if isinstance(env, dict) and str(env.get("id")) == env_id:
                return env
        return None

    def add_environment(self) -> None:
        dialog = FormDialog(
            self.tk,
            self.root,
            "Add environment",
            (
                ("id", "Environment id", "new_env"),
                ("depth", "Hierarchy depth (non-negative integer)", "0"),
                ("sources", "Available sources (comma-separated)", "base"),
                ("priority", "Explicit priority order (comma-separated)", "base"),
            ),
        )
        if dialog.result is None:
            return
        try:
            env_id = dialog.result["id"].strip()
            depth = int(dialog.result["depth"].strip())
            sources = [part.strip() for part in dialog.result["sources"].split(",") if part.strip()]
            priority = [part.strip() for part in dialog.result["priority"].split(",") if part.strip()]
            if not env_id or depth < 0 or not sources or set(sources) != set(priority):
                raise ValueError("id/source/priority fields are invalid")
            if any(isinstance(env, dict) and env.get("id") == env_id for env in self.raw.get("environments", [])):
                raise ValueError(f"environment {env_id!r} already exists")
            self.raw.setdefault("environments", []).append(
                {
                    "id": env_id,
                    "depth": depth,
                    "available_sources": sources,
                    "source_priority": priority,
                    "bindings": {},
                    "entries": {source_id: {} for source_id in sources},
                }
            )
            self._refresh_editors()
        except Exception as error:
            self.messagebox.showerror("Invalid environment", str(error))

    def edit_environment(self) -> None:
        env = self._find_environment(self.selected_environment)
        if env is None:
            self.messagebox.showinfo("Edit environment", "Select an environment first")
            return
        old_id = str(env.get("id"))
        dialog = FormDialog(
            self.tk,
            self.root,
            "Edit environment",
            (
                ("id", "Environment id", old_id),
                ("depth", "Hierarchy depth (non-negative integer)", str(env.get("depth", 0))),
                ("sources", "Available sources (comma-separated)", ",".join(env.get("available_sources", []))),
                ("priority", "Explicit priority order (comma-separated)", ",".join(env.get("source_priority", []))),
            ),
        )
        if dialog.result is None:
            return
        try:
            new_id = dialog.result["id"].strip()
            depth = int(dialog.result["depth"].strip())
            sources = [part.strip() for part in dialog.result["sources"].split(",") if part.strip()]
            priority = [part.strip() for part in dialog.result["priority"].split(",") if part.strip()]
            if not new_id or depth < 0 or not sources or set(sources) != set(priority):
                raise ValueError("id/source/priority fields are invalid")
            if new_id != old_id and any(
                isinstance(other, dict) and other.get("id") == new_id
                for other in self.raw.get("environments", [])
            ):
                raise ValueError(f"environment {new_id!r} already exists")
            env["id"] = new_id
            env["depth"] = depth
            env["available_sources"] = sources
            env["source_priority"] = priority
            entries = env.setdefault("entries", {})
            for source_id in sources:
                entries.setdefault(source_id, {})
            for source_id in list(entries):
                if source_id not in sources:
                    del entries[source_id]
            bindings = env.setdefault("bindings", {})
            for key in list(bindings):
                binding = bindings[key]
                allowed = binding.get("allowed_source_ids") if isinstance(binding, dict) else None
                if isinstance(allowed, list) and any(source_id not in sources for source_id in allowed):
                    del bindings[key]
            self.selected_environment = new_id
            self._refresh_editors()
        except Exception as error:
            self.messagebox.showerror("Invalid environment", str(error))

    def delete_environment(self) -> None:
        env = self._find_environment(self.selected_environment)
        if env is None:
            return
        if not self.messagebox.askyesno("Delete environment", f"Delete environment {env.get('id')!r}?"):
            return
        self.raw["environments"] = [item for item in self.raw.get("environments", []) if item is not env]
        self.selected_environment = None
        self._refresh_editors()

    def add_source(self) -> None:
        dialog = FormDialog(
            self.tk,
            self.root,
            "Add source",
            (
                ("id", "Source id", "new_source"),
                ("type", f"Type ({', '.join(SOURCE_TYPES)})", "baseline"),
            ),
        )
        if dialog.result is None:
            return
        source_id = dialog.result["id"].strip()
        source_type = dialog.result["type"].strip()
        if not source_id or source_type not in SOURCE_TYPES:
            self.messagebox.showerror("Invalid source", "Source id is required and type must be supported")
            return
        if any(isinstance(item, dict) and item.get("id") == source_id for item in self.raw.get("sources", [])):
            self.messagebox.showerror("Invalid source", f"Source {source_id!r} already exists")
            return
        self.raw.setdefault("sources", []).append({"id": source_id, "type": source_type})
        self._refresh_editors()

    def edit_source(self) -> None:
        if not self.selected_source:
            return
        source = next(
            (item for item in self.raw.get("sources", []) if isinstance(item, dict) and item.get("id") == self.selected_source),
            None,
        )
        if source is None:
            return
        dialog = FormDialog(
            self.tk,
            self.root,
            "Edit source",
            (
                ("type", f"Type ({', '.join(SOURCE_TYPES)})", str(source.get("type"))),
            ),
        )
        if dialog.result is None:
            return
        source_type = dialog.result["type"].strip()
        if source_type not in SOURCE_TYPES:
            self.messagebox.showerror("Invalid source", "Unsupported source type")
            return
        source["type"] = source_type
        self._refresh_editors()

    def delete_source(self) -> None:
        if not self.selected_source:
            return
        used = False
        for env in self.raw.get("environments", []):
            if isinstance(env, dict) and self.selected_source in env.get("available_sources", []):
                used = True
                break
        if used:
            self.messagebox.showerror("Source in use", "Remove the source from every environment first")
            return
        if not self.messagebox.askyesno("Delete source", f"Delete source {self.selected_source!r}?"):
            return
        self.raw["sources"] = [item for item in self.raw.get("sources", []) if item.get("id") != self.selected_source]
        self.selected_source = None
        self._refresh_editors()

    def add_entry(self) -> None:
        self._edit_entry(existing_key=None)

    def edit_entry(self) -> None:
        selection = self.entry_tree.selection()
        if not selection:
            return
        key = str(self.entry_tree.item(selection[0], "values")[0])
        self._edit_entry(existing_key=key)

    def _edit_entry(self, existing_key: str | None) -> None:
        env = self._find_environment(self.entry_env_var.get())
        source_id = self.entry_source_var.get()
        if env is None or not source_id:
            self.messagebox.showinfo("Source entry", "Select an environment and source first")
            return
        entries = env.setdefault("entries", {}).setdefault(source_id, {})
        current_value = "" if existing_key is None else str(entries.get(existing_key, ""))
        dialog = FormDialog(
            self.tk,
            self.root,
            "Add source entry" if existing_key is None else "Edit source entry",
            (
                ("key", "Numeric key", existing_key or "cpu_cores"),
                ("value", "Value (JSON number)", current_value or "1.0"),
            ),
        )
        if dialog.result is None:
            return
        key = dialog.result["key"].strip()
        try:
            if not key:
                raise ValueError("key must not be empty")
            value = json.loads(dialog.result["value"].strip())
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("value must be a JSON number")
        except Exception as error:
            self.messagebox.showerror("Invalid entry", str(error))
            return
        if existing_key is not None and existing_key != key:
            entries.pop(existing_key, None)
        entries[key] = value
        self._refresh_entries()

    def delete_entry(self) -> None:
        selection = self.entry_tree.selection()
        if not selection:
            return
        env = self._find_environment(self.entry_env_var.get())
        source_id = self.entry_source_var.get()
        if env is None or not source_id:
            return
        key = str(self.entry_tree.item(selection[0], "values")[0])
        env.setdefault("entries", {}).setdefault(source_id, {}).pop(key, None)
        self._refresh_entries()

    def open_historical(self) -> None:
        selection = self.version_tree.selection()
        if not selection:
            return
        run_id = int(self.version_tree.item(selection[0], "values")[0])
        try:
            payload = self.service.get_historical_view(run_id)
        except Exception as error:
            self._handle_error(error)
            return
        window = self.tk.Toplevel(self.root)
        window.title(f"Historical run {run_id} (read-only)")
        window.geometry("1000x700")
        text = self.tk.Text(window, wrap="none")
        vertical = self.ttk.Scrollbar(window, orient="vertical", command=text.yview)
        horizontal = self.ttk.Scrollbar(window, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        text.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        text.insert("1.0", json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        text.configure(state="disabled")

    def close(self) -> None:
        try:
            self.service.close()
        finally:
            self.root.destroy()
