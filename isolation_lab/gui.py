from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .models import SchedulerParams
from .scheduler import Engine
from .scripts import load_script
from .snapshot import replay_snapshot, restore_snapshot, save_snapshot
from .storage import SnapshotStore, Storage


def launch_gui(*, db_path: str, snapshots_path: str) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError as exc:
        raise BusinessError(
            "E_SCRIPT",
            "tkinter is not available in this Python installation; install the standard Tk bindings",
            path="$.gui",
            details={"python": __import__("sys").executable},
        ) from exc

    class Workbench:
        def __init__(self, root: Any) -> None:
            self.root = root
            self.root.title("Transaction Isolation & Lock Lab")
            self.root.geometry("1180x820")
            self.db_path = tk.StringVar(value=db_path)
            self.snapshots_path = tk.StringVar(value=snapshots_path)
            self.script_path = tk.StringVar()
            self.snapshot_id = tk.StringVar(value="gui-snapshot")
            self.global_limit = tk.IntVar(value=2)
            self.status = tk.StringVar(value="ready")
            self.current_script = None
            self.current_result = None
            self.storage: Storage | None = None
            self.running = False
            self.event_queue: queue.Queue[Any] = queue.Queue()
            self.row_state: dict[str, dict[str, Any]] = {}
            self.session_state: dict[str, dict[str, Any]] = {}
            self._build()
            self.root.after(100, self._poll_events)

        def _build(self) -> None:
            top = ttk.Frame(self.root, padding=8)
            top.pack(fill="x")
            ttk.Label(top, text="Database").grid(row=0, column=0, sticky="w")
            ttk.Entry(top, textvariable=self.db_path, width=34).grid(row=0, column=1, sticky="ew")
            ttk.Label(top, text="Snapshots").grid(row=0, column=2, sticky="w")
            ttk.Entry(top, textvariable=self.snapshots_path, width=30).grid(row=0, column=3, sticky="ew")
            ttk.Label(top, text="Global limit").grid(row=0, column=4, sticky="w")
            ttk.Spinbox(top, from_=1, to=12, textvariable=self.global_limit, width=5).grid(row=0, column=5)

            second = ttk.Frame(self.root, padding=(8, 0, 8, 8))
            second.pack(fill="x")
            ttk.Label(second, text="Script").grid(row=0, column=0, sticky="w")
            ttk.Entry(second, textvariable=self.script_path).grid(row=0, column=1, sticky="ew")
            ttk.Button(second, text="Browse", command=self._browse).grid(row=0, column=2, padx=4)
            ttk.Button(second, text="Validate", command=self._validate).grid(row=0, column=3, padx=2)
            ttk.Button(second, text="Execute", command=self._execute).grid(row=0, column=4, padx=2)
            ttk.Label(second, text="Snapshot id").grid(row=0, column=5, padx=(10, 0))
            ttk.Entry(second, textvariable=self.snapshot_id, width=20).grid(row=0, column=6)
            ttk.Button(second, text="Save snapshot", command=self._save_snapshot).grid(row=0, column=7, padx=2)
            ttk.Button(second, text="Restore", command=self._restore_snapshot).grid(row=0, column=8, padx=2)
            ttk.Button(second, text="Replay", command=self._replay_snapshot).grid(row=0, column=9, padx=2)
            ttk.Button(second, text="Ledger", command=self._show_ledger).grid(row=0, column=10, padx=2)
            second.columnconfigure(1, weight=1)

            panes = ttk.Panedwindow(self.root, orient="vertical")
            panes.pack(fill="both", expand=True, padx=8, pady=(0, 8))

            upper = ttk.Notebook(panes)
            session_frame = ttk.Frame(upper)
            lock_frame = ttk.Frame(upper)
            upper.add(session_frame, text="Sessions")
            upper.add(lock_frame, text="Locks and waits")

            session_columns = (
                "session",
                "phase",
                "isolation",
                "wait_budget",
                "held_locks",
                "waiting_for",
                "business_code",
                "data_version",
            )
            self.sessions_tree = ttk.Treeview(session_frame, columns=session_columns, show="headings")
            for column in session_columns:
                self.sessions_tree.heading(column, text=column.replace("_", " ").title())
                self.sessions_tree.column(column, width=135, anchor="w")
            self.sessions_tree.pack(fill="both", expand=True)

            lock_columns = ("row", "owners", "queue", "next_request", "grant_sequence")
            self.locks_tree = ttk.Treeview(lock_frame, columns=lock_columns, show="headings")
            for column in lock_columns:
                self.locks_tree.heading(column, text=column.replace("_", " ").title())
                self.locks_tree.column(column, width=205, anchor="w")
            self.locks_tree.pack(fill="both", expand=True)

            panes.add(upper, weight=3)
            log_frame = ttk.Frame(panes)
            self.log = tk.Text(log_frame, height=14, wrap="none")
            self.log.pack(fill="both", expand=True)
            panes.add(log_frame, weight=2)

            ttk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=(0, 6))

        def _browse(self) -> None:
            selected = filedialog.askopenfilename(
                title="Select edit script",
                filetypes=(("JSON scripts", "*.json"), ("All files", "*.*")),
            )
            if selected:
                self.script_path.set(selected)

        def _append_log(self, text: str) -> None:
            self.log.insert("end", text + "\n")
            self.log.see("end")

        def _validate(self) -> None:
            try:
                script = load_script(self.script_path.get())
            except BusinessError as exc:
                messagebox.showerror("E_SCRIPT", str(exc))
                return
            self.current_script = script
            self.status.set(
                f"valid source={script.source_version} normalized={script.normalized_version} "
                f"sessions={len(script.sessions)}"
            )
            self._append_log(self.status.get())
            self.session_state = {
                session.session_id: {
                    "session": session.session_id,
                    "phase": "pending",
                    "isolation": session.isolation_level,
                    "wait_budget": f"0/{session.lock_wait_budget_ms}",
                    "held_locks": "-",
                    "waiting_for": "-",
                    "business_code": "-",
                    "data_version": "-",
                }
                for session in script.sessions
            }
            self._refresh_sessions()

        def _execute(self) -> None:
            if self.running:
                messagebox.showwarning("execute", "a script is already running")
                return
            try:
                script = load_script(self.script_path.get())
            except BusinessError as exc:
                messagebox.showerror("E_SCRIPT", str(exc))
                return
            self.current_script = script
            self.current_result = None
            self.running = True
            self.row_state.clear()
            self.session_state = {
                session.session_id: {
                    "session": session.session_id,
                    "phase": "pending",
                    "isolation": session.isolation_level,
                    "wait_budget": f"0/{session.lock_wait_budget_ms}",
                    "held_locks": "-",
                    "waiting_for": "-",
                    "business_code": "-",
                    "data_version": "-",
                }
                for session in script.sessions
            }
            self._refresh_sessions()
            self._refresh_locks()
            self.status.set("executing")
            self._append_log(f"execute script={script.script_id}")
            if self.storage is not None:
                self.storage.close()
            self.storage = Storage(self.db_path.get())
            params = SchedulerParams(
                global_concurrency_limit=int(self.global_limit.get()),
                clock_quantum_ms=1,
            )

            def observer(event: Any) -> None:
                self.event_queue.put(("event", event))

            def worker() -> None:
                try:
                    result = Engine(self.storage, params, observer=observer).run(script)
                    self.event_queue.put(("done", result))
                except Exception as exc:  # The queue carries the exact exception to the UI.
                    self.event_queue.put(("error", exc))

            threading.Thread(target=worker, daemon=True).start()

        def _poll_events(self) -> None:
            try:
                while True:
                    kind, payload = self.event_queue.get_nowait()
                    if kind == "event":
                        self._handle_event(payload)
                    elif kind == "done":
                        self.running = False
                        self.current_result = payload
                        self.status.set(
                            f"committed data_version={payload.global_data_version} "
                            f"sessions={len(payload.outcomes)}"
                        )
                        self._append_log(self.status.get())
                        self._apply_result(payload)
                    elif kind == "error":
                        self.running = False
                        if isinstance(payload, BusinessError):
                            messagebox.showerror(payload.code, str(payload))
                        else:
                            messagebox.showerror("execution failed", str(payload))
                        self.status.set("execution failed")
            except queue.Empty:
                pass
            self.root.after(80, self._poll_events)

        def _handle_event(self, event: Any) -> None:
            payload = event.payload
            if event.event_type == "session.phase" and event.session_id:
                state = self.session_state.setdefault(event.session_id, {"session": event.session_id})
                state["phase"] = payload.get("phase", "-")
                state["wait_budget"] = (
                    f"{payload.get('wait_used_ms', 0)}/{payload.get('lock_wait_budget_ms', '-')}"
                )
                held = payload.get("held_locks") or []
                state["held_locks"] = (
                    ",".join(f"{item['row_key']}:{item['mode']}:g{item['grant_seq']}" for item in held)
                    if held else "-"
                )
                waiting_row = payload.get("waiting_row_key")
                if waiting_row:
                    state["waiting_for"] = (
                        f"{waiting_row} <- {','.join(payload.get('blocked_by') or ['queue'])}"
                    )
                else:
                    state["waiting_for"] = "-"
                if payload.get("business_code"):
                    state["business_code"] = payload["business_code"]
                self._refresh_sessions()
            elif event.event_type in ("lock.granted", "lock.waiting", "lock.released", "lock.wait_cancelled"):
                row_key = payload.get("row_key")
                if not row_key:
                    return
                self.row_state[row_key] = {
                    "row": row_key,
                    "owners": payload.get("owners", []),
                    "queue": payload.get("queue", []),
                    "next_request": payload.get("request_seq", "-"),
                    "grant_sequence": payload.get("grant_seq", "-"),
                }
                self._refresh_locks()
            if event.event_type in (
                "lock.waiting",
                "lock.granted",
                "lock.released",
                "session.lock_timeout",
                "deadlock.detected",
                "deadlock.victim",
                "session.commit",
                "session.abort",
            ):
                compact = {
                    key: payload.get(key)
                    for key in ("row_key", "mode", "request_seq", "grant_seq", "reason", "blocked_by", "business_code", "victim")
                    if key in payload
                }
                self._append_log(
                    f"{event.event_seq:04d} t={event.virtual_ms} {event.event_type} "
                    f"session={event.session_id or '-'} op={event.op_no if event.op_no is not None else '-'} "
                    f"{compact}"
                )

        def _apply_result(self, result: Any) -> None:
            self.session_state = {}
            for outcome in result.outcomes:
                held = "-"
                waiting = "-"
                for operation in outcome.operations:
                    if operation.wait and operation.wait.granted_virtual_ms is None:
                        waiting = " <- ".join([operation.wait.row_key, *operation.wait.blocked_by])
                self.session_state[outcome.session_id] = {
                    "session": outcome.session_id,
                    "phase": outcome.outcome,
                    "isolation": outcome.isolation_level,
                    "wait_budget": f"{outcome.wait_used_ms}/{next((s.lock_wait_budget_ms for s in self.current_script.sessions if s.session_id == outcome.session_id), '-')}",
                    "held_locks": held,
                    "waiting_for": waiting,
                    "business_code": outcome.business_code or "-",
                    "data_version": outcome.committed_data_version,
                }
            self._refresh_sessions()
            self.row_state = {}
            for row in result.lock_state.get("rows", []):
                self.row_state[row["row_key"]] = {
                    "row": row["row_key"],
                    "owners": row.get("owners", []),
                    "queue": row.get("queue", []),
                    "next_request": "-",
                    "grant_sequence": "-",
                }
            self._refresh_locks()

        def _refresh_sessions(self) -> None:
            for item in self.sessions_tree.get_children():
                self.sessions_tree.delete(item)
            for session_id in sorted(self.session_state):
                state = self.session_state[session_id]
                values = tuple(state.get(column, "-") for column in self.sessions_tree["columns"])
                self.sessions_tree.insert("", "end", values=values)

        def _refresh_locks(self) -> None:
            for item in self.locks_tree.get_children():
                self.locks_tree.delete(item)
            for row_key in sorted(self.row_state):
                state = self.row_state[row_key]
                owners = "; ".join(
                    f"{owner['session_id']}:{owner['mode']}:g{owner['grant_seq']}"
                    for owner in state.get("owners", [])
                ) or "-"
                queue = "; ".join(
                    f"#{request['request_seq']}:{request['session_id']}:{request['mode']}"
                    for request in sorted(state.get("queue", []), key=lambda item: item["request_seq"])
                ) or "-"
                self.locks_tree.insert(
                    "",
                    "end",
                    values=(row_key, owners, queue, state.get("next_request", "-"), state.get("grant_sequence", "-")),
                )

        def _save_snapshot(self) -> None:
            if self.current_result is None or self.current_script is None:
                messagebox.showwarning("snapshot", "execute a script before saving a snapshot")
                return
            snapshot_id = self.snapshot_id.get().strip()
            if not snapshot_id:
                messagebox.showwarning("snapshot", "snapshot id is required")
                return
            with SnapshotStore(self.snapshots_path.get()) as store:
                save_snapshot(store, snapshot_id, self.current_script, self.current_result)
            self.status.set(f"snapshot saved={snapshot_id}")
            self._append_log(self.status.get())

        def _restore_snapshot(self) -> None:
            snapshot_id = self.snapshot_id.get().strip()
            try:
                with SnapshotStore(self.snapshots_path.get()) as store, Storage(self.db_path.get()) as target:
                    report = restore_snapshot(store, snapshot_id, target_storage=target)
                    snapshot = store.load(snapshot_id)
            except BusinessError as exc:
                messagebox.showerror(exc.code, str(exc))
                return
            self.status.set(
                f"restored={report.snapshot_id} mode={report.mode} data_version={report.data_version} "
                f"changed_sessions={','.join(report.changed_sessions) or '-'} "
                f"cleared_locks={report.cleared_lock_entries}"
            )
            self._append_log(self.status.get())
            self.session_state = {}
            for outcome in snapshot["outcomes"]:
                self.session_state[outcome["session_id"]] = {
                    "session": outcome["session_id"],
                    "phase": outcome.get("outcome", "-"),
                    "isolation": outcome.get("isolation_level", "-"),
                    "wait_budget": f"{outcome.get('wait_used_ms', 0)}/{next((s.lock_wait_budget_ms for s in self.current_script.sessions if s.session_id == outcome['session_id']), '-') if self.current_script else '-'}",
                    "held_locks": "-",
                    "waiting_for": "-",
                    "business_code": outcome.get("business_code") or "-",
                    "data_version": outcome.get("committed_data_version"),
                }
            self._refresh_sessions()
            self.row_state = {}
            for row in snapshot.get("lock_table", {}).get("rows", []):
                self.row_state[row["row_key"]] = {
                    "row": row["row_key"],
                    "owners": row.get("owners", []),
                    "queue": row.get("queue", []),
                    "next_request": "-",
                    "grant_sequence": "-",
                }
            self._refresh_locks()
            for event in snapshot.get("trace", [])[-120:]:
                self._append_log(
                    f"{int(event['event_seq']):04d} t={event['virtual_ms']} {event['event_type']} "
                    f"session={event.get('session_id') or '-'} op={event.get('op_no') if event.get('op_no') is not None else '-'}"
                )

        def _replay_snapshot(self) -> None:
            snapshot_id = self.snapshot_id.get().strip()
            try:
                with SnapshotStore(self.snapshots_path.get()) as store:
                    report, replay = replay_snapshot(store, snapshot_id, raise_on_error=False)
            except BusinessError as exc:
                messagebox.showerror(exc.code, str(exc))
                return
            if report.reproducible:
                self.status.set(
                    f"replay=REPRODUCIBLE snapshot={snapshot_id} data_version={replay.global_data_version}"
                )
            else:
                self.status.set(
                    f"E_REPLAY session={report.session_id or '-'} op={report.op_no if report.op_no is not None else '-'} "
                    f"field={report.field or '-'} expected={report.expected} actual={report.actual}"
                )
                messagebox.showerror("E_REPLAY", self.status.get())
            self._append_log(self.status.get())

        def _show_ledger(self) -> None:
            try:
                with Storage(self.db_path.get()) as storage:
                    records = storage.read_ledger()
            except BusinessError as exc:
                messagebox.showerror(exc.code, str(exc))
                return
            self._append_log(f"ledger_records={len(records)}")
            for record in records[-200:]:
                self._append_log(
                    f"ledger event={record.event_id} source={record.source_version} "
                    f"normalized={record.normalized_version} type={record.event_type} "
                    f"session={record.session_id or '-'} op={record.op_no if record.op_no is not None else '-'}"
                )

    root = tk.Tk()
    Workbench(root)
    root.mainloop()
