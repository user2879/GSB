"""Tkinter diagnostic dashboard with a headless rendering fallback.

The dashboard only reads engine and ledger state; refresh never writes to the
ledger. When tkinter/tcl is unavailable (for example a minimal Linux CI),
render_headless() produces the same stable ordered sections as plain text.
"""


def render_headless(engine, ledger) -> str:
    sections = []
    sections.append("== Node leases / resource usage ==")
    for node_id in sorted(engine.manifest.nodes):
        active = engine.leases.active_count(node_id)
        concurrency, lease_limit = engine.quota[node_id]
        sections.append(
            f"node={node_id} active_leases={active} concurrency={concurrency} "
            f"lease_limit={lease_limit} global_units={engine.leases.global_units()} "
            f"global_budget={engine.manifest.global_budget}"
        )

    sections.append("")
    sections.append("== Ready shards (stable shard-id order) ==")
    ready = engine.ready_shards()
    sections.append(", ".join(ready) if ready else "(none)")

    sections.append("")
    sections.append("== Queued shards ==")
    queued = engine.queued_shards()
    sections.append(", ".join(queued) if queued else "(none)")

    sections.append("")
    sections.append("== Invalidation sets and propagation basis ==")
    if not engine.invalidations:
        sections.append("(none)")
    for item in engine.invalidations:
        sections.append(
            f"revision={item['revision']} invalidated={item['shards']}"
        )
        for step in item["basis"]:
            sections.append(f"  - {step['step']}: {step}")

    sections.append("")
    sections.append("== Unknown shards ==")
    unknown = ledger.unknown_shards()
    sections.append(str(unknown) if unknown else "(none)")

    sections.append("")
    sections.append("== Ledger version / snapshots ==")
    sections.append("format_version=1")
    for snapshot in ledger.snapshots():
        sections.append(f"snapshot={snapshot}")
    return "\n".join(sections)


class Dashboard:
    def __init__(self, engine, ledger, refresh_ms=500):
        self.engine = engine
        self.ledger = ledger
        self.refresh_ms = refresh_ms
        self.root = None
        self.text = None

    def open(self):
        import tkinter as tk

        self.root = tk.Tk()
        self.root.title("Shard Quota Ledger Diagnostics")
        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")
        self.text = tk.Text(frame, width=110, height=34, yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text.yview)
        self._refresh()
        self.root.after(self.refresh_ms, self._tick)
        self.root.mainloop()

    def _tick(self):
        self._refresh()
        if self.root is not None:
            self.root.after(self.refresh_ms, self._tick)

    def _refresh(self):
        content = render_headless(self.engine, self.ledger)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")


def open_dashboard(engine, ledger):
    try:
        Dashboard(engine, ledger).open()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"tk unavailable ({exc}); headless snapshot:\n" + render_headless(engine, ledger)
    return "dashboard closed"
