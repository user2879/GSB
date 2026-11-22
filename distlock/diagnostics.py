"""Tkinter diagnostic dashboard with a headless text-renderer fallback.

The dashboard renders a stable, read-only view: domains are shown in sorted
domain-id order, holders in sorted holder-id order, and queued requests in
their authoritative priority/seq order. Refreshing only calls the read-only
STATUS endpoint and never writes to the ledger.

When Tcl/Tk is missing (e.g. minimal Linux/WSL Python) or no display is
available, :func:`build_dashboard` returns ``None`` and callers use
:func:`render_text`, which produces the identical data as plain text. On the
target Windows environment with the official Python installer (tcl/tk enabled)
the real Tk window is used.
"""
from __future__ import annotations

from typing import Any, Callable

TK_AVAILABLE = False
TK_IMPORT_ERROR: Exception | None = None
try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
    TK_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    TK_IMPORT_ERROR = exc


def render_text(view: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("DistLock Diagnostic Dashboard (text mode)")
    lines.append(
        f"domain_set_version={view['domain_set_version']} "
        f"scale_factor={view['scale_factor']} truncation={view['truncation']} "
        f"error_bound_fixed<{view['error_bound_fixed']} (fixed-point units)")
    lines.append(
        f"ledger cursor={view['ledger_cursor']} "
        f"batches={view['ledger_batch_count']} digest={view['ledger_digest']}")
    lines.append("generations=" + ",".join(
        f"{k}:{v}" for k, v in sorted(view["generations"].items())))
    lines.append("")
    lines.append("Holders (sorted, stable):")
    for hid, info in view["holders"].items():
        lines.append(
            f"  {hid:<10} priority={info['priority']:<3} "
            f"targets={','.join(info['target_domains'])}")
    lines.append("")
    for did in sorted(view["domains"]):
        dom = view["domains"][did]
        wm = dom["watermark"]
        lines.append(
            f"Domain {did} v{dom['domain_version']} "
            f"capacity={dom['capacity']} active={len(dom['active_leases'])} "
            f"free={wm['free_capacity']} queued={wm['queue_length']} "
            f"water={wm['ratio_fixed']}/{wm['scale']} "
            f"(floor, err<{wm['error_bound_fixed']})")
        for lease in dom["active_leases"]:
            lines.append(
                f"  - holder={lease['holder_id']:<10} state={lease['state']:<8}"
                f" remaining_ms={lease['remaining_ms']}")
        for waiter in dom["queue"]:
            lines.append(
                f"  * queued seq={waiter['seq']} holder={waiter['holder_id']}"
                f" priority={waiter['priority']}")
    lines.append("")
    sched = view.get("scheduler", {})
    lines.append(
        f"scheduler peak_inflight={sched.get('peak_inflight_total')} "
        f"global_limit={sched.get('global_limit')} "
        f"per_holder_limit={sched.get('per_holder_limit')}")
    return "\n".join(lines)


if TK_AVAILABLE:
    class Dashboard:  # pragma: no cover - exercised on Windows GUI
        def __init__(self, view_provider: Callable[[], dict[str, Any]],
                     refresh_ms: int = 1000):
            self.provider = view_provider
            self.refresh_ms = refresh_ms
            self.root = tk.Tk()
            self.root.title("DistLock Diagnostic Dashboard")
            self.text = tk.Text(self.root, width=110, height=34,
                                font=("Consolas", 10), wrap="none")
            self.text.pack(side="left", fill="both", expand=True)
            scroll = ttk.Scrollbar(self.root, command=self.text.yview)
            scroll.pack(side="right", fill="y")
            self.text.configure(yscrollcommand=scroll.set, state="disabled")
            self._tick()

        def _tick(self) -> None:
            view = self.provider()  # read-only STATUS; ledger untouched
            content = render_text(view)
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("1.0", content)
            self.text.configure(state="disabled")
            self.root.after(self.refresh_ms, self._tick)

        def run(self) -> None:
            self.root.mainloop()


def build_dashboard(view_provider: Callable[[], dict[str, Any]],
                    refresh_ms: int = 1000):
    """Return a Tk Dashboard when possible, else None (caller uses text mode)."""
    if not TK_AVAILABLE:
        return None
    try:  # pragma: no cover - environment dependent
        return Dashboard(view_provider, refresh_ms)
    except Exception:
        return None
