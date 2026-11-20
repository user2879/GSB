"""Tkinter diagnostics console.

The view model is built by ``build_view_model`` (pure, headless-testable);
``DiagnosticsApp`` only renders it. All render ordering is stable (sorted
keys everywhere).
"""

from __future__ import annotations

from typing import List

from .engine import RunResult
from .ledger import Ledger


def build_view_model(result: RunResult, ledger: Ledger) -> dict:
    rows = ledger.ledger_rows(result.config_version)
    batches = ledger.batch_rows(result.config_version)
    t0 = min((b["wait_start_ms"] for b in batches), default=0)
    timeline = [{
        "batch_id": b["batch_id"],
        "backend_id": b["backend_id"],
        "queue_ms": b["wait_end_ms"] - b["wait_start_ms"],
        "run_start": b["start_ms"] - t0,
        "run_end": b["end_ms"] - t0,
        "duration_ms": b["duration_ms"],
    } for b in batches]
    timeline.sort(key=lambda t: (t["run_start"], t["batch_id"]))
    return {
        "backends": [
            {"model_id": b.model_id, "port": b.port,
             "max_concurrency": b.max_concurrency, "batch_size": b.batch_size,
             "fingerprint": result.fingerprints[b.model_id]}
            for b in result.decl.backends
        ],
        "canary": {
            rule: {variant: {"value_bp": r.value_bp,
                             "numerator": r.numerator,
                             "denominator": r.denominator,
                             "error_bound_bp": r.error_bound_bp,
                             "rounding": r.rounding}
                   for variant, r in sorted(variants.items())}
            for rule, variants in sorted(result.canary_distribution.items())
        },
        "ledger": rows,  # already ordered by request_seq
        "timeline": timeline,
        "strategy": {
            "strategy": result.decision.strategy,
            "reason": result.decision.reason,
            "scale": result.decision.scale,
            "estimates": result.decision.estimates,
            "tie_threshold": result.decision.tie_threshold,
        },
        "config": {
            "version": result.config_version,
            "hash": result.config_hash,
            "stale_versions": sorted(result.stale_versions),
            "snapshots": ledger.snapshots(),
        },
        "concurrency": {
            "peak_running": result.peak_running,
            "backend_peak": dict(sorted(result.backend_peak.items())),
        },
    }


class DiagnosticsApp:
    def __init__(self, view: dict):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Inference Gateway Routing Diagnostics")
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)

        # Backends + fingerprints
        frame = ttk.Frame(nb)
        nb.add(frame, text="Backends")
        tree = ttk.Treeview(frame, columns=("model_id", "port", "conc",
                                            "batch", "fingerprint"),
                            show="headings")
        for col in ("model_id", "port", "conc", "batch", "fingerprint"):
            tree.heading(col, text=col)
        for b in view["backends"]:
            tree.insert("", "end", values=(b["model_id"], b["port"],
                                           b["max_concurrency"],
                                           b["batch_size"], b["fingerprint"]))
        tree.pack(fill="both", expand=True)

        # Canary distribution per rule
        frame = ttk.Frame(nb)
        nb.add(frame, text="Canary")
        tree = ttk.Treeview(frame, columns=("rule", "variant", "bp", "hits",
                                            "bound"), show="headings")
        for col in ("rule", "variant", "bp", "hits", "bound"):
            tree.heading(col, text=col)
        for rule, variants in view["canary"].items():
            for variant, r in variants.items():
                tree.insert("", "end", values=(
                    rule, variant, f"{r['value_bp']}/10000",
                    f"{r['numerator']}/{r['denominator']}",
                    f"+/-{r['error_bound_bp']}bp {r['rounding']}"))
        tree.pack(fill="both", expand=True)

        # Per-request ledger
        frame = ttk.Frame(nb)
        nb.add(frame, text="Ledger")
        cols = ("seq", "rule", "backend", "variant", "batch", "fingerprint",
                "summary", "ms")
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
        for row in view["ledger"]:
            tree.insert("", "end", values=(
                row["request_seq"], row["rule_id"], row["backend_id"],
                row["variant"], row["batch_id"], row["fingerprint"],
                row["output_summary"], row["duration_ms"]))
        tree.pack(fill="both", expand=True)

        # Queue/concurrency timeline
        frame = ttk.Frame(nb)
        nb.add(frame, text="Timeline")
        canvas = tk.Canvas(frame, bg="white", height=320)
        canvas.pack(fill="both", expand=True)
        scale = 4
        for i, item in enumerate(view["timeline"]):
            y = 20 + i * 24
            x0 = 120 + item["run_start"] * scale
            x1 = 120 + max(item["run_end"], item["run_start"] + 1) * scale
            canvas.create_text(110, y + 6, text=item["batch_id"], anchor="e")
            canvas.create_rectangle(x0, y, x1, y + 12, fill="#4a90d9")
            canvas.create_text(x1 + 4, y + 6, anchor="w",
                               text=f"q{item['queue_ms']}ms "
                                    f"{item['duration_ms']}ms")

        # Strategy + config versions
        frame = ttk.Frame(nb)
        nb.add(frame, text="Strategy/Config")
        text = tk.Text(frame, wrap="word")
        s = view["strategy"]
        text.insert("end", f"strategy: {s['strategy']}\n")
        text.insert("end", f"reason: {s['reason']}\n")
        text.insert("end", f"scale: {s['scale']}\n")
        text.insert("end", f"estimates: {s['estimates']} "
                           f"(tie threshold {s['tie_threshold']})\n\n")
        c = view["config"]
        text.insert("end", f"config version: {c['version']} "
                           f"hash={c['hash']}\n")
        text.insert("end", f"stale versions: {c['stale_versions']}\n")
        for snap in c["snapshots"]:
            text.insert("end", f"  snapshot v{snap['config_version']} "
                               f"hash={snap['config_hash']}\n")
        conc = view["concurrency"]
        text.insert("end", f"\npeak running: {conc['peak_running']}\n")
        text.insert("end", f"backend peak: {conc['backend_peak']}\n")
        text.configure(state="disabled")
        text.pack(fill="both", expand=True)

    def mainloop(self) -> None:
        self.root.mainloop()


def launch(result: RunResult, ledger: Ledger) -> None:
    DiagnosticsApp(build_view_model(result, ledger)).mainloop()
