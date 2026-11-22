"""Incremental cached views and an optional Tk dashboard.

The view is a fast, read-only projection of ledger caches.  It never recomputes
authority; that separation lives in :mod:`meshbench.replay`.  A headless text
renderer always works, while the Tk dashboard is imported lazily so the domain
components stay usable on machines without Tcl/Tk.
"""

import json

from .budget import OUTCOMES


def _numpy_percentile(values, percentile):
    if not values:
        return 0
    try:
        import numpy as np
        return int(round(float(np.percentile(np.array(values, dtype="int64"), percentile))))
    except ImportError:  # pragma: no cover - numpy is expected in the venv
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(len(ordered) * percentile / 100.0))
        return ordered[index]


def build_view(config, ledger, rule_version=None):
    """Aggregate committed ledger rows into the cached dashboard payload."""
    requests = ledger.all_committed_requests()
    services = {service_id: {"instances": {}, "outcomes": {o: 0 for o in OUTCOMES},
                             "total": 0}
                for service_id in config.services}
    for row in requests:
        service = services.get(row["service_id"])
        if service is None:
            continue
        service["total"] += 1
        service["outcomes"][row["final_outcome"]] += 1
        instance = service["instances"].setdefault(
            row["instance_id"], {"outcomes": {o: 0 for o in OUTCOMES},
                                 "latencies_ms": [], "total": 0}
        )
        instance["total"] += 1
        instance["outcomes"][row["final_outcome"]] += 1
        instance["latencies_ms"].append(row["total_elapsed_ms"])

    for service in services.values():
        for instance in service["instances"].values():
            instance["p95_elapsed_ms"] = _numpy_percentile(
                instance.pop("latencies_ms"), 95
            )

    breaker_rows = ledger.breaker_state_rows(include_stale=False)
    receipts = []
    for row in ledger.current_receipts():
        if row["rule_version"] != config.version:
            continue
        receipts.append(json.loads(row["canonical_bytes"].decode("utf-8")))

    return {
        "rule_version": rule_version if rule_version is not None else config.version,
        "services": services,
        "breaker_states": breaker_rows,
        "receipts": receipts,
        "cursor_sequence": ledger.cursor_sequence(),
    }


def render_headless(view):
    lines = []
    lines.append(f"== MeshBench view (rule version v{view['rule_version']}, "
                 f"cursor={view['cursor_sequence']}) ==")
    for service_id in sorted(view["services"]):
        service = view["services"][service_id]
        lines.append(f"service {service_id}: total={service['total']} "
                     f"outcomes={json.dumps(service['outcomes'], sort_keys=True)}")
        for instance_id in sorted(service["instances"]):
            inst = service["instances"][instance_id]
            lines.append(
                f"  - {instance_id}: total={inst['total']} p95={inst['p95_elapsed_ms']}ms "
                f"outcomes={json.dumps(inst['outcomes'], sort_keys=True)}"
            )
    lines.append("breaker states:")
    for row in view["breaker_states"]:
        lines.append(
            f"  - {row['service_id']}/{row['instance_id']} {row['state']} "
            f"fails={row['consecutive_failures']} stale={row['stale']}"
        )
    lines.append("error-budget receipts:")
    for receipt in view["receipts"]:
        lines.append(
            f"  - {receipt['window_id']} counters={json.dumps(receipt['counters'], sort_keys=True)} "
            f"avail={receipt['availability_sli_scaled']} "
            f"remaining={receipt['remaining_scaled']} "
            f"insufficient={receipt['insufficient_sample']} "
            f"exhausted={receipt['budget_exhausted']}"
        )
    return "\n".join(lines)


def launch_tk_dashboard(config, ledger):  # pragma: no cover - GUI path
    """Lazily start the Tkinter dashboard; raise a clear error without Tcl/Tk."""
    try:
        import importlib
        importlib.import_module("tkinter")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Tkinter/Tcl-Tk unavailable on this Python; use headless view instead"
        ) from exc
    from .tk_dashboard import DashboardApp
    app = DashboardApp(config, ledger)
    app.mainloop()
    return app
