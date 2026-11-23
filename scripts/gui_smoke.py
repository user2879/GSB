"""Render the Tk workbench briefly and emit a machine-readable smoke result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvcc_workbench.gui import WorkbenchApp, _capture_tk_window  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(ROOT / "data/demo_large.json"))
    parser.add_argument("--ledger", default=str(ROOT / "artifacts/gui_ledger.sqlite3"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/gui_smoke.json"))
    parser.add_argument("--screenshot")
    parser.add_argument("--wait-ms", type=int, default=1800)
    args = parser.parse_args()

    app = WorkbenchApp(spec_path=args.spec, ledger_path=args.ledger)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def finish() -> None:
        payload = {
            "ok": app.round is not None and app.status_var.get() == "业务码: OK",
            "code": "OK" if app.round is not None else "E_GUI_SMOKE",
            "status": app.status_var.get(),
            "spec": args.spec,
            "ledger": args.ledger,
            "round_id": app.round.round_id if app.round else None,
            "selected_path": app.round.plan.selected_path if app.round else None,
            "query_count": len(app.query_tree.get_children()),
            "version_chain_root_count": len(app.chain_tree.get_children()),
            "schedule_trace_count": len(app.round.schedule.traces) if app.round else 0,
            "budget_constraints_ok": (
                app.round.schedule.budget["constraints_ok"] if app.round else False
            ),
            "observed_peaks": app.round.schedule.observed_peaks if app.round else None,
        }
        if args.screenshot:
            try:
                _capture_tk_window(
                    app,
                    args.screenshot,
                    (
                        app.winfo_rootx(),
                        app.winfo_rooty(),
                        app.winfo_width(),
                        app.winfo_height(),
                    ),
                )
                payload["screenshot"] = args.screenshot
            except Exception as exc:
                payload["screenshot_error"] = str(exc)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        app.destroy()

    app.after(args.wait_ms, finish)
    app.mainloop()
    return 0 if output.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
