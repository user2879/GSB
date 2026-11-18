#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quota_app.config_workspace import open_workspace
from quota_app.derive import resolve_effective
from quota_app.events import load_events
from quota_app.injections import run_injection
from quota_app.ledger import Ledger
from quota_app.preview import PreviewService
from quota_app.utils import utc_now
from scripts.generate_samples import generate


def verify() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    expected_counts = {"ALLOW": 6, "RATE_LIMITED": 2, "OVER_QUOTA": 2}

    with tempfile.TemporaryDirectory(prefix="quota-verification-") as temporary:
        data_dir = Path(temporary) / "data"
        generate(data_dir)
        configs = data_dir / "configs"
        events_path = data_dir / "events" / "access-events.ndjson"
        ledger_path = data_dir / "ledger.sqlite3"

        workspace = open_workspace(configs)
        active = workspace.require_active()
        checks["upgrade"] = {
            "ok": (
                active.revision == 3
                and len(workspace.upgrade.completed) == 2
                and all(path.exists() for path in (configs / "backups").glob("*.bak"))
            ),
            "active_revision": active.revision,
            "upgraded": [entry.as_dict() for entry in workspace.upgrade.completed],
            "documents": [
                {
                    "revision": document.revision,
                    "format": document.schema_version,
                    "upgraded_from": document.original_version,
                }
                for document in workspace.documents
            ],
        }

        effective = resolve_effective(active)
        checks["derivation"] = {
            "ok": (
                effective["child"].derived_capacity_bytes == 1100
                and effective["child"].capacity_bytes == 700
                and effective["child"].arbitration == "explicit"
                and effective["mirror"].derived_capacity_bytes == 1000
                and effective["mirror"].capacity_bytes == 1000
                and effective["mirror"].arbitration == "derived"
            ),
            "child": effective["child"].to_json(),
            "mirror": effective["mirror"].to_json(),
        }

        events = load_events(events_path)
        preview = PreviewService(events, budget_ms=1000)
        fast, authoritative, comparison = preview.run(active)
        counts = dict(authoritative.decision_counts)
        checks["judgment_preview"] = {
            "ok": counts == expected_counts and comparison.ok and fast.error_bound_single == 0,
            "decisions": counts,
            "event_count": len(events),
            "fast_ms": fast.elapsed_ms,
            "authoritative_ms": authoritative.elapsed_ms,
            "preview_matches_authoritative": comparison.ok,
            "single_error_upper": authoritative.error_bound_single,
            "aggregate_error_upper": authoritative.error_bound_aggregate,
        }

        with Ledger(ledger_path) as ledger:
            ledger.register_config(active)
            ingest = ledger.ingest_events(active, events, batch_size=3)
            replay = ledger.verify_replay()
            view = ledger.current_view()
            checks["ledger"] = {
                "ok": (
                    ingest.decisions == expected_counts
                    and ingest.batches_committed == 4
                    and replay.ok
                    and len(view["judgments"]) == 10
                    and view["error_bounds"] == {"single": 0, "aggregate": 0}
                ),
                "ingest": ingest.as_dict(),
                "replay": replay.as_dict(),
                "active_revision": view["active_revision"],
                "judgments": len(view["judgments"]),
            }

        crash = {"raised": False}

        def crash_hook(stage: str, batch_id: int) -> None:
            if stage == "before_commit":
                crash["raised"] = True
                raise RuntimeError("verification forced interruption")

        recovery_db = data_dir / "recovery.sqlite3"
        with Ledger(recovery_db) as ledger:
            ledger.register_config(active)
            try:
                ledger.ingest_events(active, events, batch_size=3, crash_hook=crash_hook)
            except RuntimeError:
                pass
        with Ledger(recovery_db) as ledger:
            recovery = ledger.recover_incomplete()
            rerun = ledger.ingest_events(active, events, batch_size=3)
            replay_after = ledger.verify_replay()
            checks["recovery"] = {
                "ok": (
                    crash["raised"]
                    and len(recovery.unknown_batches) == 1
                    and not recovery.exit_report_available
                    and recovery.rerun_required
                    and rerun.judgments_written == 10
                    and replay_after.ok
                ),
                "recovery": recovery.as_dict(),
                "rerun": rerun.as_dict(),
                "replay": replay_after.as_dict(),
            }

        injections = [
            run_injection(kind, config_dir=configs, events_path=events_path).as_dict()
            for kind in ("corrupt-config", "missing-checkpoint", "timestamp-regression")
        ]
        checks["injections"] = {
            "ok": all(item["passed"] for item in injections),
            "results": injections,
        }

        try:
            from quota_app.gui import launch_gui, tk

            gui_import_ok = True
            tk_available = tk is not None
        except Exception as exc:  # pragma: no cover
            gui_import_ok = False
            tk_available = False
            checks["gui_error"] = str(exc)
        checks["gui"] = {
            "ok": gui_import_ok,
            "module_import": gui_import_ok,
            "tkinter_available_on_host": tk_available,
            "note": (
                "Tkinter window launch was not executed on this headless host."
                if not tk_available
                else "Tkinter import is available; window launch requires an interactive desktop."
            ),
        }

    return {
        "generated_at_utc": utc_now(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "sqlite_version": sqlite3.sqlite_version,
        },
        "checks": checks,
        "overall_ok": all(section.get("ok", False) for section in checks.values()),
    }


def run_tests() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    summary = completed.stderr.strip().splitlines()
    return {
        "exit_code": completed.returncode,
        "ok": completed.returncode == 0,
        "summary_lines": summary[-4:],
        "stdout_tail": completed.stdout.strip().splitlines()[-5:],
        "stderr_tail": summary[-12:],
    }


def main() -> int:
    result = verify()
    result["tests"] = run_tests()
    result["overall_ok"] = bool(result["overall_ok"] and result["tests"]["ok"])
    output = PROJECT_ROOT / "artifacts" / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
