"""Command line entry point and deterministic demo / fault-injection driver."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from .engine import ViewConclusion
from .errors import WorkbenchError
from .fixedpoint import FixedPoint
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .session import Session, key_conclusion_digest
from .statistics_view import session_distribution_report
from .spec import load_json, load_specification

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_data"
OUTPUT_DIR = ROOT / "output"


def _inject_spec_corruption(env_doc: dict) -> dict:
    """Duplicate an observation seq -> first E_SPEC violation."""
    doc = copy.deepcopy(env_doc)
    doc["environments"][0]["observations"]["timeout_ms"][1]["seq"] = 1
    return doc


def _inject_grid_violation(env_doc: dict) -> dict:
    """Valid type, out of supported scale grid -> E_GRID."""
    doc = copy.deepcopy(env_doc)
    doc["scale_exponent"] = 42
    return doc


def _inject_version_mismatch(snapshot_path: Path) -> Path:
    """Tamper a persisted snapshot's environment version -> E_VERSION."""
    with snapshot_path.open("r", encoding="utf-8") as fh:
        snapshot = json.load(fh)
    snapshot["environment_version"] = "tampered-" + snapshot["environment_version"][:16]
    tampered_path = snapshot_path.with_name("snapshot_tampered.json")
    with tampered_path.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
    return tampered_path


def _write_temp(doc: dict, name: str) -> Path:
    path = OUTPUT_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return path


def _render_conclusion(vc: ViewConclusion) -> dict:
    return {
        "view_id": vc.view_id,
        "version": vc.version,
        "added_keys": vc.added_keys,
        "removed_keys": vc.removed_keys,
        "changed_keys": vc.changed_keys,
        "keys": {
            key: {
                "policy": kc.policy,
                "authority_q": kc.authority_q,
                "authority_real": kc.authority_real,
                "bound_q": kc.bound_q,
                "adjacent_diff_q": kc.adjacent_diff_q,
                "error_influenced": kc.error_influenced,
                "attribution": kc.attribution,
                "attribution_contributions": kc.attribution_contributions,
                "accepted_obs_ids": kc.accepted_obs_ids,
                "rejected": [asdict(v) for v in kc.rejected],
                "conflict_detail": kc.conflict_detail,
            }
            for key, kc in sorted(vc.keys.items())
        },
    }


def run_demo(args: argparse.Namespace) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(OUTPUT_DIR / "events.jsonl")
    env_path = SAMPLE_DIR / "environments.json"
    view_path = SAMPLE_DIR / "views.json"
    ledger_path = OUTPUT_DIR / "arbitration_ledger.sqlite"
    snapshot_path = OUTPUT_DIR / "session_snapshot.json"

    try:
        env_doc = load_json(env_path)
        if args.inject == "spec_corruption":
            env_doc = _inject_spec_corruption(env_doc)
            env_path = _write_temp(env_doc, "env_corrupted.json")
            logger.log("ERROR", "fault_injected", injection="spec_corruption",
                       target=str(env_path))
        if args.inject == "grid_violation":
            env_doc = _inject_grid_violation(env_doc)
            env_path = _write_temp(env_doc, "env_grid_violation.json")
            logger.log("ERROR", "fault_injected", injection="grid_violation",
                       target=str(env_path))

        spec = load_specification(env_path, view_path)
        if ledger_path.exists() and args.fresh:
            ledger_path.unlink()
        ledger = Ledger(ledger_path)
        session = Session(spec, ledger)

        if args.crash_view:
            try:
                session.run_all(crash_view_id=args.crash_view)
            except RuntimeError as exc:
                logger.log("ERROR", "forced_termination", view_id=args.crash_view,
                           message=str(exc))
            # Reopen a brand new ledger handle against the same file.
            ledger.close()
            ledger = Ledger(ledger_path)
            session = Session(spec, ledger)
            unknown = session.unknown_views()
            logger.log("INFO", "recovery_detected_unknown", unknown_views=unknown)
            session.rerun_unknown()

        if args.inject == "version_mismatch":
            session.run_all()
            session.save_snapshot(snapshot_path)
            tampered = _inject_version_mismatch(snapshot_path)
            logger.log("ERROR", "fault_injected", injection="version_mismatch",
                       target=str(tampered))
            tampered_snapshot = load_json(tampered)
            # Version identification precedes digest integrity so the changed
            # environment set version is the first reported difference.
            session.verify_snapshot(tampered_snapshot, check_digest=False)

        session.run_all()
        report = {
            "scale": FixedPoint(spec.scale_exponent).describe_bounds(),
            "views": [_render_conclusion(session.conclusions[v])
                      for v in sorted(session.conclusions)],
            "distribution_stats": session_distribution_report(session),
            "unknown_views": session.unknown_views(),
        }
        report_path = OUTPUT_DIR / "arbitration_report.json"
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=False)
        snapshot_path = session.save_snapshot(snapshot_path)
        logger.log("INFO", "run_completed",
                   views=sorted(session.conclusions),
                   report=str(report_path), snapshot=str(snapshot_path))
        ledger.close()
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    except WorkbenchError as exc:
        record = logger.log("ERROR", "workbench_error", **exc.to_dict())
        print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False),
              file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arbworkbench",
        description="Cross-environment configuration arbitration workbench",
    )
    parser.add_argument(
        "--inject",
        choices=["spec_corruption", "grid_violation", "version_mismatch"],
        help="inject a deterministic fault and report its structured error code",
    )
    parser.add_argument("--crash-view", help="simulate SIGKILL after writing a view row")
    parser.add_argument("--fresh", action="store_true",
                        help="delete any pre-existing ledger before running")
    parser.add_argument("--gui", action="store_true", help="launch the Tkinter GUI")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gui:
        from .gui import launch_gui
        launch_gui(SAMPLE_DIR / "environments.json", SAMPLE_DIR / "views.json",
                   OUTPUT_DIR / "arbitration_ledger.sqlite")
        return 0
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
