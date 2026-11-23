"""Command-line interface for sample generation, validation and diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .engine import WorkbenchEngine
from .errors import WorkbenchError
from .loading import load_calibration, load_commands, load_config, load_weight_pack
from .report import write_html_report, write_json_report
from .samples import DEFAULT_SEED, generate_samples
from .snapshot import SnapshotStore
from .util import canonical_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-workbench",
        description="Model compression/deployment diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("generate-samples", help="create deterministic local fixtures")
    sample_parser.add_argument("--out", default="samples")
    sample_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    validate_parser = subparsers.add_parser("validate", help="validate all local inputs")
    validate_parser.add_argument("--config", default="samples/config.yaml")
    validate_parser.add_argument("--weights", default="samples/weights.jsonl")
    validate_parser.add_argument("--calibration", default="samples/calibration.jsonl")
    validate_parser.add_argument("--commands", default="samples/commands.jsonl")

    run_parser = subparsers.add_parser("run", help="run commands, emit report and dashboard")
    run_parser.add_argument("--config", default="samples/config.yaml")
    run_parser.add_argument("--weights", default="samples/weights.jsonl")
    run_parser.add_argument("--calibration", default="samples/calibration.jsonl")
    run_parser.add_argument("--commands", default="samples/commands.jsonl")
    run_parser.add_argument("--snapshot", default=".workbench_snapshot")
    run_parser.add_argument("--report", default="reports/diagnostic-report.json")
    run_parser.add_argument("--html", default="reports/dashboard.html")
    run_parser.add_argument("--expect-version", type=int, default=None)
    run_parser.add_argument("--skip-full-verify", action="store_true")
    run_parser.add_argument("--prepare-if-missing", action="store_true")
    return parser


def _load_inputs(args: argparse.Namespace):
    config = load_config(args.config)
    weights, weight_issues = load_weight_pack(args.weights)
    calibration, calibration_issues = load_calibration(args.calibration)
    commands, command_issues = load_commands(args.commands)
    return config, weights, calibration, commands, weight_issues + calibration_issues + command_issues


def _digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _check_snapshot_mismatch(
    snapshot: Path,
    config_path: str,
    weights_path: str,
    calibration_path: str,
) -> dict[str, Any] | None:
    if not (snapshot / "manifest.json").exists():
        return None
    before = _digest_tree(snapshot)
    mismatch_config = load_config(config_path)
    mismatch_config.snapshot_version += 1
    mismatch_weights, weight_issues = load_weight_pack(weights_path)
    mismatch_calibration, calibration_issues = load_calibration(calibration_path)
    error_payload: dict[str, Any] | None = None
    try:
        WorkbenchEngine(
            mismatch_config,
            mismatch_weights,
            mismatch_calibration,
            snapshot_store=SnapshotStore(snapshot),
            base_issues=weight_issues + calibration_issues,
        )
    except WorkbenchError as exc:
        error_payload = exc.to_dict()
    after = _digest_tree(snapshot)
    return {
        "detected": bool(error_payload and error_payload.get("code") == "E_SNAPSHOT_VERSION_MISMATCH"),
        "code": error_payload.get("code") if error_payload else None,
        "snapshot_unchanged": before == after,
        "snapshot_digest": before,
    }


def _validation_report(
    *,
    responses: list[dict[str, Any]],
    incremental_equals_full: bool | None,
    snapshot_resume_equals_full: bool | None,
    snapshot_mismatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "all_commands_within_budget": all(
            bool(response.get("within_budget")) for response in responses
        ),
        "command_count": len(responses),
        "stale_response_count": sum(
            1 for response in responses if response.get("status") == "stale"
        ),
        "point_budget_pending_count": sum(
            1
            for response in responses
            if "E_POINT_BUDGET_PENDING" in response.get("codes", [])
        ),
        "incremental_equals_full": incremental_equals_full,
        "snapshot_resume_equals_full": snapshot_resume_equals_full,
        "snapshot_mismatch": snapshot_mismatch,
    }


def _run(args: argparse.Namespace) -> int:
    if args.prepare_if_missing and not all(
        Path(path).exists()
        for path in (args.config, args.weights, args.calibration, args.commands)
    ):
        generate_samples(Path(args.config).parent if Path(args.config).parent else ".")
    config, weights, calibration, commands, base_issues = _load_inputs(args)
    responses: list[dict[str, Any]] = []
    store = SnapshotStore(args.snapshot)
    engine = WorkbenchEngine(
        config,
        weights,
        calibration,
        snapshot_store=store,
        expected_version=args.expect_version,
        base_issues=base_issues,
        command_base_dir=Path(args.commands).parent,
    )
    try:
        for command in commands:
            summary = engine.process_command(command)
            responses.append(
                {
                    **summary,
                    "op": command.op,
                    "layer": command.layer,
                    "line_number": command.line_number,
                }
            )
            # The interactive measurement ends above. Persisting the configured
            # interval snapshot is deliberately outside the response budget.
            engine.maybe_commit()
        engine.drain_all()
        incremental = engine.result_payload()
        diagnostics_before_verify = engine.diagnostics()
        engine.commit_snapshot()

        incremental_equals_full: bool | None = None
        snapshot_resume_equals_full: bool | None = None
        if not args.skip_full_verify:
            full = engine.full_rebuild()
            incremental_equals_full = canonical_json(incremental) == canonical_json(full)

            config2 = load_config(args.config)
            weights2, weight_issues2 = load_weight_pack(args.weights)
            calibration2, calibration_issues2 = load_calibration(args.calibration)
            resumed = WorkbenchEngine(
                config2,
                weights2,
                calibration2,
                snapshot_store=SnapshotStore(args.snapshot),
                expected_version=args.expect_version,
                base_issues=weight_issues2 + calibration_issues2,
                command_base_dir=Path(args.commands).parent,
            )
            try:
                resumed_result = resumed.result_payload()
                snapshot_resume_equals_full = canonical_json(full) == canonical_json(
                    resumed_result
                )
            finally:
                resumed.close()

        snapshot_mismatch = _check_snapshot_mismatch(
            Path(args.snapshot), args.config, args.weights, args.calibration
        )
        report = engine.diagnostic_report()
        report["diagnostics_before_full_verify"] = diagnostics_before_verify
        report["command_responses"] = responses
        report["validation"] = _validation_report(
            responses=responses,
            incremental_equals_full=incremental_equals_full,
            snapshot_resume_equals_full=snapshot_resume_equals_full,
            snapshot_mismatch=snapshot_mismatch,
        )
        write_json_report(args.report, report)
        write_html_report(args.html, report)
        final_summary = engine.summary()
        print(
            json.dumps(
                {
                    "status": final_summary["status"],
                    "completed_layers": final_summary["completed_layers"],
                    "total_layers": final_summary["total_layers"],
                    "max_abs_error": final_summary["max_abs_error"],
                    "mse": final_summary["mse"],
                    "stale_responses": report["validation"]["stale_response_count"],
                    "point_budget_pending_responses": report["validation"][
                        "point_budget_pending_count"
                    ],
                    "all_commands_within_budget": report["validation"][
                        "all_commands_within_budget"
                    ],
                    "incremental_equals_full": incremental_equals_full,
                    "snapshot_resume_equals_full": snapshot_resume_equals_full,
                    "snapshot_version_mismatch_detected": bool(
                        snapshot_mismatch and snapshot_mismatch.get("detected")
                    ),
                    "snapshot_unchanged_after_mismatch": bool(
                        snapshot_mismatch and snapshot_mismatch.get("snapshot_unchanged")
                    ),
                    "report": str(Path(args.report)),
                    "dashboard": str(Path(args.html)),
                    "snapshot": str(Path(args.snapshot)),
                },
                indent=2,
            )
        )
        return 0
    finally:
        engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-samples":
        paths = generate_samples(args.out, seed=args.seed)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0
    if args.command == "validate":
        try:
            config, weights, calibration, commands, issues = _load_inputs(args)
        except WorkbenchError as exc:
            print(json.dumps({"ok": False, "error": exc.to_dict()}, indent=2))
            return 2
        payload = {
            "ok": not issues,
            "layers": sorted(weights),
            "layer_count": len(weights),
            "calibration_samples": len(calibration.samples),
            "commands": len(commands),
            "config": config.to_dict(),
            "issues": [issue.to_dict() for issue in issues],
        }
        print(json.dumps(payload, indent=2))
        return 0 if not issues else 1
    if args.command == "run":
        try:
            return _run(args)
        except WorkbenchError as exc:
            print(json.dumps({"ok": False, "error": exc.to_dict()}, indent=2))
            return 3
    parser.error("unknown command")
    return 2
