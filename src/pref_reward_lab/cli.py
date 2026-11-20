from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .canonical import read_json
from .engine import ExperimentLab
from .errors import LabError, SimulatedTermination
from .sample import generate_dataset
from .snapshot import SnapshotStore
from .verify import run_verification


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    invalidation = report["invalidation"]
    return {
        "ok": True,
        "deterministic_core_fingerprint": report["deterministic_core_fingerprint"],
        "records": len(report["records"]),
        "boundary_samples": len(report["boundary_samples"]),
        "selected_tier": report["strategy"]["selected_tier"],
        "effective_hash_buckets": report["strategy"]["effective_hash_buckets"],
        "precision_tier": report["strategy"]["precision_tier"],
        "failure_business_codes": report["failure_business_codes"],
        "injection_business_codes": report["injection_business_codes"],
        "invalidation": {
            "reused_counts": invalidation["reused_counts"],
            "recomputed_counts": invalidation["recomputed_counts"],
            "metrics_recomputed": invalidation["metrics_recomputed"],
            "changed_pair_count": len(invalidation["changed_pair_ids"]),
            "deleted_pair_count": len(invalidation["deleted_pair_ids"]),
            "added_pair_count": len(invalidation["added_pair_ids"]),
            "affected_bucket_count": len(invalidation["affected_bucket_ids"]),
        },
        "snapshot": report["snapshot"],
        "report_path": report["conversion"]["artifact_paths"].get("report_path", "report.json"),
    }


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="local sample directory")
    parser.add_argument("--config", help="config JSON (default: DATA/config.json)")
    parser.add_argument("--state", help="snapshot/cache directory (default: DATA/.lab_state)")
    parser.add_argument("--out", help="report/artifact directory (default: DATA/output)")
    parser.add_argument("--injections", help="injection JSON (default: DATA/injections.json)")
    parser.add_argument("--source-version", choices=("v1", "v2"), help="override config source version")
    parser.add_argument("--resume", action="store_true", help="resume only from a fully committed snapshot")
    parser.add_argument("--expect-version", type=int, help="expected global snapshot version for resume")
    parser.add_argument(
        "--expect-source-version",
        choices=("v1", "v2"),
        help="expected source version for resume",
    )
    parser.add_argument("--force-recompute", action="store_true", help="ignore reusable stage cache")
    parser.add_argument(
        "--crash-after-stage",
        choices=("features", "scoring", "conversion", "cache"),
        help="write an incomplete temp artifact and terminate before manifest replacement",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pref-reward-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a fixed-seed local sample")
    generate.add_argument("--out", required=True)
    generate.add_argument("--pairs", type=int, default=120)
    generate.add_argument("--seed", type=int, default=20260101)
    generate.add_argument("--no-invalid-rows", action="store_true")

    run = subparsers.add_parser("run", help="run one incremental round")
    _add_common_run_arguments(run)

    verify = subparsers.add_parser("verify", help="run edit-sequence and reproducibility verification")
    verify.add_argument("--work-root", required=True)
    verify.add_argument("--pairs", type=int, default=120)
    verify.add_argument("--seed", type=int, default=20260101)

    edit = subparsers.add_parser("edit", help="apply JSON edit operations to a sample directory")
    edit.add_argument("--data", required=True)
    edit.add_argument("--config")
    edit.add_argument("--edit-file", required=True, help="JSON array of edit operations")
    edit.add_argument("--state")
    edit.add_argument("--out")
    edit.add_argument("--run-each", action="store_true")

    inspect = subparsers.add_parser("inspect-snapshot", help="inspect or validate the latest snapshot")
    inspect.add_argument("--state", required=True)
    inspect.add_argument("--config", help="validate against this config hash when supplied")
    inspect.add_argument("--expect-version", type=int)
    return parser


def _lab_from_args(args: argparse.Namespace) -> ExperimentLab:
    return ExperimentLab(
        args.data,
        config_path=args.config,
        state_dir=args.state,
        output_dir=args.out,
        injections_path=args.injections,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            paths = generate_dataset(
                args.out,
                pair_count=args.pairs,
                seed=args.seed,
                include_invalid_rows=not args.no_invalid_rows,
            )
            print(json.dumps({"ok": True, "paths": paths}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            lab = _lab_from_args(args)
            report = lab.run(
                resume=args.resume,
                force_recompute=args.force_recompute,
                crash_after_stage=args.crash_after_stage,
                source_version_override=args.source_version,
                expected_global_version=args.expect_version,
                expected_source_version=args.expect_source_version,
            )
            summary = _summary(report)
            summary["report_path"] = str(Path(args.out or Path(args.data) / "output") / "report.json")
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "verify":
            report = run_verification(args.work_root, pair_count=args.pairs, seed=args.seed)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "all_incremental_full_matches": report["all_incremental_full_matches"],
                        "concurrency_serial_parallel_match": report[
                            "concurrency_serial_parallel_match"
                        ],
                        "conversion_roundtrip": report["conversion_roundtrip"],
                        "crash_simulation": report["crash_simulation"],
                        "verification_fingerprint": report["verification_fingerprint"],
                        "report_path": str(Path(args.work_root) / "verification_report.json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "edit":
            lab = ExperimentLab(
                args.data,
                config_path=args.config,
                state_dir=args.state,
                output_dir=args.out,
            )
            raw_edits = read_json(args.edit_file)
            if not isinstance(raw_edits, list):
                raise ValueError("edit file must contain a JSON array")
            results: list[dict[str, Any]] = []
            for edit in raw_edits:
                if not isinstance(edit, dict):
                    raise ValueError("each edit must be an object")
                lab.apply_edit(edit)
                if args.run_each:
                    report = lab.run()
                    results.append(
                        {
                            "edit": edit,
                            "core_fingerprint": report["deterministic_core_fingerprint"],
                            "snapshot_version": report["snapshot"]["global_version"],
                        }
                    )
                else:
                    results.append({"edit": edit, "applied": True})
            print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "inspect-snapshot":
            store = SnapshotStore(args.state)
            if args.config:
                from .canonical import fingerprint
                from .config import LabConfig

                config = LabConfig.load(args.config)
                report = store.recover(
                    config_hash=fingerprint(config.to_dict(), prefix="config:"),
                    expected_global_version=args.expect_version,
                )
            else:
                manifest = store.latest_manifest()
                report = {"status": "empty", "manifest": None} if manifest is None else {
                    "status": "manifest_present",
                    "manifest": manifest,
                }
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except SimulatedTermination as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "E_SIMULATED_TERMINATION",
                        "message": str(exc),
                        "notice": "manifest was not replaced; resume will use the previous complete generation",
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 130
    except LabError as exc:
        print(json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False, indent=2))
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {"code": "E_CLI", "message": f"{type(exc).__name__}: {exc}"},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    parser.error("unknown command")
    return 2
