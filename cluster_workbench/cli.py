from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .commands import load_commands
from .errors import issue_dicts
from .snapshot import SnapshotStore
from .util import atomic_write_text, canonical_json, to_jsonable
from .workbench import Workbench


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cluster_workbench",
        description="Deterministic interactive clustering and density workbench",
    )
    parser.add_argument("--config", required=True, help="experiment YAML configuration")
    parser.add_argument("--data", required=True, help="deterministic synthetic data description")
    parser.add_argument("--commands", required=True, help="interactive command stream YAML")
    parser.add_argument("--injections", help="optional fault-injection manifest YAML")
    parser.add_argument("--workspace", default="workbench_state", help="cache and snapshot workspace")
    parser.add_argument("--report", default="diagnostic_report.json", help="final diagnostic JSON path")
    parser.add_argument("--responses", help="optional interactive response log JSON path")
    parser.add_argument("--skip-rebuild-check", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands, command_issues = load_commands(args.commands)
    workbench = Workbench.from_files(
        args.config,
        args.data,
        injections_path=args.injections,
        workspace=args.workspace,
    )
    workbench.base_issues.extend(command_issues)
    responses: list[dict[str, Any]] = []
    for command in commands:
        responses.append(workbench.submit_command(command))
    workbench.close()
    final_report = workbench.latest_report
    if final_report is None:
        raise RuntimeError("workbench produced no final report")

    equivalence: dict[str, Any]
    if args.skip_rebuild_check:
        equivalence = {"checked": False, "equal": None}
    else:
        with tempfile.TemporaryDirectory(prefix="cluster-workbench-rebuild-") as temporary:
            full_report = workbench.full_rebuild_report(Path(temporary) / "cache")
        canonical_incremental = canonical_report_fields(final_report)
        canonical_full = canonical_report_fields(full_report)
        equal = canonical_json(canonical_incremental) == canonical_json(canonical_full)
        equivalence = {
            "checked": True,
            "equal": bool(equal),
            "canonical_fields": sorted(canonical_incremental.keys()),
            "incremental_fingerprint": __import__("hashlib").sha256(
                canonical_json(canonical_incremental).encode("utf-8")
            ).hexdigest(),
            "full_rebuild_fingerprint": __import__("hashlib").sha256(
                canonical_json(canonical_full).encode("utf-8")
            ).hexdigest(),
        }
        if not equal:
            equivalence["first_difference"] = first_difference(canonical_incremental, canonical_full)

    diagnostic = dict(final_report)
    diagnostic["equivalence_check"] = equivalence
    diagnostic["interaction_responses"] = responses
    snapshot_payload, snapshot_issues = SnapshotStore(
        Path(args.workspace) / "snapshots"
    ).recover()
    diagnostic["snapshot_check"] = {
        "recovered": snapshot_payload is not None,
        "snapshot_id": (
            snapshot_payload.get("manifest", {}).get("snapshot_id")
            if snapshot_payload is not None
            else None
        ),
        "report_matches_final_state": bool(
            snapshot_payload is not None
            and snapshot_payload.get("report", {}).get("state_fingerprint")
            == final_report.get("state_fingerprint")
        ),
        "issues": issue_dicts(snapshot_issues),
    }
    diagnostic["execution_environment"] = {
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "pyyaml": yaml.__version__,
        "platform_note": "portable Python/Numpy/PyYAML implementation; no container dependency",
    }
    report_path = Path(args.report)
    atomic_write_text(report_path, json.dumps(to_jsonable(diagnostic), ensure_ascii=False, indent=2))
    if args.responses:
        atomic_write_text(
            args.responses,
            json.dumps(to_jsonable(responses), ensure_ascii=False, indent=2),
        )
    if not args.quiet:
        summary = {
            "report": str(report_path),
            "command_count": len(commands),
            "candidate_count": final_report.get("counts", {}).get("candidates", 0),
            "failure_count": len(final_report.get("failures", [])),
            "warning_count": len(final_report.get("warnings", [])),
            "best_candidate": final_report.get("best_candidate", {}).get("id") if final_report.get("best_candidate") else None,
            "stale_responses": sum(1 for item in responses if item.get("stale")),
            "equivalence": equivalence,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if equivalence.get("equal") in {None, True} else 1


def canonical_report_fields(report: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for candidate in report.get("candidates", []):
        candidates.append(
            {
                "id": candidate.get("id"),
                "method": candidate.get("method"),
                "params": candidate.get("params"),
                "feature_columns": candidate.get("feature_columns"),
                "train_fraction": candidate.get("train_fraction"),
                "status": candidate.get("status"),
                "metrics": candidate.get("metrics"),
                "metrics_comparable": candidate.get("metrics_comparable"),
                "noncomparable_reasons": candidate.get("noncomparable_reasons"),
                "warnings": candidate.get("warnings"),
                "errors": candidate.get("errors"),
                "labels": candidate.get("labels"),
                "cluster": candidate.get("cluster"),
                "preprocess": candidate.get("preprocess"),
                "cache_keys": candidate.get("cache_keys"),
                "rank": candidate.get("rank"),
            }
        )
    budget = report.get("budget", {})
    return {
        "schema_version": report.get("schema_version"),
        "state_fingerprint": report.get("state_fingerprint"),
        "data": report.get("data"),
        "candidates": candidates,
        "best_candidate": report.get("best_candidate"),
        "budget": {
            "entries": budget.get("entries"),
            "consumed_total": budget.get("consumed_total"),
            "remaining": budget.get("remaining"),
            "content_fingerprint": budget.get("content_fingerprint"),
        },
        "failures": report.get("failures"),
        "warnings": report.get("warnings"),
        "injections": report.get("injections"),
        "diagnostics": {
            "cache_key": report.get("diagnostics", {}).get("cache_key"),
            "content_fingerprint": report.get("diagnostics", {}).get("content_fingerprint"),
        },
        "counts": report.get("counts"),
    }


def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any]:
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return {"path": f"{path}.{key}", "left": left.get(key), "right": right.get(key)}
            if canonical_json(left[key]) != canonical_json(right[key]):
                return first_difference(left[key], right[key], f"{path}.{key}")
        return {"path": path, "left": left, "right": right}
    if isinstance(left, list):
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            if canonical_json(left_item) != canonical_json(right_item):
                return first_difference(left_item, right_item, f"{path}[{index}]")
        if len(left) != len(right):
            return {"path": f"{path}.length", "left": len(left), "right": len(right)}
        return {"path": path, "left": left, "right": right}
    return {"path": path, "left": left, "right": right}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
