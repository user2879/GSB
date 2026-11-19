from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from joinlab.anomaly import (
    inject_skew,
    inject_snapshot_missing_version,
    inject_type_mismatch,
)
from joinlab.canonical import digest_json
from joinlab.errors import JoinLabError
from joinlab.logging_utils import StructuredLogger, atomic_write_text
from joinlab.params import AnalysisParams, load_params_file, normalise_params
from joinlab.session import WorkbenchSession
from joinlab.snapshot import load_snapshot
from joinlab.spec import load_workspace


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "samples" / "base_tables.json"
DEFAULT_SESSION = ROOT / "samples" / "session.json"
DEFAULT_DB = ROOT / "artifacts" / "joinlab.sqlite"
DEFAULT_SNAPSHOT = ROOT / "artifacts" / "snapshot.json"
DEFAULT_STATE = ROOT / "artifacts" / "version_state.json"
DEFAULT_LOG = ROOT / "artifacts" / "logs" / "joinlab.jsonl"


def _logger(path: str | None) -> StructuredLogger:
    return StructuredLogger(path)


def _params(args: argparse.Namespace, workspace_defaults: dict[str, Any], k_default: int) -> AnalysisParams:
    session_values = load_params_file(args.session) if args.session else {}
    merged = dict(workspace_defaults)
    merged.update(session_values)
    overrides = {
        "bucket_count": getattr(args, "bucket_count", None),
        "sample_size": getattr(args, "sample_size", None),
        "seed": getattr(args, "seed", None),
        "quantile_policy": getattr(args, "quantile_policy", None),
        "attribution_k": getattr(args, "attribution_k", None),
    }
    merged.update({key: value for key, value in overrides.items() if value is not None})
    return normalise_params(merged, None, k_default)


def _session(args: argparse.Namespace) -> WorkbenchSession:
    spec = load_workspace(args.workspace)
    params = _params(args, spec.default_params, spec.attribution_k)
    return WorkbenchSession(
        spec=spec,
        db_path=args.db,
        params=params,
        logger=_logger(args.log),
        version_state_path=args.state,
    )


def _report_digest(report: dict[str, Any]) -> str:
    return digest_json(
        {
            "histograms": {name: value["digest"] for name, value in report["histograms"].items()},
            "samples": {name: value["sample_digest"] for name, value in report["samples"].items()},
            "estimate": report["estimate"],
            "truth": report["truth"]["output_sequence_digest"],
            "validation": report["validation"],
        }
    )


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def command_run(args: argparse.Namespace, save: bool = True) -> dict[str, Any]:
    session = _session(args)
    report = session.build_current()
    snapshot_result = None
    if save or args.snapshot:
        snapshot_result = session.save_snapshot(args.snapshot)
    result = {
        "ok": True,
        "command": "run",
        "report_digest": _report_digest(report),
        "params_version": report["params_version"],
        "summary": {
            "left_rows": report["tables"][report["join"]["left_table"]]["filtered_row_count"],
            "right_rows": report["tables"][report["join"]["right_table"]]["filtered_row_count"],
            "histogram_estimate": report["estimate"]["histogram_path"]["estimate_rows"],
            "sample_estimate": report["estimate"]["sample_path"]["estimate_rows"],
            "true_rows": report["truth"]["true_row_count"],
            "validation": report["validation"]["status"],
            "top_attribution": report["attribution"],
        },
        "snapshot": snapshot_result,
        "report": report,
    }
    if args.output:
        atomic_write_text(Path(args.output), json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return result


def command_restore(args: argparse.Namespace) -> dict[str, Any]:
    session = _session(args)
    restored = session.restore_snapshot(args.snapshot)
    result = {
        "ok": True,
        "command": "restore",
        "replay": restored["replay"],
        "replacement": restored["replacement"],
        "report_digest": _report_digest(restored["report"]),
        "report": restored["report"],
    }
    if args.output:
        atomic_write_text(Path(args.output), json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return result


def _capture_error(
    kind: str,
    artifact: Path,
    action,
    logger: StructuredLogger,
) -> dict[str, Any]:
    try:
        action()
    except JoinLabError as exc:
        logger.error(exc.code, exc.message, field_path=exc.field_path, details=exc.details, injection=kind)
        return {
            "ok": True,
            "command": "inject",
            "injection": kind,
            "artifact": str(artifact),
            "observed_error": exc.to_dict(),
        }
    return {
        "ok": False,
        "command": "inject",
        "injection": kind,
        "artifact": str(artifact),
        "message": "预期的业务错误未发生",
    }


def command_inject(args: argparse.Namespace) -> dict[str, Any]:
    logger = _logger(args.log)
    if args.kind == "skew":
        artifact = inject_skew(args.workspace, args.out)
        return _capture_error(args.kind, artifact, lambda: load_workspace(artifact), logger)
    if args.kind == "type":
        artifact = inject_type_mismatch(args.workspace, args.out)
        return _capture_error(args.kind, artifact, lambda: load_workspace(artifact), logger)
    if args.kind == "snapshot_version_missing":
        artifact = inject_snapshot_missing_version(args.snapshot, Path(args.out) / "snapshot_missing_version.json")
        return _capture_error(args.kind, artifact, lambda: load_snapshot(artifact), logger)
    raise ValueError(args.kind)


def command_verify(args: argparse.Namespace) -> dict[str, Any]:
    logger = _logger(args.log)
    injected_root = ROOT / "artifacts" / "injected"
    first = _session(args)
    report_a = first.build_current()
    snapshot_a = first.save_snapshot(args.snapshot)
    digest_a = _report_digest(report_a)

    second = _session(args)
    report_b = second.build_current()
    digest_b = _report_digest(report_b)
    deterministic = {
        "ok": digest_a == digest_b,
        "first_digest": digest_a,
        "second_digest": digest_b,
    }

    changed = AnalysisParams(
        bucket_count=first.params.bucket_count + 3,
        sample_size=first.params.sample_size + 32,
        seed=first.params.seed + 1,
        quantile_policy=first.params.quantile_policy,
        attribution_k=first.params.attribution_k,
    )
    replacement_result = first.replace_parameters(changed)

    restore_session = _session(args)
    restore_session.build_current()
    restored = restore_session.restore_snapshot(args.snapshot)

    skew_artifact = inject_skew(args.workspace, injected_root / "skew")
    skew_error = _capture_error("skew", skew_artifact, lambda: load_workspace(skew_artifact), logger)
    type_artifact = inject_type_mismatch(args.workspace, injected_root / "type")
    type_error = _capture_error("type", type_artifact, lambda: load_workspace(type_artifact), logger)
    missing_snapshot = inject_snapshot_missing_version(
        args.snapshot,
        injected_root / "snapshot" / "snapshot_missing_version.json",
    )
    snapshot_error = _capture_error(
        "snapshot_version_missing",
        missing_snapshot,
        lambda: load_snapshot(missing_snapshot),
        logger,
    )
    tampered_document = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    first_histogram = next(iter(tampered_document.get("histograms", {})), None)
    if first_histogram is not None:
        tampered_document["histograms"][first_histogram]["digest"] = "tampered-for-replay-check"
    tampered_snapshot = injected_root / "snapshot" / "snapshot_tampered.json"
    atomic_write_text(
        tampered_snapshot,
        json.dumps(tampered_document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    replay_session = _session(args)
    replay_session.build_current()
    replay_error = _capture_error(
        "snapshot_tamper",
        tampered_snapshot,
        lambda: replay_session.restore_snapshot(tampered_snapshot),
        logger,
    )

    codes = {
        "skew": skew_error.get("observed_error", {}).get("code"),
        "type": type_error.get("observed_error", {}).get("code"),
        "snapshot": snapshot_error.get("observed_error", {}).get("code"),
        "replay": replay_error.get("observed_error", {}).get("code"),
    }
    verification = {
        "ok": bool(
            deterministic["ok"]
            and codes == {
                "skew": "E_SKEW",
                "type": "E_TYPE",
                "snapshot": "E_SNAP",
                "replay": "E_REPLAY",
            }
        ),
        "command": "verify",
        "determinism": deterministic,
        "baseline_summary": {
            "params_version": report_a["params_version"],
            "histogram_estimate": report_a["estimate"]["histogram_path"]["estimate_rows"],
            "sample_estimate": report_a["estimate"]["sample_path"]["estimate_rows"],
            "true_rows": report_a["truth"]["true_row_count"],
            "validation": report_a["validation"]["status"],
            "top_attribution": report_a["attribution"],
        },
        "snapshot": snapshot_a,
        "parameter_replacement": replacement_result["replacement"],
        "snapshot_restore": {
            "verified": restored["replay"]["verified"],
            "checks": restored["replay"]["checks"],
            "replacement": restored["replacement"],
            "report_digest": _report_digest(restored["report"]),
        },
        "anomalies": {
            "skew": skew_error,
            "type": type_error,
            "snapshot": snapshot_error,
            "replay": replay_error,
        },
        "observed_codes": codes,
    }
    output_path = ROOT / "artifacts" / "demo_report.json"
    atomic_write_text(output_path, json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    verification["output_path"] = str(output_path)
    return verification


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--session", default=str(DEFAULT_SESSION))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--output")


def _add_param_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bucket-count", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quantile-policy", choices=("nearest_rank", "lower_rank"))
    parser.add_argument("--attribution-k", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="等值联接规模估计与采样诊断工作台")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="装载数据、构建统计并执行真值联接")
    _add_common(run_parser)
    _add_param_overrides(run_parser)

    restore_parser = subparsers.add_parser("restore", help="从快照整体恢复并逐项重算校验")
    _add_common(restore_parser)
    _add_param_overrides(restore_parser)

    inject_parser = subparsers.add_parser("inject", help="生成异常输入并验证业务码")
    _add_common(inject_parser)
    inject_parser.add_argument("--kind", required=True, choices=("skew", "type", "snapshot_version_missing"))
    inject_parser.add_argument("--out", default=str(ROOT / "artifacts" / "injected"))

    verify_parser = subparsers.add_parser("verify", help="执行完整本地验收并写出 artifacts/demo_report.json")
    _add_common(verify_parser)
    _add_param_overrides(verify_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        if args.command == "run":
            result = command_run(args)
        elif args.command == "restore":
            result = command_restore(args)
        elif args.command == "inject":
            result = command_inject(args)
        elif args.command == "verify":
            result = command_verify(args)
        else:
            parser.error("unknown command")
            return 2
        _print_json(result)
        if args.command == "verify" and not result["ok"]:
            return 2
        if args.command == "inject" and not result["ok"]:
            return 2
        return 0
    except JoinLabError as exc:
        logger = _logger(getattr(args, "log", None))
        logger.error(exc.code, exc.message, field_path=exc.field_path, details=exc.details)
        _print_json(exc.to_dict())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
