from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json
from .errors import PipelineError, data_error
from .model import RunConfig
from .runner import replay_snapshot, resume_snapshot, run_pipeline
from .structured_log import StructuredLogger


DEFAULT_LEDGER = "var/ledger/pipeline.sqlite"
DEFAULT_SNAPSHOT_DIR = "var/snapshots"
DEFAULT_OUT_DIR = "var/runs/current"
DEFAULT_LOG = "var/logs/pipeline.jsonl"

CONFIG_KEYS = {
    "parallelism",
    "memory_budget_bytes",
    "budget_version",
    "split_threshold",
    "split_parts",
    "row_group_size",
    "ledger_batch_size",
    "split_strategy",
    "skew_safety_limit",
    "inject_skew_amplify",
    "inject_budget_shortfall",
    "inject_digest_task",
}

OVERRIDE_KEYS = {
    "parallelism": "parallelism",
    "memory_budget_bytes": "memory_budget_bytes",
    "budget_version": "budget_version",
    "split_threshold": "split_threshold",
    "split_parts": "split_parts",
    "row_group_size": "row_group_size",
    "ledger_batch_size": "ledger_batch_size",
    "split_strategy": "split_strategy",
    "skew_safety_limit": "skew_safety_limit",
    "inject_skew_amplify": "inject_skew_amplify",
    "inject_budget_shortfall": "inject_budget_shortfall",
    "inject_digest_task": "inject_digest_task",
}


def _load_config_document(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise data_error(f"cannot read configuration file: {error}", str(config_path)) from error
    except json.JSONDecodeError as error:
        raise data_error(
            f"invalid configuration JSON: {error.msg}",
            f"{config_path}:{error.lineno}:{error.colno}",
        ) from error
    if not isinstance(value, Mapping):
        raise data_error("configuration root must be a JSON object", str(config_path))
    unknown = sorted(set(value) - CONFIG_KEYS)
    if unknown:
        raise data_error(
            f"unknown configuration key: {unknown[0]}",
            f"{config_path}.{unknown[0]}",
            details={"unknown_keys": unknown},
        )
    return dict(value)


def build_config(config_path: str | None, namespace: argparse.Namespace) -> RunConfig:
    payload: dict[str, Any] = {}
    if config_path is not None:
        payload.update(_load_config_document(config_path))
    for argument_name, config_name in OVERRIDE_KEYS.items():
        value = getattr(namespace, argument_name, None)
        if value is not None:
            payload[config_name] = value
    config = RunConfig.from_json_dict(payload)
    config.validate()
    return config


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON run-parameter file")
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--memory-budget-bytes", type=int)
    parser.add_argument("--budget-version", type=int)
    parser.add_argument("--split-threshold", type=int)
    parser.add_argument("--split-parts", type=int)
    parser.add_argument("--row-group-size", type=int)
    parser.add_argument("--ledger-batch-size", type=int)
    parser.add_argument("--split-strategy")
    parser.add_argument("--skew-safety-limit", type=int)
    parser.add_argument("--inject-skew-amplify", type=int)
    parser.add_argument("--inject-budget-shortfall", action="store_true", default=None)
    parser.add_argument("--inject-digest-task")


def _add_common_outputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--log-file", default=DEFAULT_LOG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="columnar-pipeline",
        description="Deterministic columnar aggregation with skew splits, budgets, ledger and replay",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute a query")
    run_parser.add_argument("--data", required=True, help="local table/query JSON file")
    run_parser.add_argument("--run-id", help="explicit run id (defaults to a deterministic fingerprint)")
    _add_config_options(run_parser)
    _add_common_outputs(run_parser)

    replay_parser = subparsers.add_parser("replay", help="replay every recorded shard and compare the ledger")
    replay_parser.add_argument("--snapshot", required=True)
    replay_parser.add_argument("--data")
    replay_parser.add_argument("--ledger", default=DEFAULT_LEDGER)
    replay_parser.add_argument("--out-dir")
    replay_parser.add_argument("--log-file", default=DEFAULT_LOG)

    resume_parser = subparsers.add_parser("resume", help="open a snapshot, replay it or replace all state")
    resume_parser.add_argument("--snapshot", required=True)
    resume_parser.add_argument("--data")
    resume_parser.add_argument("--config", help="optional replacement parameter file")
    _add_common_outputs(resume_parser)
    return parser


def _print_json(value: Mapping[str, Any]) -> None:
    sys.stdout.write(canonical_json(dict(value)) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="strict")
        except Exception:
            pass
    parser = build_parser()
    namespace = parser.parse_args(argv)
    try:
        if namespace.command == "run":
            config = build_config(namespace.config, namespace)
            summary = run_pipeline(
                namespace.data,
                config,
                out_dir=namespace.out_dir,
                ledger_path=namespace.ledger,
                snapshot_dir=namespace.snapshot_dir,
                log_path=namespace.log_file,
                run_id=namespace.run_id,
            )
        elif namespace.command == "replay":
            summary = replay_snapshot(
                namespace.snapshot,
                data_path=namespace.data,
                ledger_path=namespace.ledger,
                out_dir=namespace.out_dir,
                log_path=namespace.log_file,
            )
        elif namespace.command == "resume":
            config_override = (
                build_config(namespace.config, namespace)
                if namespace.config is not None
                else None
            )
            summary = resume_snapshot(
                namespace.snapshot,
                data_path=namespace.data,
                ledger_path=namespace.ledger,
                snapshot_dir=namespace.snapshot_dir,
                out_dir=namespace.out_dir,
                log_path=namespace.log_file,
                config_override=config_override,
            )
        else:
            raise AssertionError(namespace.command)
        public_summary = {key: value for key, value in summary.items() if key != "result_json"}
        _print_json(public_summary)
        return 0
    except PipelineError as error:
        if not getattr(error, "_logged", False):
            StructuredLogger(getattr(namespace, "log_file", None)).error(error)
        _print_json(error.as_dict())
        return error.exit_code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
