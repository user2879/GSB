"""Command-line entry points for running, validating and editing operator chains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .edits import apply_edit_files
from .engine import OperatorChainEngine, compare_execution_artifacts
from .errors import BusinessError
from .util import atomic_write_json, read_json


def _load_engine(args: argparse.Namespace) -> OperatorChainEngine:
    return OperatorChainEngine.from_files(args.chain, args.input, args.config, args.workspace)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True))


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        engine = _load_engine(args)
        result = engine.run(
            resume=not args.no_resume,
            strict_resume=args.strict_resume,
            stop_after=args.stop_after,
            include_values=not args.no_values,
        )
    except BusinessError as exc:
        _print_json({"status": "failed", "failures": [exc.as_dict()]})
        return 2
    if args.out:
        atomic_write_json(args.out, result)
    _print_json(result)
    return 0 if result.get("status") == "complete" else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        engine = _load_engine(args)
        from .chain import parse_chain
        chain = parse_chain(engine.raw_chain, engine.input_spec)
    except BusinessError as exc:
        _print_json({"valid": False, "failure": exc.as_dict()})
        return 2
    _print_json(
        {
            "valid": True,
            "input": engine.input_spec.name,
            "shape": list(engine.input_spec.shape),
            "topological_order": chain.topo_order,
            "outputs": chain.outputs,
            "config_hash": engine.config.config_hash(),
        }
    )
    return 0


def _cmd_edit(args: argparse.Namespace) -> int:
    edit = json.loads(args.edit) if args.edit else None
    if edit is None and args.edit_file:
        edit = read_json(args.edit_file)
    if not isinstance(edit, dict):
        _print_json({"status": "failed", "failure": {"code": "E_CHAIN_INVALID", "message": "an edit JSON object is required"}})
        return 2
    try:
        chain, config = apply_edit_files(args.chain, args.config, edit)
    except BusinessError as exc:
        _print_json({"status": "failed", "failure": exc.as_dict()})
        return 2
    _print_json({"status": "edited", "chain_nodes": [node.get("id") for node in chain.get("nodes", [])], "config": config})
    return 0


def _cmd_compare_fusion(args: argparse.Namespace) -> int:
    base_config = read_json(args.config)
    chain = read_json(args.chain)
    input_spec = read_json(args.input)
    plain_config = dict(base_config)
    plain_config["fusion_strategy"] = "none"
    fused_config = dict(base_config)
    fused_config["fusion_strategy"] = "auto"
    plain = OperatorChainEngine(chain, input_spec, plain_config, Path(args.workspace) / "plain").run()
    fused = OperatorChainEngine(chain, input_spec, fused_config, Path(args.workspace) / "fused").run()
    comparison = compare_execution_artifacts(plain, fused)
    differing_nodes = []
    plain_by_id = {node["id"]: node for node in plain["nodes"]}
    for node in fused["nodes"]:
        other = plain_by_id.get(node["id"], {})
        if node.get("values_hash") != other.get("values_hash"):
            differing_nodes.append(node["id"])
    _print_json(
        {
            "identical_invariant_fields": comparison["equal"],
            "differing_nodes": differing_nodes,
            "fusion_pairs": fused.get("fusion", {}).get("pairs", {}),
            "plain_signature": comparison["left_signature"],
            "fused_signature": comparison["right_signature"],
        }
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    report = run_demo(args.workspace)
    _print_json(report)
    return 0 if report.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opchain", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser, *, workspace_required: bool = True) -> None:
        subparser.add_argument("--chain", required=True, help="operator-chain JSON")
        subparser.add_argument("--input", required=True, help="deterministic input tensor JSON")
        subparser.add_argument("--config", required=True, help="execution config JSON")
        subparser.add_argument("--workspace", required=workspace_required, help="workspace for caches and snapshots")

    run_parser = subparsers.add_parser("run", help="execute incrementally in a workspace")
    add_common(run_parser)
    run_parser.add_argument("--out", help="also write the full result JSON here")
    run_parser.add_argument("--no-values", action="store_true", help="omit nested values while retaining hashes")
    run_parser.add_argument("--no-resume", action="store_true", help="disable snapshot resume checks")
    run_parser.add_argument("--strict-resume", action="store_true", help="fail with E_SNAPSHOT_STALE on mismatch")
    run_parser.add_argument("--stop-after", type=int, default=None, help="test hook: stop after N processed nodes")
    run_parser.set_defaults(func=_cmd_run)

    validate_parser = subparsers.add_parser("validate", help="validate chain topology and parameters")
    add_common(validate_parser)
    validate_parser.set_defaults(func=_cmd_validate)

    edit_parser = subparsers.add_parser("edit", help="apply one edit to chain/config JSON files")
    edit_parser.add_argument("--chain", required=True)
    edit_parser.add_argument("--config", required=True)
    edit_parser.add_argument("--edit", help="edit object as inline JSON")
    edit_parser.add_argument("--edit-file", help="edit object JSON file")
    edit_parser.set_defaults(func=_cmd_edit)

    compare_parser = subparsers.add_parser("compare-fusion", help="compare fused and unfused execution paths")
    add_common(compare_parser)
    compare_parser.set_defaults(func=_cmd_compare_fusion)

    demo_parser = subparsers.add_parser("demo", help="run the local acceptance demo")
    demo_parser.add_argument("--workspace", default=".opchain-demo")
    demo_parser.set_defaults(func=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BusinessError as exc:
        _print_json({"status": "failed", "failures": [exc.as_dict()]})
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

