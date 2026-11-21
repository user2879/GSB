"""Command line entry point for configmesh."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import __version__, injections
from .errors import BusinessError, ErrorCode
from .fixedpoint import Grid, compare_derived
from .manifest import load_bundle
from .session import Session


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str))


def _default_paths(args: argparse.Namespace) -> tuple[str, str]:
    manifest = args.manifest or os.path.join("examples", "manifest.json")
    registry = args.registry or os.path.join("examples", "registry.json")
    return manifest, registry


def cmd_run(args: argparse.Namespace) -> int:
    manifest, registry = _default_paths(args)
    session = Session(args.session, args.workdir, args.grid_exp,
                      global_limit=args.global_limit,
                      per_node_quota=args.per_node_quota)
    try:
        session.load_inputs(manifest, registry)
        timeout_map: dict[str, str] = {}
        if args.inject_timeout_task:
            for task_id in args.inject_timeout_task:
                timeout_map[task_id] = "timeout_once"
        outputs = session.run_all(timeout_map)
        for output in outputs:
            session.replay_snapshot(output["version"])
        summary = {
            "session": args.session,
            "versions": [item["version"] for item in outputs],
            "peak_concurrency": session.scheduler.peak_concurrency,
            "global_limit": args.global_limit,
            "converged": all(item["convergence"]["converged"] for item in outputs),
            "preview_consistent": all(item["preview_check"]["preview_consistent"] for item in outputs),
        }
        _print(summary)
        return 0
    finally:
        session.close()


def cmd_validate(args: argparse.Namespace) -> int:
    manifest, registry = _default_paths(args)
    registry_specs, versions = load_bundle(manifest, registry)
    _print({
        "valid": True,
        "nodes": sorted(registry_specs),
        "versions": [version.to_dict() for version in versions],
    })
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    manifest, registry = _default_paths(args)
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    log_results: list[dict[str, Any]] = []
    if args.kind == "corrupt-manifest":
        descriptor = injections.inject_corrupt_manifest(manifest)
    elif args.kind == "cycle":
        descriptor = injections.inject_derived_cycle(manifest)
    else:
        raise BusinessError(ErrorCode.E_INTERNAL, f"unknown injection {args.kind}")
    session = Session(args.session, workdir, args.grid_exp)
    try:
        try:
            session.load_inputs(manifest, registry)
        except BusinessError as exc:
            session.logger.error(exc.code.value, "injection_rejected",
                                 injection=descriptor["injection"],
                                 expected=descriptor["expected_code"],
                                 details=exc.details)
            log_results.append(exc.to_dict())
        finally:
            injections.restore(manifest)
        try:
            Grid.create(injections.illegal_grid_parameter())
        except BusinessError as exc:
            session.logger.error(exc.code.value, "illegal_grid_rejected",
                                 grid_exp=injections.illegal_grid_parameter(),
                                 details=exc.details)
            log_results.append(exc.to_dict())
        registry_specs, versions = load_bundle(manifest, registry)
        session.registry, session.versions = registry_specs, versions
        session.start_nodes()
        session.run_all()
        log_results.append({"status": "legal_input_processed_after_injection",
                            "versions": [v.version for v in versions]})
        _print({"injected": descriptor, "outcomes": log_results})
        return 0
    finally:
        session.close()


def cmd_compare(args: argparse.Namespace) -> int:
    manifest, registry = _default_paths(args)
    _registry, versions = load_bundle(manifest, registry)
    grid = Grid.create(args.grid_exp)
    from .compiler import compile_version
    compiled = compile_version(versions[-1], grid)
    if args.key_a not in compiled or args.key_b not in compiled:
        raise BusinessError(ErrorCode.E_MANIFEST, "comparison keys must exist in latest version",
                            [{"key_a": args.key_a, "key_b": args.key_b}])
    item_a = compiled[args.key_a]
    item_b = compiled[args.key_b]
    result = compare_derived(
        args.key_a, item_a.qvalue, item_a.error_bound["bound_q"],
        args.key_b, item_b.qvalue, item_b.error_bound["bound_q"],
    )
    _print(result)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    manifest, registry = _default_paths(args)
    session = Session(args.session, args.workdir, args.grid_exp,
                      global_limit=args.global_limit,
                      per_node_quota=args.per_node_quota)
    try:
        session.load_inputs(manifest, registry)
        session.start_nodes()
        session.run_all()
        checks = [session.replay_snapshot(version.version) for version in session.versions]
        _print({"replay": checks})
        return 0
    finally:
        session.close()


def cmd_preview_drift(args: argparse.Namespace) -> int:
    manifest, registry = _default_paths(args)
    session = Session(args.session, args.workdir, args.grid_exp)
    try:
        session.load_inputs(manifest, registry)
        session.start_nodes()
        first = session.versions[0]
        session.run_version(first)
        from .compiler import build_dispatch_plan, compile_version
        compiled = compile_version(first, session.grid)
        plan = build_dispatch_plan(compiled, first)
        target_node = first.targets[0]
        derived_key = next(key for key, item in first.items.items() if item.is_derived())
        session.preview.simulate_ui_drift(target_node, 0, derived_key, -999999)
        try:
            session.preview.verify_cache(first, plan, session.grid)
        except BusinessError as exc:
            session.logger.error(exc.code.value, "preview_drift_detected",
                                 details=exc.details)
            _print(exc.to_dict())
            return 0
        raise BusinessError(ErrorCode.E_INTERNAL, "expected E_PREVIEW was not raised")
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="configmesh",
                                     description="local cross-component config distribution platform")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--manifest")
    parser.add_argument("--registry")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--session", default="demo-session")
    common.add_argument("--workdir", default=os.path.join("workdir", "demo"))
    common.add_argument("--grid-exp", type=int, default=3)
    common.add_argument("--global-limit", type=int, default=3)
    common.add_argument("--per-node-quota", type=int, default=2)

    p_run = sub.add_parser("run", parents=[common], help="full pipeline")
    p_run.add_argument("--inject-timeout-task", action="append", default=[],
                       help="task id (e.g. v1-node_alpha-s0) that times out once")
    p_run.set_defaults(func=cmd_run)

    p_validate = sub.add_parser("validate", parents=[common], help="strict validation only")
    p_validate.set_defaults(func=cmd_validate)

    p_inject = sub.add_parser("inject", parents=[common], help="fault injection scenarios")
    p_inject.add_argument("--kind", choices=["corrupt-manifest", "cycle"], required=True)
    p_inject.set_defaults(func=cmd_inject)

    p_compare = sub.add_parser("compare", parents=[common], help="stable derived-item comparison")
    p_compare.add_argument("--key-a", required=True)
    p_compare.add_argument("--key-b", required=True)
    p_compare.set_defaults(func=cmd_compare)

    p_replay = sub.add_parser("replay", parents=[common], help="snapshot replay verification")
    p_replay.set_defaults(func=cmd_replay)

    p_preview = sub.add_parser("preview-drift", parents=[common],
                               help="demonstrate E_PREVIEW via UI cache corruption")
    p_preview.set_defaults(func=cmd_preview_drift)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BusinessError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
