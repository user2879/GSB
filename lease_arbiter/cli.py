from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, canonical_json, digest
from .engine import run_arbitration
from .errors import ArbiterError, DivergenceError
from .recovery import replay_against_ledger
from .runtime import run_live
from .spec import load_spec, validate_files
from .verifier import verify_spec


EXIT_CODES = {
    "E_SPEC": 2,
    "E_DIVERGE": 3,
    "E_REPLAY": 3,
    "E_PORT": 4,
    "E_TIMEOUT": 4,
    "E_PROTOCOL": 4,
    "E_LEASE": 5,
    "E_STALE": 5,
    "E_LEDGER": 5,
}


def _emit(payload: dict[str, Any], output: str | None = None) -> None:
    text = canonical_json(payload) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def _write_projection(projection: dict[str, Any], output: str | None) -> None:
    if not output:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(projection) + b"\n")


def _load_spec(args: argparse.Namespace):
    return load_spec(args.nodes, args.leases, args.events)


def cmd_validate(args: argparse.Namespace) -> int:
    spec = validate_files(args.nodes, args.leases, args.events)
    _emit(
        {
            "ok": True,
            "code": "OK",
            "declaration_version": spec.declaration_version,
            "spec_digest": spec.digest,
            "global_concurrency": spec.global_concurrency,
            "node_count": len(spec.nodes),
            "lease_count": len(spec.leases),
            "event_count": len(spec.events),
            "files": {
                "nodes": spec.nodes_file,
                "leases": spec.leases_file,
                "events": spec.events_file,
            },
        },
        args.out,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = _load_spec(args)
    result = run_live(
        spec,
        ledger_path=args.ledger,
        requested_path=args.path,
        fresh=args.fresh,
        crash_after_event=args.crash_after_event,
    )
    report = {"ok": True, "command": "run", **result}
    _emit(report, args.out)
    _write_projection(result["comparison_projection"], args.projection_out)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    spec = _load_spec(args)
    report = verify_spec(spec)
    report["command"] = "verify"
    _emit(report, args.out)
    _write_projection(report["comparison"]["sequential_projection"], args.projection_out)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    spec = _load_spec(args)
    report = replay_against_ledger(spec, args.ledger)
    report["command"] = "replay"
    _emit(report, args.out)
    return 0


def _first_projection_difference(left: Any, right: Any, path: str = "$") -> tuple[str, Any, Any] | None:
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{path}.{key}", left.get(key), right.get(key)
            difference = _first_projection_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_projection_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return path, left, right
    return None


def cmd_compare(args: argparse.Namespace) -> int:
    left_doc = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right_doc = json.loads(Path(args.right).read_text(encoding="utf-8"))
    left = left_doc.get("comparison_projection", left_doc)
    right = right_doc.get("comparison_projection", right_doc)
    left_bytes = canonical_bytes(left)
    right_bytes = canonical_bytes(right)
    if left_bytes != right_bytes:
        difference = _first_projection_difference(left, right)
        path, left_value, right_value = difference or ("$", None, None)
        event_seq = None
        if path.startswith("$.events["):
            index_text = path.split("[", 1)[1].split("]", 1)[0]
            try:
                index = int(index_text)
                event_seq = left["events"][index]["seq"]
            except (ValueError, IndexError, KeyError):
                event_seq = None
        raise DivergenceError(event_seq, path, left_value, right_value, "result files are not byte-identical")
    report = {
        "ok": True,
        "command": "compare",
        "byte_identical": True,
        "bytes": len(left_bytes),
        "projection_sha256": digest(left),
    }
    _emit(report, args.out)
    return 0


def cmd_node(args: argparse.Namespace) -> int:
    from .node_server import run_node

    return run_node(args.node_id, args.host, args.port, args.read_timeout_ms)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lease-arbiter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate declarations and event stream")
    validate.add_argument("--nodes", required=True)
    validate.add_argument("--leases", required=True)
    validate.add_argument("--events", required=True)
    validate.add_argument("--out")
    validate.set_defaults(func=cmd_validate)

    run = subparsers.add_parser("run", help="run a live multi-process arbitration session")
    run.add_argument("--nodes", required=True)
    run.add_argument("--leases", required=True)
    run.add_argument("--events", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--path", choices=("auto", "sequential", "batch"), default="auto")
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--crash-after-event", type=int)
    run.add_argument("--out")
    run.add_argument("--projection-out")
    run.set_defaults(func=cmd_run)

    verify = subparsers.add_parser("verify", help="independently run and byte-compare both arbitration paths")
    verify.add_argument("--nodes", required=True)
    verify.add_argument("--leases", required=True)
    verify.add_argument("--events", required=True)
    verify.add_argument("--out")
    verify.add_argument("--projection-out")
    verify.set_defaults(func=cmd_verify)

    replay = subparsers.add_parser("replay", help="recover pending work and replay committed ledger events")
    replay.add_argument("--nodes", required=True)
    replay.add_argument("--leases", required=True)
    replay.add_argument("--events", required=True)
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--out")
    replay.set_defaults(func=cmd_replay)

    compare = subparsers.add_parser("compare", help="compare comparison projections from two result files")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--out")
    compare.set_defaults(func=cmd_compare)

    node = subparsers.add_parser("node", help=argparse.SUPPRESS)
    node.add_argument("--node-id", required=True)
    node.add_argument("--host", default="127.0.0.1")
    node.add_argument("--port", type=int, required=True)
    node.add_argument("--read-timeout-ms", type=int, default=1000)
    node.set_defaults(func=cmd_node)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ArbiterError as exc:
        _emit(exc.to_dict())
        return EXIT_CODES.get(exc.code, 1)
    except KeyboardInterrupt:
        _emit({"ok": False, "code": "E_INTERRUPTED", "message": "interrupted"})
        return 130
    except Exception as exc:
        _emit({"ok": False, "code": "E_INTERNAL", "message": str(exc), "type": type(exc).__name__})
        return 1
