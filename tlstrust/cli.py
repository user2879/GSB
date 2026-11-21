"""Command-line interface for the TLS trust-rotation workbench."""

from __future__ import annotations

import argparse
import json
import ssl
import sys
from pathlib import Path
from typing import Any

from .certfamily import generate_family, load_family
from .demo import run_demo
from .errors import E_HANDSHAKE, WorkbenchError
from .eventlog import JsonlEventLog
from .faults import run_injection
from .handshake import capture_real_handshake, load_input_snapshot, write_input_snapshot
from .ledger import Ledger
from .trust import demo_revisions, load_trust_document, write_trust_document
from .verify import compute_flips, validate_many
from .workspace import run_validation


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def _check_ssl() -> None:
    if not ssl.OPENSSL_VERSION:
        raise WorkbenchError(E_HANDSHAKE, "Python ssl has no OpenSSL backend")


def _result_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "revision": result["revision"],
            "conclusion": result["conclusion"],
            "failure_category": result["failure_category"],
            "first_failure": result.get("first_failure"),
            "index": result["index"],
            "result_digest": result["result_digest"],
        }
        for result in sorted(results, key=lambda item: item["revision"])
    ]


def cmd_generate(args: argparse.Namespace) -> int:
    family = generate_family(
        args.out,
        roots=args.roots,
        intermediate_levels=args.intermediate_levels,
        leaves=args.leaves,
        validity_days=args.validity_days,
        port=args.port,
        seed=args.seed,
        key_bits=args.key_bits,
        **({"not_before_ms": args.not_before_ms} if args.not_before_ms is not None else {}),
    )
    trust_path = None
    if args.trust_out:
        write_trust_document(args.trust_out, demo_revisions(family))
        load_trust_document(args.trust_out, family)
        trust_path = str(Path(args.trust_out))
    _print_json(
        {
            "ok": True,
            "family": str(family.root / "family.json"),
            "roots": [item["fingerprint"] for item in family.data["roots"]],
            "leaves": [item["id"] for item in family.data["leaves"]],
            "trust_document": trust_path,
        }
    )
    return 0


def cmd_handshake(args: argparse.Namespace) -> int:
    _check_ssl()
    family = load_family(args.family)
    snapshot = capture_real_handshake(
        family,
        leaf_id=args.leaf,
        port=args.port,
        inject_chain_fingerprint=args.inject_chain,
    )
    if args.snapshot_out:
        write_input_snapshot(args.snapshot_out, snapshot)
    _print_json(snapshot)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    family = load_family(args.family)
    document = load_trust_document(args.trust, family)
    snapshot = load_input_snapshot(args.snapshot)
    results = validate_many(snapshot, document)
    flips = compute_flips(results)
    output: dict[str, Any] = {
        "ok": True,
        "trust_version": None,
        "revision_chain_digest": document.chain_digest,
        "snapshot_digest": snapshot["snapshot_digest"],
        "results": _result_rows(results),
        "flips": flips,
    }
    if args.ledger:
        ledger = Ledger(args.ledger)
        try:
            run = run_validation(
                ledger,
                document,
                snapshot,
                source_path=str(args.trust),
            )
            session_file = ledger.create_session_snapshot(
                run["run_id"],
                args.session_dir or (Path(args.ledger).parent / "sessions"),
            )
            output.update(
                {
                    "trust_version": run["trust_version"],
                    "run_id": run["run_id"],
                    "results": _result_rows(run["results"]),
                    "flips": run["flips"],
                    "session_snapshot": {
                        "format_version": session_file["payload"]["format_version"],
                        "content_digest": session_file["payload"]["content_digest"],
                        "path": session_file["path"],
                    },
                }
            )
        finally:
            ledger.close()
    _print_json(output)
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    try:
        result = ledger.rollback_to_revision(args.target)
        view = ledger.build_current_view()
        _print_json(
            {
                "ok": True,
                "rollback": result,
                "current_view": {
                    "trust_version": view["trust_version"],
                    "view_digest": view["view_digest"],
                    "run_ids": [run["run_id"] for run in view["runs"]],
                },
            }
        )
    finally:
        ledger.close()
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    try:
        source = args.session
        if source is None:
            snapshots = ledger.list_session_snapshots(current_only=True)
            if not snapshots:
                raise WorkbenchError(E_HANDSHAKE, "no current session snapshot is available", field_path="session")
            source = snapshots[-1]["content_digest"]
        payload = ledger.load_session_snapshot(source)
        family = load_family(args.family) if args.family else None
        _print_json(ledger.replay_session(payload, family))
    finally:
        ledger.close()
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    try:
        _print_json(ledger.compare_indexes(args.left_index, args.right_index))
    finally:
        ledger.close()
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    _check_ssl()
    summary = run_demo(
        args.workdir,
        fresh=not args.reuse,
        key_bits=args.key_bits,
        port=args.port,
        roots=args.roots,
        intermediate_levels=args.intermediate_levels,
        leaves=args.leaves,
        validity_days=args.validity_days,
        seed=args.seed,
        not_before_ms=args.not_before_ms,
    )
    _print_json(summary)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import run_ui

    run_ui(args.workdir)
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    _check_ssl()
    result = run_injection(
        args.kind,
        args.workdir,
        fresh=not args.reuse,
        key_bits=args.key_bits,
        port=args.port,
    )
    _print_json(result)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tlstrust",
        description="Local Windows-oriented TLS trust-anchor rotation and chain-validation workbench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate a deterministic local certificate family")
    generate.add_argument("--out", required=True)
    generate.add_argument("--roots", type=int, default=2)
    generate.add_argument("--intermediate-levels", type=int, default=2)
    generate.add_argument("--leaves", type=int, default=1)
    generate.add_argument("--validity-days", type=int, default=90)
    generate.add_argument("--port", type=int, default=0)
    generate.add_argument("--seed", default="tlstrust-deterministic-v1")
    generate.add_argument("--key-bits", type=int, default=2048)
    generate.add_argument("--not-before-ms", type=int)
    generate.add_argument("--trust-out")
    generate.set_defaults(func=cmd_generate)

    handshake = sub.add_parser("handshake", help="perform a real loopback TLS handshake and capture its wire chain")
    handshake.add_argument("--family", required=True)
    handshake.add_argument("--leaf")
    handshake.add_argument("--port", type=int, default=0)
    handshake.add_argument("--snapshot-out")
    handshake.add_argument("--inject-chain", action="store_true")
    handshake.set_defaults(func=cmd_handshake)

    validate = sub.add_parser("validate", help="validate a captured chain under every revision")
    validate.add_argument("--family", required=True)
    validate.add_argument("--trust", required=True)
    validate.add_argument("--snapshot", required=True)
    validate.add_argument("--ledger")
    validate.add_argument("--session-dir")
    validate.set_defaults(func=cmd_validate)

    rollback = sub.add_parser("rollback", help="atomically roll a ledger to a historical revision")
    rollback.add_argument("--ledger", required=True)
    rollback.add_argument("--target", type=int, required=True)
    rollback.set_defaults(func=cmd_rollback)

    replay = sub.add_parser("replay", help="load a session snapshot and verify byte-for-byte replay")
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--session")
    replay.add_argument("--family")
    replay.set_defaults(func=cmd_replay)

    compare = sub.add_parser("compare", help="compare two persisted judgments by stable index")
    compare.add_argument("--ledger", required=True)
    compare.add_argument("--left-index", required=True)
    compare.add_argument("--right-index", required=True)
    compare.set_defaults(func=cmd_compare)

    demo = sub.add_parser("demo", help="run the three deterministic demo scenarios")
    demo.add_argument("--workdir", default="data/demo")
    demo.add_argument("--reuse", action="store_true", help="reuse an existing marked workbench directory")
    demo.add_argument("--key-bits", type=int, default=2048)
    demo.add_argument("--port", type=int, default=0)
    demo.add_argument("--roots", type=int, default=2)
    demo.add_argument("--intermediate-levels", type=int, default=2)
    demo.add_argument("--leaves", type=int, default=1)
    demo.add_argument("--validity-days", type=int, default=90)
    demo.add_argument("--seed", default="tlstrust-deterministic-v1")
    demo.add_argument("--not-before-ms", type=int)
    demo.set_defaults(func=cmd_demo)

    ui = sub.add_parser("ui", help="open the Tkinter diagnostic view")
    ui.add_argument("--workdir", default="data/demo")
    ui.set_defaults(func=cmd_ui)

    inject = sub.add_parser("inject", help="inject chain, revocation or snapshot corruption")
    inject.add_argument("--kind", choices=("chain", "revoke", "snapshot"), required=True)
    inject.add_argument("--workdir", default="data/inject")
    inject.add_argument("--reuse", action="store_true")
    inject.add_argument("--key-bits", type=int, default=2048)
    inject.add_argument("--port", type=int, default=0)
    inject.set_defaults(func=cmd_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log: JsonlEventLog | None = None
    workdir = getattr(args, "workdir", None)
    if workdir:
        log = JsonlEventLog(Path(workdir) / "logs" / "events.jsonl")
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        if log is not None:
            try:
                log.error("cli.error", exc)
            except OSError:
                pass
        _print_json(exc.to_dict())
        return 2
    except (ValueError, OSError) as exc:
        error = WorkbenchError("E_INPUT", str(exc), field_path="$")
        if log is not None:
            try:
                log.error("cli.error", error)
            except OSError:
                pass
        _print_json(error.to_dict())
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
