"""Command-line entry point for the local workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .constants import SEED, VERSION
from .errors import BusinessError
from .generator import generate_assets
from .ledger import LedgerStore
from .pipeline import (
    build_carriers,
    evaluate_workspace,
    preview_workspace,
    recover_workspace,
)


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))


def _workspace(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalworkbench", description="Deterministic cross-format codec quality diagnostic workbench")
    parser.add_argument("--version", action="version", version=VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic 10s audio/video fixtures and truth")
    generate.add_argument("--workspace", required=True)
    generate.add_argument("--seed", type=int, default=SEED)

    carriers = subparsers.add_parser("carriers", help="decode fixture and build the three carrier samples")
    carriers.add_argument("--workspace", required=True)

    evaluate = subparsers.add_parser("evaluate", help="parse carriers, decode, compute metrics, and commit a ledger")
    evaluate.add_argument("--workspace", required=True)
    evaluate.add_argument("--inject-file", type=Path)
    evaluate.add_argument("--expect-revision", type=int)
    evaluate.add_argument("--crash-point", choices=("after_meta", "before_manifest", "after_manifest"))

    recompute = subparsers.add_parser("recompute", help="apply a mutation and execute the exact invalidation closure")
    recompute.add_argument("--workspace", required=True)
    recompute.add_argument("--mutation-file", type=Path, required=True)
    recompute.add_argument("--inject-file", type=Path)
    recompute.add_argument("--expect-revision", type=int)
    recompute.add_argument("--crash-point", choices=("after_meta", "before_manifest", "after_manifest"))

    recover = subparsers.add_parser("recover", help="recover the newest committed ledger and reclaim uncommitted products")
    recover.add_argument("--workspace", required=True)

    preview = subparsers.add_parser("preview", help="decode one encoded frame to a PPM image for playback inspection")
    preview.add_argument("--workspace", required=True)
    preview.add_argument("--frame", type=int, default=0)
    preview.add_argument("--output", type=Path, required=True)

    status = subparsers.add_parser("status", help="show the newest committed ledger and normal-exit marker state")
    status.add_argument("--workspace", required=True)

    demo = subparsers.add_parser("demo", help="generate, build carriers, and run the baseline evaluation")
    demo.add_argument("--workspace", required=True)
    demo.add_argument("--seed", type=int, default=SEED)
    demo.add_argument("--reuse-assets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_assets(_workspace(args.workspace), seed=args.seed)
            _print({"ok": True, **result})
            return 0
        if args.command == "carriers":
            result = build_carriers(_workspace(args.workspace))
            _print({"ok": True, "carriers": result})
            return 0
        if args.command == "evaluate":
            result = evaluate_workspace(
                _workspace(args.workspace),
                injection_path=args.inject_file,
                crash_point=args.crash_point,
                expected_revision=args.expect_revision,
            )
            _print({"ok": result["status"] == "ok", "result": result})
            return 0 if result["status"] == "ok" else 2
        if args.command == "recompute":
            result = evaluate_workspace(
                _workspace(args.workspace),
                injection_path=args.inject_file,
                mutation_path=args.mutation_file,
                crash_point=args.crash_point,
                expected_revision=args.expect_revision,
            )
            _print({"ok": result["status"] == "ok", "result": result})
            return 0 if result["status"] == "ok" else 2
        if args.command == "recover":
            result = recover_workspace(_workspace(args.workspace))
            _print({"ok": True, "recovery": result})
            return 0
        if args.command == "preview":
            result = preview_workspace(_workspace(args.workspace), args.frame, args.output)
            _print({"ok": True, "preview": result})
            return 0
        if args.command == "status":
            store = LedgerStore(_workspace(args.workspace))
            latest = store.load_latest_committed()
            if latest is None:
                _print({"ok": False, "error": BusinessError("E_RECOVERY", "no committed ledger").to_dict()})
                return 2
            _print(
                {
                    "ok": True,
                    "revision": latest["manifest"]["revision"],
                    "run_id": latest["manifest"]["run_id"],
                    "manifest_hash": latest["manifest_hash"],
                    "closed": latest["closed"],
                    "manifest": latest["manifest"],
                    "run_summary": latest["shards"].get("run_summary.json"),
                }
            )
            return 0
        if args.command == "demo":
            workspace = _workspace(args.workspace)
            generation = None
            if not args.reuse_assets:
                generation = generate_assets(workspace, seed=args.seed)
            carriers_result = build_carriers(workspace)
            evaluation = evaluate_workspace(workspace)
            ok = evaluation["status"] == "ok"
            _print(
                {
                    "ok": ok,
                    "generation": generation,
                    "carriers": carriers_result,
                    "evaluation": evaluation,
                }
            )
            return 0 if ok else 2
    except BusinessError as exc:
        _print({"ok": False, "error": exc.to_dict()})
        return 1
    except KeyboardInterrupt:
        _print({"ok": False, "error": {"code": "E_IO", "message": "interrupted"}})
        return 130
    _print({"ok": False, "error": {"code": "E_IO", "message": "unknown command"}})
    return 1


if __name__ == "__main__":
    sys.exit(main())
