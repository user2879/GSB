"""Command line interface for tier_mover."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .engine import Engine
from .errors import BUSINESS_EXIT_CODE, TierMoverError, VersionConflictError
from .faults import Injection
from .ledger import Ledger
from .logging_util import JsonlLogger
from .manifest import load_manifest
from .replay import replay
from .samplegen import build_sample
from .storage import staging_dir


def _print(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def _injection(args: argparse.Namespace) -> Injection:
    kind: Optional[str] = args.inject
    capacity_override = args.capacity_override if kind == "capacity" else None
    policy_override = args.policy_version_override if kind == "version" else None
    tier_override = args.tier_param_version_override if kind == "version" else None
    return Injection(
        kind=kind,
        target_key=args.target_key,
        capacity_override=capacity_override,
        policy_version_override=policy_override,
        tier_param_version_override=tier_override,
        kill_after_commit=args.kill_after_commit,
    )


def _apply_version_override(manifest, injection: Injection):
    if injection.kind != "version":
        return manifest
    from dataclasses import replace

    return replace(
        manifest,
        policy_version=injection.policy_version_override or manifest.policy_version,
        tier_param_version=injection.tier_param_version_override
        or manifest.tier_param_version,
    )


def _apply_truncate(workdir: Path, manifest, target_key: Optional[str]) -> None:
    key = target_key or manifest.objects[0].key
    obj = manifest.object_by_key(key)
    path = staging_dir(workdir) / obj.digest
    data = path.read_bytes()
    truncated = max(1, len(data) // 2)
    path.write_bytes(data[:truncated])


def cmd_sample(args: argparse.Namespace) -> int:
    result = build_sample(
        args.workdir,
        policy_version=args.policy_version,
        tier_param_version=args.tier_param_version,
        hot_capacity=args.hot_capacity,
        cold_capacity=args.cold_capacity,
        hot_watermark=args.hot_watermark,
        cold_reclaim=args.cold_reclaim,
        batch_size=args.batch_size,
    )
    _print({"status": "ok", **{k: v for k, v in result.items() if k != "objects"},
            "object_count": len(result["objects"])})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    log_path = workdir / "logs" / f"{args.phase}.jsonl"
    with JsonlLogger(log_path, echo=args.echo) as logger:
        manifest = load_manifest(args.manifest)
        injection = _injection(args)
        manifest = _apply_version_override(manifest, injection)
        if injection.kind == "truncate":
            _apply_truncate(workdir, manifest, injection.target_key)
            logger.event("injection_applied", kind="truncate",
                         target_key=injection.target_key or manifest.objects[0].key)
        if injection.active:
            logger.event("injection_configured", kind=injection.kind,
                         target_key=injection.target_key)
        engine = Engine(workdir, manifest, logger, injection)
        if args.phase == "initial":
            conclusions = engine.run_initial()
        else:
            conclusions = engine.run_cycle()
        out = workdir / f"{args.phase}_conclusions.json"
        out.write_text(json.dumps(conclusions, sort_keys=True, indent=2) + "\n",
                       encoding="utf-8")
        _print({"status": "ok", "conclusions_path": str(out),
                "migrations": len(conclusions["migrations"]),
                "total_move_cost": conclusions["total_move_cost"]})
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    with JsonlLogger(workdir / "logs" / "recover.jsonl", echo=args.echo) as logger:
        manifest = load_manifest(args.manifest)
        report = Engine(workdir, manifest, logger).recover()
        _print({"status": "ok", **report})
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    with JsonlLogger(workdir / "logs" / "replay.jsonl", echo=args.echo) as logger:
        manifest = load_manifest(args.manifest)
        result = replay(workdir, manifest)
        logger.event("replay_ok", migrations=result["migrations_replayed"])
        _print(result)
    return 0


def cmd_replace(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    with JsonlLogger(workdir / "logs" / "replace.jsonl", echo=args.echo) as logger:
        manifest = load_manifest(args.manifest)
        injection = _injection(args)
        manifest = _apply_version_override(manifest, injection)
        ledger = Ledger.open(workdir)
        try:
            ledger.check_versions(manifest)
            _print({"status": "unchanged", "reason": "versions already match"})
            return 0
        except VersionConflictError as exc:
            if injection.kind != "version" and not args.force:
                logger.event("version_conflict", **exc.context)
                raise
            result = ledger.replace_snapshot(manifest)
        finally:
            ledger.close()
        logger.event("snapshot_replaced", archived=result["archived_snapshot"])
        _print({"status": "replaced", **result})
    return 0


def cmd_conclusions(args: argparse.Namespace) -> int:
    ledger = Ledger.open(args.workdir)
    try:
        _print(ledger.export_conclusions())
    finally:
        ledger.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tier_mover",
                                     description="Deterministic hot/cold tiering engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="generate deterministic sample data")
    p_sample.add_argument("workdir")
    p_sample.add_argument("--policy-version", default="policy-v1")
    p_sample.add_argument("--tier-param-version", default="tier-v1")
    p_sample.add_argument("--hot-capacity", type=int, default=16384)
    p_sample.add_argument("--cold-capacity", type=int, default=65536)
    p_sample.add_argument("--hot-watermark", type=int, default=12288)
    p_sample.add_argument("--cold-reclaim", type=int, default=32768)
    p_sample.add_argument("--batch-size", type=int, default=4)
    p_sample.set_defaults(func=cmd_sample)

    def add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("workdir")
        p.add_argument("--manifest", required=True)
        p.add_argument("--inject", choices=["capacity", "truncate", "version", "kill"])
        p.add_argument("--target-key")
        p.add_argument("--capacity-override", type=int, default=1)
        p.add_argument("--policy-version-override", default="policy-INJECTED")
        p.add_argument("--tier-param-version-override", default="tier-INJECTED")
        p.add_argument("--kill-after-commit", type=int, default=1)
        p.add_argument("--echo", action="store_true")

    p_initial = sub.add_parser("initial", help="first-session initial placement")
    add_run_args(p_initial)
    p_initial.set_defaults(func=cmd_run)
    p_initial.set_defaults(phase="initial")

    p_cycle = sub.add_parser("cycle", help="derive and execute a migration cycle")
    add_run_args(p_cycle)
    p_cycle.set_defaults(func=cmd_run)
    p_cycle.set_defaults(phase="cycle")

    p_recover = sub.add_parser("recover", help="recover after an interruption")
    p_recover.add_argument("workdir")
    p_recover.add_argument("--manifest", required=True)
    p_recover.add_argument("--echo", action="store_true")
    p_recover.set_defaults(func=cmd_recover)

    p_replay = sub.add_parser("replay", help="deterministic replay validation")
    p_replay.add_argument("workdir")
    p_replay.add_argument("--manifest", required=True)
    p_replay.add_argument("--echo", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    p_replace = sub.add_parser("replace", help="version check + whole-snapshot replacement")
    p_replace.add_argument("workdir")
    p_replace.add_argument("--manifest", required=True)
    p_replace.add_argument("--force", action="store_true")
    p_replace.add_argument("--inject", choices=["version"])
    p_replace.add_argument("--target-key")
    p_replace.add_argument("--policy-version-override", default="policy-INJECTED")
    p_replace.add_argument("--tier-param-version-override", default="tier-INJECTED")
    p_replace.add_argument("--kill-after-commit", type=int, default=1)
    p_replace.add_argument("--echo", action="store_true")
    p_replace.set_defaults(func=cmd_replace)

    p_conc = sub.add_parser("conclusions", help="export deterministic ledger conclusions")
    p_conc.add_argument("workdir")
    p_conc.set_defaults(func=cmd_conclusions)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TierMoverError as exc:
        payload = exc.to_dict()
        print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False),
              file=sys.stderr)
        return BUSINESS_EXIT_CODE
    except FileNotFoundError as exc:
        print(json.dumps({"code": "E_IO", "message": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return BUSINESS_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
