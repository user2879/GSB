#!/usr/bin/env python3
"""Command line interface for the block snapshot component.

Run ``python scripts/blocksnap_cli.py --help`` for the full command list.
The CLI only depends on the Python standard library and prints JSON results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_snap import BlockSnapService, DomainError  # noqa: E402
from block_snap.samples import PROFILES, build_profile, reset_workspace  # noqa: E402

DEFAULT_WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workdata")


def print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Block storage snapshot & rollback CLI")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-samples", help="clean workspace and build the three sample volumes")
    p.add_argument("--only", choices=sorted(PROFILES), nargs="*")

    sub.add_parser("list-samples")

    p = sub.add_parser("create-volume")
    p.add_argument("name")
    p.add_argument("--block-size", type=int, required=True)
    p.add_argument("--block-count", type=int, required=True)
    p.add_argument("--seed", default="vol")

    p = sub.add_parser("inject")
    p.add_argument("volume")
    p.add_argument("--spec", required=True, help="hex spec, e.g. 0a:ff,1c:03")

    p = sub.add_parser("snapshot")
    p.add_argument("volume")
    p.add_argument("name")

    p = sub.add_parser("list")
    p.add_argument("volume")

    p = sub.add_parser("manifest")
    p.add_argument("volume")

    p = sub.add_parser("plan")
    p.add_argument("volume")
    p.add_argument("snapshot_id")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--chain-span", type=int, default=2)
    p.add_argument("--force", choices=("bitmap", "full", "chain"))

    p = sub.add_parser("rollback")
    p.add_argument("volume")
    p.add_argument("snapshot_id")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--chain-span", type=int, default=2)
    p.add_argument("--force", choices=("bitmap", "full", "chain"))
    p.add_argument("--cross-check", action="store_true")
    p.add_argument("--crash-after", type=int, default=None)

    p = sub.add_parser("recover")
    p.add_argument("volume")
    p.add_argument("--job-id")

    p = sub.add_parser("token")
    p.add_argument("volume")
    p.add_argument("snapshot_id")

    p = sub.add_parser("audit")
    p.add_argument("volume")

    p = sub.add_parser("logs")
    p.add_argument("--code")
    p.add_argument("--event")
    p.add_argument("--snapshot-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = BlockSnapService(args.workspace)
    try:
        if args.command == "init-samples":
            service.close()
            reset_workspace(args.workspace)
            service = BlockSnapService(args.workspace)
            profiles = [PROFILES[k] for k in (args.only or sorted(PROFILES))]
            for profile in profiles:
                build_profile(service, profile)
            print_json({"built": [p.name for p in profiles], "workspace": args.workspace})
        elif args.command == "list-samples":
            print_json({k: {"description": v.description, "block_size": v.block_size,
                            "block_count": v.block_count, "snapshots": list(v.snapshot_names)}
                        for k, v in PROFILES.items()})
        elif args.command == "create-volume":
            volume = service.create_volume(args.name, args.block_size, args.block_count, args.seed)
            print_json({"path": volume.path, "block_size": volume.block_size,
                        "block_count": volume.block_count})
        elif args.command == "inject":
            print_json({"changed_blocks": service.inject_changes(args.volume, args.spec)})
        elif args.command == "snapshot":
            print_json(service.create_snapshot(args.volume, args.name))
        elif args.command == "list":
            print_json(service.list_snapshots(args.volume))
        elif args.command == "manifest":
            print_json(service.manifest_view(args.volume))
        elif args.command == "plan":
            print_json(service.plan_rollback(
                args.volume, args.snapshot_id, bitmap_threshold=args.threshold,
                chain_span=args.chain_span, force_path=args.force))
        elif args.command == "rollback":
            print_json(service.rollback(
                args.volume, args.snapshot_id, bitmap_threshold=args.threshold,
                chain_span=args.chain_span, force_path=args.force,
                verify_all_paths=args.cross_check, crash_after_blocks=args.crash_after))
        elif args.command == "recover":
            print_json(service.recover(args.volume, args.job_id))
        elif args.command == "token":
            print_json(service.completion_token(args.volume, args.snapshot_id))
        elif args.command == "audit":
            print_json(service.audit(args.volume))
        elif args.command == "logs":
            print_json(service.query_logs(
                code=args.code, event=args.event, snapshot_id=args.snapshot_id))
        return 0
    except DomainError as err:
        service.manager.logger.log_error(err)
        print_json({"error": err.to_dict()})
        return 2
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
