"""Entry point: python main.py [--mode cli|ui] [--db platform.db]
[--no-nodes] [--recover] [--replay] [--corrupt-node n1]"""
from __future__ import annotations

import argparse

from sharedbuf.orchestrator import Orchestrator
from sharedbuf.platform import Platform
from sharedbuf.sample import sample_config
from sharedbuf.ui import run_cli, run_ui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "ui"], default="cli")
    parser.add_argument("--db", default="platform.db")
    parser.add_argument("--no-nodes", action="store_true",
                        help="run with the local in-process executor")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--corrupt-node", default=None)
    args = parser.parse_args()

    cfg = sample_config()
    platform = Platform(cfg, db_path=args.db)
    platform.start_reaper()
    orch = None
    try:
        if not args.no_nodes:
            corrupt = {args.corrupt_node} if args.corrupt_node else set()
            orch = Orchestrator(platform, cfg["nodes"], corrupt_nodes=corrupt)
            started = orch.start()
            print(f"nodes started: {started}")
            platform.node_runner = orch.node_runner()
        if args.recover:
            print("recover:", platform.recover())
        if args.replay:
            print("replay:", platform.replay())
        if args.mode == "ui":
            run_ui(platform)
        else:
            run_cli(platform)
    finally:
        if orch is not None:
            orch.shutdown()
        platform.close()


if __name__ == "__main__":
    main()
