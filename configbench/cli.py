"""命令行入口：在 Windows 本地以子进程节点运行完整实验流程。

示例：
  python -m configbench.cli validate --registry samples/registry.json \
      --manifest samples/manifest.json --samples samples/samples.json
  python -m configbench.cli run --ledger workspace/bench.db
  python -m configbench.cli recover --ledger workspace/bench.db
"""


import argparse
import json
import os
import sys

from .errors import BenchError
from .manifest import load_all
from .platform import Platform


def _print(obj):
    print(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2,
                     default=str))


def _load(args):
    return load_all(args.registry, args.manifest, args.samples)


def cmd_validate(args):
    registry, manifest, samples = _load(args)
    _print({"ok": True, "nodes": sorted(registry.nodes),
            "versions": [v.version for v in manifest.versions],
            "latency_budget_ms": manifest.latency_budget_ms,
            "samples": {"seed": samples.seed, "n_samples": samples.n_samples}})
    return 0


def _build_platform(args):
    registry, manifest, samples = _load(args)
    return Platform(registry, manifest, samples, args.ledger,
                    max_global=args.max_global,
                    per_node_quota=args.per_node_quota)


def cmd_run(args):
    p = _build_platform(args)
    try:
        p.start_nodes()
        report = {"publishes": []}
        for version in [v.version for v in p.manifest.versions]:
            pub = p.publish_version(version)
            report["publishes"].append({
                "version": version,
                "affected_nodes": pub["propagation"]["affected_nodes"],
                "direct_changed": pub["propagation"]["direct_changed"],
                "success": pub["receipts"]["success"],
                "total": pub["receipts"]["total"],
                "peaks": pub["peaks"],
            })
            compare = p.compare_paths(version)
            report.setdefault("consistency", []).append(compare)
            p.query_preview(f"cli-q-v{version}")
        report["latency"] = p.latency_report()
        _print(report)
        return 0
    finally:
        p.close()


def cmd_recover(args):
    p = _build_platform(args)
    try:
        _print(p.recover())
        return 0
    finally:
        p.close()


def cmd_query(args):
    p = _build_platform(args)
    try:
        _print(p.query_preview(args.request_id))
        return 0
    finally:
        p.close()


def cmd_recompute(args):
    p = _build_platform(args)
    try:
        _print(p.authoritative_recompute(args.version))
        return 0
    finally:
        p.close()


def cmd_latency(args):
    p = _build_platform(args)
    try:
        _print(p.latency_report())
        return 0
    finally:
        p.close()


def build_parser():
    parser = argparse.ArgumentParser(prog="configbench")
    parser.add_argument("--registry", default="samples/registry.json")
    parser.add_argument("--manifest", default="samples/manifest.json")
    parser.add_argument("--samples", default="samples/samples.json")
    parser.add_argument("--ledger", default="workspace/bench.db")
    parser.add_argument("--max-global", dest="max_global", type=int, default=4)
    parser.add_argument("--per-node-quota", dest="per_node_quota", type=int,
                        default=2)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("recover").set_defaults(func=cmd_recover)
    q = sub.add_parser("query"); q.add_argument("--request-id", default=None)
    q.set_defaults(func=cmd_query)
    r = sub.add_parser("recompute"); r.add_argument("version", type=int)
    r.set_defaults(func=cmd_recompute)
    sub.add_parser("latency").set_defaults(func=cmd_latency)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BenchError as exc:
        _print(exc.to_dict())
        return 2


if __name__ == "__main__":
    sys.exit(main())
