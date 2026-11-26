from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .ledger import Ledger, load_or_replace_snapshot
from .engine import DiagnosticEngine
from .runtime import NodeCluster
from .spec import load_spec_bundle
from .workspace import Workspace, propagation_report


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = load_spec_bundle(args.root)
    print_json(
        {
            "ok": True,
            "nodes": len(bundle.nodes),
            "shards": len(bundle.shards),
            "quota_versions": [item["version"] for item in bundle.quota_versions],
            "events": len(bundle.events),
        }
    )
    return 0


def cmd_regenerate_samples(args: argparse.Namespace) -> int:
    workspace = Workspace(args.root)
    result = workspace.regenerate_sample_catalogs()
    print_json({"ok": True, **result})
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    root = Path(args.root)
    bundle = load_spec_bundle(root)
    with Ledger(args.db) as ledger:
        engine = DiagnosticEngine(root, ledger, bundle, quota_version=args.quota_version)
        readiness = engine.evaluate_readiness()
        decisions = engine.candidate_decisions(readiness)
        report = propagation_report(bundle)
        print_json(
            {
                "ok": True,
                "topology": list(engine.topology.order),
                "missing": {k: v for k, v in engine.topology.missing.items()},
                "readiness": readiness,
                "water": engine.current_water(),
                "candidate_decisions": decisions,
                "empty_propagation": report,
            }
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    bundle = load_spec_bundle(root)
    with Ledger(args.db) as ledger:
        recovery = ledger.recover_interrupted() if args.recover else {"unknown_shards": []}
        engine = DiagnosticEngine(root, ledger, bundle, quota_version=args.quota_version)
        cluster = NodeCluster(root, bundle.nodes)
        cluster.start_all()
        try:
            report = engine.run_all(
                cluster,
                batch_size=args.batch_size,
                lease_ttl_ms=args.lease_ttl_ms,
                timeout_ms=args.timeout_ms,
                max_seconds=args.max_seconds,
            )
            snapshot, replaced = load_or_replace_snapshot(root / "ledger" / "session_snapshot.json", ledger)
            print_json(
                {
                    "ok": not report["failures"],
                    "database": str(args.db),
                    "recovery": recovery,
                    "committed_now": report["committed_now"],
                    "failures": report["failures"],
                    "readiness": report["readiness"],
                    "replay": report["replay"],
                    "snapshot_replaced": replaced,
                    "snapshot_reason": snapshot.get("reason"),
                }
            )
            return 0 if not report["failures"] else 1
        finally:
            cluster.stop_all()


def cmd_replay(args: argparse.Namespace) -> int:
    with Ledger(args.db) as ledger:
        result = ledger.replay_and_verify()
        snapshot, replaced = load_or_replace_snapshot(Path(args.snapshot), ledger)
        print_json({"ok": True, "replay": result, "snapshot_replaced": replaced, "snapshot_reason": snapshot.get("reason")})
    return 0


def cmd_replace_sample(args: argparse.Namespace) -> int:
    sample = {"x": json.loads(args.x), "y": args.y}
    result = Workspace(args.root).replace_sample(args.shard_id, args.row_index, sample)
    print_json({"ok": True, "replaced": result})
    return 0


def cmd_propagate(args: argparse.Namespace) -> int:
    bundle = load_spec_bundle(args.root)
    changed_shards = args.changed_shard or []
    changed_nodes = args.changed_node or []
    report = propagation_report(bundle, changed_shard_ids=changed_shards, changed_node_ids=changed_nodes)
    print_json({"ok": True, **report})
    return 0


def _load_json_arg(json_text: str | None, file_name: str | None) -> dict[str, Any]:
    if file_name:
        return json.loads(Path(file_name).read_text(encoding="utf-8"))
    return json.loads(json_text or "{}")


def cmd_add_shard(args: argparse.Namespace) -> int:
    shard = _load_json_arg(args.json, args.file)
    result = Workspace(args.root).add_shard(shard)
    print_json({"ok": True, "added": result})
    return 0


def cmd_remove_shard(args: argparse.Namespace) -> int:
    result = Workspace(args.root).remove_shard(args.shard_id)
    print_json({"ok": True, **result})
    return 0


def cmd_add_quota_version(args: argparse.Namespace) -> int:
    version = _load_json_arg(args.json, args.file)
    result = Workspace(args.root).add_quota_version(version)
    print_json({"ok": True, "added_version": result["version"]})
    return 0


def cmd_set_scale(args: argparse.Namespace) -> int:
    workspace = Workspace(args.root)
    bundle = workspace.load()
    parent = max(item["version"] for item in bundle.quota_versions)
    version_doc: dict[str, Any] = {
        "version": args.version or parent + 1,
        "parent_version": parent,
        "session_scale": args.scale,
        "upsert_node_quotas": {},
        "delete_node_quotas": [],
        "upsert_batch_requirements": {},
        "delete_batch_requirements": [],
        "note": args.note or f"set session_scale={args.scale}",
    }
    if args.global_budget is not None:
        version_doc["global_budget_units"] = args.global_budget
    result = workspace.add_quota_version(version_doc)
    workspace.append_event("scale_changed", {"version": result["version"], "session_scale": args.scale})
    print_json({"ok": True, "added_version": result["version"], "session_scale": args.scale})
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    from .acceptance import run_acceptance

    result = run_acceptance(Path(args.root), output=Path(args.output))
    print_json(result)
    return 0 if result.get("ok") else 1


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import launch

    return launch(Path(args.root), args.db)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shard-diag", description="本地确定性推理分片下发诊断应用")
    parser.add_argument("--root", default=".", help="项目根目录（也可放在子命令后）")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS, help="项目根目录")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", parents=[common], help="校验 JSON 声明").set_defaults(func=cmd_validate)
    sub.add_parser("regenerate-samples", parents=[common], help="按固定种子重生成样本").set_defaults(func=cmd_regenerate_samples)

    plan = sub.add_parser("plan", parents=[common], help="输出就绪冻结与水位裁决")
    plan.add_argument("--db", default="ledger/diagnostic.sqlite")
    plan.add_argument("--quota-version", type=int, default=None)
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", parents=[common], help="启动节点并调度全部分片")
    run.add_argument("--db", default="ledger/diagnostic.sqlite")
    run.add_argument("--quota-version", type=int, default=None)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--lease-ttl-ms", type=int, default=30000)
    run.add_argument("--timeout-ms", type=int, default=5000)
    run.add_argument("--max-seconds", type=float, default=60.0)
    run.add_argument("--no-recover", dest="recover", action="store_false")
    run.set_defaults(func=cmd_run, recover=True)

    replay = sub.add_parser("replay", parents=[common], help="重放 SQLite 事件并校验")
    replay.add_argument("--db", default="ledger/diagnostic.sqlite")
    replay.add_argument("--snapshot", default="ledger/session_snapshot.json")
    replay.set_defaults(func=cmd_replay)

    replace = sub.add_parser("replace-sample", parents=[common], help="替换单份确定性样本")
    replace.add_argument("shard_id")
    replace.add_argument("row_index", type=int)
    replace.add_argument("x", help="JSON 整数数组，例如 '[1,2,3,4]'")
    replace.add_argument("y", type=int)
    replace.set_defaults(func=cmd_replace_sample)

    propagate = sub.add_parser("propagate", parents=[common], help="输出失效传播集合")
    propagate.add_argument("--changed-shard", action="append")
    propagate.add_argument("--changed-node", action="append")
    propagate.set_defaults(func=cmd_propagate)

    add_shard = sub.add_parser("add-shard", parents=[common], help="新增分片并强校验")
    add_shard.add_argument("--json", help="分片 JSON 对象")
    add_shard.add_argument("--file", help="包含分片 JSON 的文件")
    add_shard.set_defaults(func=cmd_add_shard)

    remove_shard = sub.add_parser("remove-shard", parents=[common], help="删除分片")
    remove_shard.add_argument("shard_id")
    remove_shard.set_defaults(func=cmd_remove_shard)

    add_quota = sub.add_parser("add-quota-version", parents=[common], help="追加配额版本")
    add_quota.add_argument("--json", help="版本 JSON 对象")
    add_quota.add_argument("--file", help="包含版本 JSON 的文件")
    add_quota.set_defaults(func=cmd_add_quota_version)

    set_scale = sub.add_parser("set-scale", parents=[common], help="通过新版本修改会话定点缩放因子")
    set_scale.add_argument("scale", type=int)
    set_scale.add_argument("--version", type=int)
    set_scale.add_argument("--global-budget", type=int)
    set_scale.add_argument("--note")
    set_scale.set_defaults(func=cmd_set_scale)

    acceptance = sub.add_parser("acceptance", parents=[common], help="运行端到端验收并写 JSON 证据")
    acceptance.add_argument("--output", default="artifacts/acceptance-result.json")
    acceptance.set_defaults(func=cmd_acceptance)

    gui = sub.add_parser("gui", parents=[common], help="启动 Tkinter 诊断界面")
    gui.add_argument("--db", default="ledger/diagnostic.sqlite")
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DiagnosticError as exc:
        print_json(exc.to_dict())
        return 2
    except FileNotFoundError as exc:
        print_json({"ok": False, "code": "E_SPEC", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
