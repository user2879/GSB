"""演示 CLI：构建、增量维护、查询、快照、恢复、重放与异常注入。

用法示例（Windows 命令行同样适用）：
    python demo.py all
    python demo.py build --out snapshots/session.json
    python demo.py update --snapshot snapshots/session.json --budget 3
    python demo.py query --snapshot snapshots/session.json --kind adjacency --start 1
    python demo.py restore --snapshot snapshots/session.json
    python demo.py replay --snapshot snapshots/session.json
    python demo.py inject corrupt-graph|tamper-index|zero-budget
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from graphidx import (
    BudgetError,
    DomainError,
    JsonLogger,
    MaintenanceSession,
    replay_snapshot,
)
from graphidx.canon import canonical

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
LOGS = ROOT / "logs"
DEFAULT_SNAPSHOT = ROOT / "snapshots" / "session.json"


def make_logger(name: str) -> JsonLogger:
    return JsonLogger(LOGS / f"{name}.jsonl", echo=False)


def print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def new_session(args) -> MaintenanceSession:
    return MaintenanceSession.from_files(
        SAMPLES / "graph.json",
        SAMPLES / "indexes.json",
        SAMPLES / "views.json",
        concurrency=getattr(args, "concurrency", 4),
        budget=getattr(args, "budget", 100),
        logger=make_logger("session"),
    )


def cmd_build(args) -> int:
    session = new_session(args)
    report = session.build_indexes()
    session.save_snapshot(args.out)
    print_json(report)
    return 0


def cmd_update(args) -> int:
    session = MaintenanceSession.from_snapshot(args.snapshot, logger=make_logger("session"))
    if args.budget is not None:
        session.budget = args.budget
    changesets = json.loads((SAMPLES / "changesets.json").read_text(encoding="utf-8"))["changesets"]
    reports = []
    for changeset in changesets:
        if args.only and changeset["id"] != args.only:
            continue
        reports.append(session.apply_changeset(changeset))
    session.save_snapshot(args.snapshot)
    print_json(reports)
    return 0


def cmd_query(args) -> int:
    session = MaintenanceSession.from_snapshot(args.snapshot, logger=make_logger("query"))
    if args.kind == "adjacency":
        result = session.query.traverse_adjacency(args.start)
    elif args.kind == "label":
        result = session.query.traverse_by_label(args.start, args.label)
    else:
        result = session.query.view_reachability(args.view)
    print_json(result)
    print("canonical:", canonical(result))
    return 0


def cmd_restore(args) -> int:
    session = new_session(args)
    session.build_indexes()
    report = session.restore(args.snapshot)
    print_json(report)
    return 0


def cmd_replay(args) -> int:
    report = replay_snapshot(str(args.snapshot), logger=make_logger("replay"))
    print_json(report)
    return 0


def cmd_inject(args) -> int:
    logger = make_logger("inject")
    injected = SAMPLES / "injected"
    injected.mkdir(exist_ok=True)
    kind = args.kind
    try:
        if kind == "corrupt-graph":
            target = injected / "corrupt_graph.json"
            shutil.copy(SAMPLES / "graph.json", target)
            with target.open("a", encoding="utf-8") as fh:
                fh.write('\n{"id": 1, "label": "Person"} /* corrupted tail */')
            MaintenanceSession.from_files(
                target, SAMPLES / "indexes.json", SAMPLES / "views.json", logger=logger
            )
        elif kind == "tamper-index":
            snapshot = injected / "tampered_snapshot.json"
            session = new_session(args)
            session.build_indexes()
            session.save_snapshot(snapshot)
            data = json.loads(snapshot.read_text(encoding="utf-8"))
            first = next(iter(sorted(data["entries"]["by_label"])))
            data["entries"]["by_label"][first]["digest"] = "sha256:" + "0" * 64
            snapshot.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            MaintenanceSession.from_snapshot(snapshot, logger=logger)
        elif kind == "zero-budget":
            session = new_session(args)
            session.budget = 0
            session.build_indexes()
            changesets = json.loads((SAMPLES / "changesets.json").read_text(encoding="utf-8"))["changesets"]
            session.apply_changeset(changesets[0])
        else:  # pragma: no cover
            raise SystemExit(f"unknown injection kind {kind}")
    except DomainError as exc:
        logger.error(exc.code, exc.message, injection=kind, details=exc.details)
        print_json(exc.to_dict())
        return 0
    print(f"injection {kind} did not trigger an error", file=sys.stderr)
    return 1


def cmd_all(args) -> int:
    snapshot = DEFAULT_SNAPSHOT
    print("== 1. 全量构建 ==")
    session = new_session(args)
    print_json(session.build_indexes())

    print("== 2. 增量变更与预算内重算（budget=4，演示排队） ==")
    session.budget = 4
    changesets = json.loads((SAMPLES / "changesets.json").read_text(encoding="utf-8"))["changesets"]
    for changeset in changesets:
        report = session.apply_changeset(changeset)
        print_json({k: report[k] for k in ("changeset", "graph_version", "affected_entries", "invalid_views", "recomputed", "skipped")})

    print("== 3. 查询（同步后确定性结果） ==")
    print_json(session.query.traverse_adjacency(1))
    print_json(session.query.traverse_by_label(4, "Person") if 4 in session.graph.nodes else session.query.traverse_by_label(1, "Person"))
    print_json(session.query.view_reachability("reach_team"))

    print("== 4. 快照持久化 ==")
    session.save_snapshot(snapshot)
    print(f"snapshot -> {snapshot}")

    print("== 5. 恢复：版本识别与整体替换 ==")
    diverged = new_session(args)
    diverged.build_indexes()
    diverged.apply_changeset({"id": "csX", "ops": [{"op": "set_edge_weight", "edge": "e2", "weight": 99}]})
    print_json(diverged.restore(snapshot))

    print("== 6. 重放校验 ==")
    print_json(replay_snapshot(str(snapshot), logger=make_logger("replay")))

    print("== 7. 异常注入 ==")
    for kind in ("corrupt-graph", "tamper-index", "zero-budget"):
        ns = argparse.Namespace(**vars(args))
        ns.kind = kind
        cmd_inject(ns)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="graphidx 演示 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--concurrency", type=int, default=4)
        p.add_argument("--budget", type=int, default=100)

    p = sub.add_parser("build")
    add_common(p)
    p.add_argument("--out", default=str(DEFAULT_SNAPSHOT))
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("update")
    add_common(p)
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.add_argument("--only", default=None)
    p.set_defaults(fn=cmd_update)

    p = sub.add_parser("query")
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.add_argument("--kind", choices=["adjacency", "label", "view"], default="adjacency")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--label", default="Person")
    p.add_argument("--view", default="reach_team")
    p.set_defaults(fn=cmd_query)

    p = sub.add_parser("restore")
    add_common(p)
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.set_defaults(fn=cmd_restore)

    p = sub.add_parser("replay")
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("inject")
    add_common(p)
    p.add_argument("kind", choices=["corrupt-graph", "tamper-index", "zero-budget"])
    p.set_defaults(fn=cmd_inject)

    p = sub.add_parser("all")
    add_common(p)
    p.set_defaults(fn=cmd_all)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except DomainError as exc:
        make_logger("session").error(exc.code, exc.message, details=exc.details)
        print_json(exc.to_dict())
        return 1


if __name__ == "__main__":
    sys.exit(main())
