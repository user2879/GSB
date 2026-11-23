"""副本修复预算调度诊断工作台 CLI 入口。

用法：
  python main.py run    --spec S --workdir W [--crash-after N]
                        [--inject-corrupt-block B | --inject-oversub | --inject-snapshot-mismatch]
  python main.py verify --spec S --workdir W     # 完整性 + 超订检查
  python main.py evolve --spec S --workdir W --group G --new-spec NS
  python main.py replay --spec S --workdir W [--group G]
  python main.py report --spec S --workdir W     # 当前视图（误差界 + 裁决）
  python main.py gui    --spec S --workdir W     # Tkinter 工作台（Windows 本地）
"""

from __future__ import annotations

import argparse
import json
import sys

from repairbench.errors import WorkbenchError
from repairbench.replay import replay_group
from repairbench.workbench import Session


def _emit(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def _apply_injections(session: Session, args, pre_load: bool = False) -> None:
    if pre_load:
        # 快照摘要不符须在 load 之前注入，使 load 的版本识别检查触发 E_SNAPSHOT。
        if args.inject_snapshot_mismatch:
            session.inject_snapshot_mismatch()
        return
    if args.inject_corrupt_block:
        session.inject_corrupt_block(args.inject_corrupt_block)
    if args.inject_oversub:
        session.inject_oversubscription()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="repairbench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("run", "verify", "evolve", "replay", "report", "gui", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
        p.add_argument("--workdir", required=True)
        p.add_argument("--group")
        p.add_argument("--new-spec")
        p.add_argument("--crash-after", type=int)
        p.add_argument("--inject-corrupt-block")
        p.add_argument("--inject-oversub", action="store_true")
        p.add_argument("--inject-snapshot-mismatch", action="store_true")
    args = parser.parse_args(argv)

    session = Session(args.workdir)
    try:
        if args.cmd == "run":
            _emit(session.load(args.spec))
            _apply_injections(session, args)
            _emit(session.run_all(crash_after_commits=args.crash_after))
        elif args.cmd == "verify":
            _emit(session.load(args.spec))
            _apply_injections(session, args)
            session.check_oversubscription()
            _emit({"verify": "ok"})
        elif args.cmd == "evolve":
            _emit(session.load(args.spec))
            with open(args.new_spec, "r", encoding="utf-8") as fh:
                new_spec = json.load(fh)
            _emit(session.evolve(args.group, new_spec))
        elif args.cmd == "replay":
            _apply_injections(session, args, pre_load=True)
            _emit(session.load(args.spec, recover=True))
            _apply_injections(session, args)
            groups = [args.group] if args.group else sorted(session.groups)
            for gid in groups:
                _emit(replay_group(session, gid))
        elif args.cmd == "report":
            _emit(session.load(args.spec))
            groups = [args.group] if args.group else sorted(session.groups)
            for gid in groups:
                _emit(session.current_view(gid))
        elif args.cmd == "probe":
            from repairbench.netprobe import probe_ports
            from repairbench.spec import load_spec
            _emit(probe_ports(load_spec(args.spec)))
        elif args.cmd == "gui":
            _emit(session.load(args.spec))
            from repairbench.gui import launch
            launch(session)
        return 0
    except WorkbenchError as err:
        session.log.emit_error(err)
        _emit({"error": err.to_dict()})
        return 1


if __name__ == "__main__":
    sys.exit(main())
