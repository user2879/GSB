"""命令行入口与可重复演示。

子命令::

    generate   生成确定性 half 帧序列与真值清单
    shell      交互式工作台
    demo       一条命令：生成素材 -> 固定操作序列 -> 模拟强制终止 ->
               重启恢复 -> 输出逐帧哈希清单与恢复报告
    verify     全量哈希 + 实算校验
    hashes     输出逐帧哈希清单
    selftest   两次完整演示逐字节一致性自检
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import sys

from .engine import WorkbenchEngine
from .errors import WorkbenchError
from .persistence import (DIR_OUTPUTS, SHUTDOWN_NAME, atomic_write_json,
                          output_name)
from .shell import InteractiveShell
from .state import canonical_json
from .synthesis import DEFAULT_FRAME_COUNT, DEFAULT_HEIGHT, DEFAULT_WIDTH, SequenceConfig, generate_sequence

DEMO_OPS = [
    {"frames": [0, 1, 2, 3, 4], "changes": {"exposure_stops": 0.5}, "kind": "range", "note": "区间提亮片头"},
    {"frames": [10, 11, 12, 13, 14], "changes": {"tonemap": "reinhard", "tonemap_param": 8.0},
     "kind": "range", "note": "区间切换 Reinhard"},
    {"frames": [30], "changes": {"colorspace": "bt709", "oetf": "srgb"},
     "kind": "frame", "note": "逐帧 BT.709+sRGB"},
    {"frames": list(range(20, 26)), "changes": {"exposure_stops": -1.0},
     "kind": "range", "note": "区间压暗"},
    {"frames": [45, 46, 47], "changes": {"tonemap": "reinhard", "tonemap_param": 4.0,
                                         "colorspace": "bt709"},
     "kind": "range", "note": "区间组合调整"},
]


def build_config(args) -> SequenceConfig:
    if args.quick:
        return SequenceConfig(width=320, height=180, frame_count=60, seed=20240517)
    return SequenceConfig(width=getattr(args, "width", DEFAULT_WIDTH),
                          height=getattr(args, "height", DEFAULT_HEIGHT),
                          frame_count=getattr(args, "frames", DEFAULT_FRAME_COUNT),
                          seed=getattr(args, "seed", 20240517))


def cmd_generate(args) -> int:
    cfg = build_config(args)
    manifest = generate_sequence(args.workspace, cfg)
    atomic_write_json(os.path.join(args.workspace, "source_manifest.json"), manifest)
    print(f"已生成 {cfg.frame_count} 帧 {cfg.width}x{cfg.height} half 线性光 -> "
          f"{os.path.join(args.workspace, 'frames')}")
    return 0


def load_source_manifest(workspace: str) -> dict:
    path = os.path.join(workspace, "source_manifest.json")
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _write_hash_listing(engine: WorkbenchEngine, path: str) -> list[dict]:
    listing = engine.hash_listing()
    atomic_write_json(path, listing)
    return listing


def run_demo(workspace: str, cfg: SequenceConfig, *, crash: bool,
             crash_after_op: int = 2, leave_orphan: bool = True) -> dict:
    """执行固定演示序列。crash=True 时在指定操作后模拟强制终止。"""
    src_path = os.path.join(workspace, "source_manifest.json")
    if not os.path.exists(src_path):
        manifest = generate_sequence(workspace, cfg)
        atomic_write_json(src_path, manifest)
    source = load_source_manifest(workspace)
    engine = WorkbenchEngine(workspace, source)

    for index, op in enumerate(DEMO_OPS):
        engine.apply_operation(op["frames"], op["changes"],
                               kind=op["kind"], note=op["note"])
        if crash and index == crash_after_op:
            if leave_orphan:
                # 模拟“写入未完成”：制造一个未被 manifest 引用的孤儿产物
                orphan = os.path.join(engine.store.paths[DIR_OUTPUTS],
                                      output_name(9999, 999999))
                with open(orphan, "wb") as fh:
                    fh.write(b"\x00" * 64)
                tmp_orphan = os.path.join(engine.store.paths["tmp"], ".partial.tmp")
                with open(tmp_orphan, "wb") as fh:
                    fh.write(b"partial-write")
            # 强制终止：不写 shutdown report，直接抛出，进程退出
            raise SystemExit(137)
    return {"engine": engine, "source": source}


def cmd_demo(args) -> int:
    workspace = args.workspace
    if args.clean and os.path.exists(workspace):
        shutil.rmtree(workspace)
    cfg = build_config(args)

    # 阶段 1：带强制终止的运行
    crashed = False
    try:
        run_demo(workspace, cfg, crash=True, crash_after_op=2)
    except SystemExit as exc:
        crashed = exc.code == 137
    assert crashed, "强制终止模拟未生效"
    crash_manifest = os.path.join(workspace, "manifest.json")
    committed_seq_before = None
    with open(crash_manifest, "rb") as fh:
        committed_seq_before = json.loads(fh.read())["committed_at_seq"]
    assert not os.path.exists(os.path.join(workspace, SHUTDOWN_NAME)), \
        "强制终止不得产生退出报告"

    # 阶段 2：重启恢复（应回收孤儿并报 E_RECOVER_ORPHAN，之后继续可操作）
    source = load_source_manifest(workspace)
    engine = WorkbenchEngine(workspace, source)
    recovery = engine.recover()
    assert recovery["recovered"] is True
    assert recovery["committed_at_seq"] == committed_seq_before

    # 阶段 3：从最近稳定状态继续剩余操作（崩溃发生在第 2 个操作之后，
    # 已提交的演示操作序号为 0..crash_after_op）
    for op in DEMO_OPS[2 + 1:]:
        engine.apply_operation(op["frames"], op["changes"],
                               kind=op["kind"], note=op["note"])
    engine.verify_all()
    listing = _write_hash_listing(
        engine, os.path.join(workspace, "hash_listing.json"))
    shutdown = engine.controlled_shutdown()
    atomic_write_json(os.path.join(workspace, "recovery_report.json"), {
        "crashed": True,
        "exit_code_simulated": 137,
        "committed_seq_at_crash": committed_seq_before,
        "recovery": {k: v for k, v in recovery.items() if k != "recovered"},
        "shutdown": shutdown,
        "frame_count": len(listing),
    })
    print(f"演示完成：崩溃于 v{committed_seq_before}，恢复后推进至 "
          f"v{len(engine.state_table.operations) - 1}，共 {len(listing)} 帧哈希")
    print(f"恢复报告: {os.path.join(workspace, 'recovery_report.json')}")
    return 0


def cmd_selftest(args) -> int:
    """两次完整演示逐字节一致性。"""
    base = args.workspace
    ws_a = os.path.join(base, "run_a")
    ws_b = os.path.join(base, "run_b")
    for ws in (ws_a, ws_b):
        if os.path.exists(ws):
            shutil.rmtree(ws)
    cfg = build_config(args)
    for ws in (ws_a, ws_b):
        args_copy = argparse.Namespace(**vars(args))
        args_copy.workspace = ws
        args_copy.clean = True
        cmd_demo(args_copy)
    compare = ["hash_listing.json", "recovery_report.json", "manifest.json"]
    all_same = True
    for name in compare:
        same = filecmp.cmp(os.path.join(ws_a, name), os.path.join(ws_b, name), shallow=False)
        all_same &= same
        print(f"逐字节比对 {name}: {'一致' if same else '不一致'}")
    # 源帧逐字节
    src_same = True
    fa = os.path.join(ws_a, "frames")
    fb = os.path.join(ws_b, "frames")
    for name in sorted(os.listdir(fa)):
        src_same &= filecmp.cmp(os.path.join(fa, name), os.path.join(fb, name), shallow=False)
    print(f"逐字节比对 frames/: {'一致' if src_same else '不一致'}")
    return 0 if (all_same and src_same) else 2


def cmd_shell(args) -> int:
    source = load_source_manifest(args.workspace)
    engine = WorkbenchEngine(args.workspace, source)
    if engine.store.shutdown_report_exists():
        engine.store.clear_shutdown_report()
    shell = InteractiveShell(engine, input_stream=sys.stdin, output_stream=sys.stdout)
    shell.run()
    engine.controlled_shutdown()
    return 0


def cmd_verify(args) -> int:
    source = load_source_manifest(args.workspace)
    engine = WorkbenchEngine(args.workspace, source)
    result = engine.verify_all()
    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_hashes(args) -> int:
    source = load_source_manifest(args.workspace)
    engine = WorkbenchEngine(args.workspace, source)
    listing = _write_hash_listing(engine, os.path.join(args.workspace, "hash_listing.json"))
    for row in listing:
        print(f"frame {row['frame']:04d} v{row['version']:06d} {row['output_sha256']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hdr-sdr-workbench",
                                 description="HDR -> SDR 专业色彩转换工作台")
    parser.add_argument("--workspace", default="./workspace", help="工作区目录")
    parser.add_argument("--quick", action="store_true", help="快速模式 320x180（仍 60 帧）")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--seed", type=int, default=20240517)
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="生成确定性帧序列")
    p_gen.set_defaults(func=cmd_generate)

    p_demo = sub.add_parser("demo", help="固定操作序列 + 强制终止 + 恢复演示")
    p_demo.add_argument("--clean", action="store_true", help="开始前清空工作区")
    p_demo.set_defaults(func=cmd_demo)

    p_self = sub.add_parser("selftest", help="两次演示逐字节一致性")
    p_self.set_defaults(func=cmd_selftest)

    p_shell = sub.add_parser("shell", help="交互式工作台")
    p_shell.set_defaults(func=cmd_shell)

    p_ver = sub.add_parser("verify", help="全量校验")
    p_ver.set_defaults(func=cmd_verify)

    p_hash = sub.add_parser("hashes", help="逐帧哈希清单")
    p_hash.set_defaults(func=cmd_hashes)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.makedirs(args.workspace, exist_ok=True)
    try:
        return args.func(args)
    except WorkbenchError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
