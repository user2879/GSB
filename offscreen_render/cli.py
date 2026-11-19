"""命令行入口：generate / render / verify，统一 JSON 结果与约定退出码。"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading

from .errors import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_OK, RenderError
from .samplegen import generate_sample


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()


def _install_signal_handler(event: threading.Event, state_dir: str | None) -> None:
    def handler(signum, frame):  # noqa: ANN001
        event.set()
        if state_dir:
            try:
                from .persistence import write_interrupt_marker
                write_interrupt_marker(state_dir)
            except RenderError:
                pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # 非主线程或平台不支持时跳过；验证脚本可直接写 INTERRUPT 文件。
            pass


def _cmd_generate(args: argparse.Namespace) -> int:
    paths = generate_sample(
        args.sample_dir, args.state_dir, args.output_dir,
        width=args.width, height=args.height, supersample=args.supersample)
    _emit({"ok": True, "action": "generate", **paths})
    return EXIT_OK


def _cmd_render(args: argparse.Namespace) -> int:
    event = threading.Event()
    # 先解析场景以拿到 state_dir，再安装信号处理器。
    from .scene import load_scene
    try:
        scene = load_scene(args.scene)
        from .persistence import (
            interrupt_marker_exists,
            remove_interrupt_marker,
        )
        if interrupt_marker_exists(scene.config.state_dir):
            if args.resume or args.keep_interrupt:
                # 恢复模式严格尊重已有中断标记：在首个分块边界即中止。
                event.set()
            else:
                # 全新运行主动清除上轮残留标记。
                remove_interrupt_marker(scene.config.state_dir)
        _install_signal_handler(event, scene.config.state_dir)
        from .pipeline import RenderPipeline
        outcome = RenderPipeline(scene, event).execute(run_stability=not args.no_stability)
    except RenderError as exc:
        if exc.code == "E_4001":
            _emit({"ok": False, "interrupted": True, **exc.to_dict()})
            return EXIT_INTERRUPTED
        _emit(exc.to_dict())
        return EXIT_ERROR
    _emit({"ok": True, "action": "render", **outcome.__dict__})
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="offscreen-render",
                                     description="本地离屏多介质透射渲染引擎")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性本地样本")
    gen.add_argument("--sample-dir", required=True)
    gen.add_argument("--state-dir", required=True)
    gen.add_argument("--output-dir", required=True)
    gen.add_argument("--width", type=int, default=32)
    gen.add_argument("--height", type=int, default=24)
    gen.add_argument("--supersample", type=int, default=2)
    gen.set_defaults(func=_cmd_generate)

    rnd = sub.add_parser("render", help="按场景描述渲染并输出灰度图与报告")
    rnd.add_argument("--scene", required=True)
    rnd.add_argument("--resume", action="store_true", help="尊重已有中断标记并恢复")
    rnd.add_argument("--keep-interrupt", action="store_true")
    rnd.add_argument("--no-stability", action="store_true")
    rnd.set_defaults(func=_cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RenderError as exc:
        _emit(exc.to_dict())
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
