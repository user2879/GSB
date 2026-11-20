"""端到端演示入口（Windows 命令行 / 任意平台均可运行）。

子命令：
  run          确定性裁决 + 账本 + 快照
  replay       恢复快照并一致性重放比对（E_REPLAY 校验）
  faults       注入三类异常（E_CHANNEL / E_ESTIMATE / E_SNAPSHOT）
  udp          真实 127.0.0.1 回环 UDP 收发链路验证
  inspect      导出账本快照与裁决摘要（渲染顺序稳定）
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tunnel_reassembly.config import load_json
from tunnel_reassembly.engine import build_groups, estimate_versions
from tunnel_reassembly.errors import BusinessError
from tunnel_reassembly.faults import corrupt_snapshot_digest, force_estimate_pending, inject_bad_channel
from tunnel_reassembly.logutil import StructuredLogger
from tunnel_reassembly.ui import DiagnosticView, build_view_model
from tunnel_reassembly.session import Session

DEFAULT_CONFIG = ROOT / "samples" / "channels.json"
DEFAULT_TRACE = ROOT / "samples" / "trace.json"
DEFAULT_DB = ROOT / "logs" / "session.db"
DEFAULT_LOG = ROOT / "logs" / "events.jsonl"


def _summarize(session: Session) -> dict:
    summary: dict[str, dict] = {}
    for cid in sorted(session.results):
        result = session.results[cid]
        summary[cid] = {
            "estimate_versions": [
                {"version": v.version, "skew_ms": v.skew_ms, "samples": v.sample_count}
                for v in result.estimates
                if v.sample_count > 0
            ],
            "window_events": [
                {"seq": e.window_seq, "action": e.action, "base_ms": e.base_ms, "close_ms": e.close_ms}
                for e in result.window_events
            ],
            "flips": [
                {
                    "group": f.group_seq,
                    "from_version": f.from_version,
                    "to_version": f.to_version,
                    "kind": f.flip_kind,
                    "from_window": f.from_window,
                    "to_window": f.to_window,
                    "from_adjusted_ms": f.from_adjusted_ms,
                    "to_adjusted_ms": f.to_adjusted_ms,
                }
                for f in result.flips
            ],
            "outcomes": [
                {"group": o.group_seq, "status": o.status, "window": o.window_seq, "reason": o.reason}
                for o in result.outcomes
            ],
        }
    return summary


def cmd_run(args) -> int:
    logger = StructuredLogger(args.log)
    session = Session(args.db)
    try:
        config = load_json(args.config)
        trace = load_json(args.trace)
        session.load_inputs(config, trace)
        session.run_deterministic()
        digest = session.snapshot("snap-0001")
        logger.event("INFO", "OK", "确定性裁决完成并写入快照", snapshot="snap-0001", content_digest=digest)
        print(json.dumps(_summarize(session), indent=2, ensure_ascii=False))
        print(f"\n快照 content_digest={digest}")
        print(f"账本={args.db} 日志={args.log}")
    finally:
        session.close()
        logger.close()
    return 0


def cmd_replay(args) -> int:
    logger = StructuredLogger(args.log)
    session = Session(args.db)
    try:
        config = load_json(args.config)
        trace = load_json(args.trace)
        session.load_inputs(config, trace)
        payload, _rerun = session.restore_and_replay()
        logger.event(
            "INFO",
            "OK",
            "恢复后重放逐项一致",
            unknown_inflight=payload.get("unknown_inflight", 0),
            snapshot=payload["snapshot_id"],
        )
        print("E_REPLAY 校验通过：估计序列/窗口/翻转/结局/逐片记录逐项一致")
        print(f"UNKNOWN 在途组数量={payload.get('unknown_inflight', 0)}（强制终止语义下要求重跑）")
        print("快照列表=" + json.dumps(session.ledger.list_snapshots(), ensure_ascii=False))
    finally:
        session.close()
        logger.close()
    return 0


def cmd_faults(args) -> int:
    logger = StructuredLogger(args.log)
    config = load_json(args.config)
    trace = load_json(args.trace)

    print("== 异常注入 1：通道声明损坏（期望 E_CHANNEL）==")
    try:
        inject_bad_channel(config)
    except BusinessError as exc:
        logger.event("ERROR", exc.code, exc.detail, path=exc.path)
        print(f"{exc}")

    print("\n== 异常注入 2：偏差样本不足（期望 E_ESTIMATE）==")
    session = Session(tempfile.mktemp(suffix=".db"))
    try:
        session.load_inputs(config, trace)
        spec = session.specs[0]
        groups = build_groups(session._events_from_trace(spec.channel_id))
        pending = force_estimate_pending(spec, groups, logger)
        print(f"通道 {spec.channel_id} 被标记 PENDING 的组数={pending}，估计未定拒绝开窗")
    finally:
        session.close()

    print("\n== 异常注入 3：快照摘要损坏（期望 E_SNAPSHOT）==")
    session = Session(args.db)
    try:
        session.load_inputs(config, trace)
        corrupt_snapshot_digest(session.ledger)
        try:
            session.ledger.load_latest_snapshot(session.config_version, session.trace_version)
        except BusinessError as exc:
            logger.event("ERROR", exc.code, exc.detail, path=exc.path)
            print(f"{exc}")
            print("触发整体替换：replace_all 清空通道/参数/窗口状态/快照")
            session.ledger.replace_all()
    finally:
        session.close()
        logger.close()
    print(f"\n结构化日志已写入 {args.log}")
    return 0


def cmd_udp(args) -> int:
    from tunnel_reassembly.clock import MonotonicClock
    from tunnel_reassembly.transport import UdpReceiver, UdpSender, assert_port_free

    logger = StructuredLogger(args.log)
    config = load_json(args.config)
    trace = load_json(args.trace)
    session = Session(tempfile.mktemp(suffix=".db"))
    try:
        session.load_inputs(config, trace)
        spec = session.specs[0]
        assert_port_free(spec.host, spec.port)
        print(f"真实回环 UDP：在 {spec.host}:{spec.port} 监听通道 {spec.channel_id}")
        receiver = UdpReceiver(spec, MonotonicClock())
        sender = UdpSender()
        expected = 0
        try:
            for group in session.trace[spec.channel_id]:
                for frag in group["fragments"]:
                    sender.send_fragment(
                        spec,
                        {
                            "group_seq": group["group_seq"],
                            "frag_offset": frag["frag_offset"],
                            "frag_length": frag["frag_length"],
                            "total_frags": group["total_frags"],
                            "send_ms": group["send_ms"],
                            "payload": frag["payload"],
                        },
                    )
                    expected += 1
            received = []
            for _ in range(expected * 2):
                event = receiver.recv_one()
                if event is None:
                    break
                received.append(event)
                if len(received) == expected:
                    break
        finally:
            receiver.close()
            sender.close()
        logger.event(
            "INFO",
            "OK",
            "真实回环 UDP 收片完成",
            sent=expected,
            received=len(received),
            channel=spec.channel_id,
        )
        print(f"发送 {expected} 片，接收 {len(received)} 片（含重复片）")
        if len(received) != expected:
            raise BusinessError("E_PROTOCOL", f"收发片数不一致 {expected} != {len(received)}")
    finally:
        session.close()
        logger.close()
    return 0


def cmd_ui(args) -> int:
    session = Session(args.db)
    try:
        session.load_inputs(load_json(args.config), load_json(args.trace))
        session.run_deterministic()
        view = build_view_model(session)
        DiagnosticView(view).render()
    finally:
        session.close()
    return 0


def cmd_inspect(args) -> int:
    session = Session(args.db)
    try:
        print("快照列表：")
        print(json.dumps(session.ledger.list_snapshots(), indent=2, ensure_ascii=False))
    finally:
        session.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="隧道重组窗口裁决组件演示")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("replay").set_defaults(func=cmd_replay)
    sub.add_parser("faults").set_defaults(func=cmd_faults)
    sub.add_parser("udp").set_defaults(func=cmd_udp)
    sub.add_parser("inspect").set_defaults(func=cmd_inspect)
    sub.add_parser("ui").set_defaults(func=cmd_ui)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BusinessError as exc:
        print(f"业务错误 {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
