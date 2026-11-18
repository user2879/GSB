"""确定性样本生成：线格式抓包 + v1/v2/v3 事件清单。

所有序列与时间戳均为固定常量，无随机源，保证每次生成字节级一致，
从而让回环重放与账本校验可重复。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .constants import FLAG_ACK, FLAG_RETX
from .wire import Frame, encode_frame, flow_key

BASE_TS_US = 1_000_000  # t=1s
STEP_US = 10_000        # 每帧间隔 10ms

FLOW_A = dict(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=40001,
              dst_port=50001, protocol=6)
FLOW_B = dict(src_ip="10.0.0.3", dst_ip="10.0.0.4", src_port=40002,
              dst_port=50002, protocol=6)


def _frame(flow: Dict[str, Any], seq: int, ts_ms: int,
           flags: int = FLAG_ACK, payload: bytes = b"x") -> Dict[str, Any]:
    return {**flow, "seq": seq,
            "timestamp_us": BASE_TS_US + ts_ms * 1000,
            "flags": flags, "payload": payload}


def build_capture_frames() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """构造两条流，覆盖真实丢包、重传填充、乱序迟到三种形态。

    发送顺序即时间顺序（确定性）：
    Flow A（单方向，覆盖 1..13）：
    - seq 5、6 永不出现 -> true_loss；
    - seq 7、8 首跳缺失，随后带 RETX 的重复帧回填 -> retransmit_fill；
    - seq 9、10 首跳缺失，随后无 RETX 的迟到帧回填 -> ooo_late。
    Flow B：seq 100、101、105，102..104 永久缺失，用于统计误差演示。
    """
    frames: List[Dict[str, Any]] = []
    ts = 0

    def add(flow, seq, flags=FLAG_ACK):
        nonlocal ts
        frames.append(_frame(flow, seq, ts, flags=flags))
        ts += 10

    # 正常段
    for seq in (1, 2, 3, 4):
        add(FLOW_A, seq)
    # 跳到 11：打开缺口 5..10
    add(FLOW_A, 11)
    # 7、8 以重传形式回填（仍缺 5、6、9、10）
    add(FLOW_A, 7, flags=FLAG_ACK | FLAG_RETX)
    add(FLOW_A, 8, flags=FLAG_ACK | FLAG_RETX)
    # 9、10 以无 RETX 迟到帧回填
    add(FLOW_A, 9)
    add(FLOW_A, 10)
    # 补齐尾部正常段
    add(FLOW_A, 12)
    add(FLOW_A, 13)

    for seq in (100, 101, 105):
        add(FLOW_B, seq)

    meta = {
        "flow_a": flow_key(**{k: FLOW_A[k] for k in
                              ("src_ip", "dst_ip", "src_port",
                               "dst_port", "protocol")}),
        "flow_b": flow_key(**{k: FLOW_B[k] for k in
                              ("src_ip", "dst_ip", "src_port",
                               "dst_port", "protocol")}),
    }
    return frames, meta


def write_capture(path: str | Path) -> Dict[str, Any]:
    frames, meta = build_capture_frames()
    blob = b"".join(encode_frame(**item) for item in frames)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    meta["path"] = str(path)
    meta["frame_count"] = len(frames)
    return meta


def write_event_logs(out_dir: str | Path, flow_a: str,
                     flow_b: str) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = {
        "format": "loss-event-list",
        "version": "v1",
        "flow": flow_b,
        "events": [
            {"event_id": 1, "loss_count": 3},
        ],
        "coverage": {
            flow_b: {
                f"{FLOW_B['src_ip']}:{FLOW_B['src_port']}->"
                f"{FLOW_B['dst_ip']}:{FLOW_B['dst_port']}": [100, 105]}},
    }
    v2 = {
        "format": "loss-event-list",
        "version": "v2",
        "events": [
            {"event_id": 10, "flow": flow_a,
             "link_id": "L1", "window_ms": [1000, 1060],
             "loss_count": 2,
             "direction": f"{FLOW_A['src_ip']}:{FLOW_A['src_port']}->"
                          f"{FLOW_A['dst_ip']}:{FLOW_A['dst_port']}"},
        ],
        "coverage": {
            flow_a: {
                f"{FLOW_A['src_ip']}:{FLOW_A['src_port']}->"
                f"{FLOW_A['dst_ip']}:{FLOW_A['dst_port']}": [1, 13]}},
    }
    v3 = {
        "format": "loss-event-list",
        "version": "v3",
        "events": [
            {"event_id": 20, "flow": flow_a,
             "link_id": "L1", "window_ms": [1000, 1130],
             "loss_count": 4,
             "retransmitted": True, "ooo_depth": 3,
             "direction": f"{FLOW_A['src_ip']}:{FLOW_A['src_port']}->"
                          f"{FLOW_A['dst_ip']}:{FLOW_A['dst_port']}"},
        ],
        "coverage": {
            flow_a: {
                f"{FLOW_A['src_ip']}:{FLOW_A['src_port']}->"
                f"{FLOW_A['dst_ip']}:{FLOW_A['dst_port']}": [1, 13]}},
    }
    paths = {}
    for name, payload in (("events_v1.json", v1),
                          ("events_v2.json", v2),
                          ("events_v3.json", v3)):
        target = out_dir / name
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        paths[name] = str(target)
    return paths


def generate_workspace(out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    capture_meta = write_capture(out_dir / "capture.bin")
    event_paths = write_event_logs(
        out_dir, capture_meta["flow_a"], capture_meta["flow_b"])
    return {"capture": capture_meta, "events": event_paths,
            "root": str(out_dir)}
