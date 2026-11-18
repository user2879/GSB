"""本地回环 TCP 上的缺口形态复现与结论校验。

进程内同时启动模拟对端（server 线程）与重放端（client），仅绑定
``127.0.0.1``，无任何外部网络依赖。协议复用自定义线格式帧，每帧前加
4 字节大端长度前缀，对端按帧解码后回传观测摘要。

校验内容逐流、逐缺口区间比较：
- 缺口区间边界（start_seq/end_seq）；
- 成因标签（true_loss/retransmit_fill/ooo_late/undetermined）。
首个不一致即以 E_REPLAY 报告流、区间与两侧成因标签。
"""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import ReplayMismatchError
from .gaps import FlowGapConclusion
from .wire import Frame, decode_buffer, encode_frame

_LOOPBACK = "127.0.0.1"


class _EmptyConclusion:
    gaps = []
_LENGTH_PREFIX = struct.Struct(">I")


@dataclass
class ReplayReport:
    host: str
    port: int
    flows: int
    frames_sent: int
    frames_observed: int
    matched_gaps: int
    repeatable: bool = True
    peer_observations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "flows": self.flows,
            "frames_sent": self.frames_sent,
            "frames_observed": self.frames_observed,
            "matched_gaps": self.matched_gaps,
            "repeatable": self.repeatable,
            "peer_observations": self.peer_observations,
        }


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        block = conn.recv(size - len(chunks))
        if not block:
            raise ReplayMismatchError(
                "回环连接提前关闭，帧接收不完整",
                {"reason": "connection_closed", "need": size,
                 "have": len(chunks)})
        chunks.extend(block)
    return bytes(chunks)


def _read_frames(conn: socket.socket) -> List[Frame]:
    blob = bytearray()
    length_buf = _recv_exact(conn, _LENGTH_PREFIX.size)
    (total,) = _LENGTH_PREFIX.unpack(length_buf)
    remaining = total
    while remaining:
        block = conn.recv(min(65536, remaining))
        if not block:
            raise ReplayMismatchError("回环连接截断，线格式数据不完整",
                                      {"reason": "truncated_wire"})
        blob.extend(block)
        remaining -= len(block)
    decoded = decode_buffer(bytes(blob))
    return decoded.frames


def _peer_server(host: str, port: int,
                 holder: Dict[str, Any]) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)

    def serve() -> None:
        conn, _ = server.accept()
        with conn:
            try:
                frames = _read_frames(conn)
                decoded = decode_buffer(
                    b"".join(_frame_bytes(f) for f in frames))
                holder["frames"] = decoded.frames
                summary = _summary(decoded.frames)
                payload = _json_len(summary)
                conn.sendall(_LENGTH_PREFIX.pack(len(payload)) + payload)
            except ReplayMismatchError as exc:
                holder["error"] = exc

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    holder["thread"] = thread
    holder["server"] = server
    return server


def _frame_bytes(frame: Frame) -> bytes:
    return encode_frame(
        timestamp_us=frame.timestamp_us, src_ip=frame.src_ip,
        dst_ip=frame.dst_ip, src_port=frame.src_port,
        dst_port=frame.dst_port, protocol=frame.protocol,
        flags=frame.flags, seq=frame.seq, payload=frame.payload,
        flow_tag=frame.flow_tag)


def _json_len(obj: Any) -> bytes:
    import json
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _summary(frames: List[Frame]) -> Dict[str, Any]:
    by_flow: Dict[str, List[Frame]] = {}
    for frame in frames:
        by_flow.setdefault(frame.flow_key, []).append(frame)
    out: Dict[str, Any] = {}
    for flow, flow_frames in sorted(by_flow.items()):
        per_dir: Dict[str, List[int]] = {}
        for frame in flow_frames:
            per_dir.setdefault(frame.direction_key, []).append(frame.seq)
        out[flow] = {d: sorted(set(seqs)) for d, seqs in sorted(per_dir.items())}
    return out


def _peer_gaps(observed_seqs: Dict[str, List[int]]
               ) -> Dict[str, List[Tuple[int, int]]]:
    """对端仅依据实际收到的序号集合计算缺口区间。"""
    result: Dict[str, List[Tuple[int, int]]] = {}
    for direction, seqs in observed_seqs.items():
        ordered = sorted(seqs)
        if not ordered:
            continue
        gaps: List[Tuple[int, int]] = []
        lo, hi = ordered[0], ordered[0]
        for seq in ordered[1:]:
            if seq == hi + 1:
                hi = seq
                continue
            gaps.append((hi + 1, seq - 1))
            lo, hi = seq, seq
        result[direction] = gaps
    return result


def replay_and_verify(frames_by_flow: Dict[str, List[Frame]],
                      conclusions: Dict[str, FlowGapConclusion], *,
                      port: int = 41793,
                      host: str = _LOOPBACK,
                      tamper: bool = False) -> ReplayReport:
    """重放帧序列并把对端观测与诊断结论逐项比较。"""
    holder: Dict[str, Any] = {}
    server = _peer_server(host, port, holder)
    frames_sent = 0
    try:
        with socket.create_connection((host, port), timeout=5) as client:
            wire_blob = bytearray()
            ordered: List[Frame] = []
            for flow in sorted(frames_by_flow):
                flow_frames = sorted(
                    frames_by_flow[flow],
                    key=lambda f: (f.direction_key, f.seq, f.timestamp_us,
                                   f.offset))
                ordered.extend(flow_frames)
            if tamper:
                # 注入一个诊断结论中“真实丢包”的序号，使对端观测到的
                # 缺口区间与结论不一致，制造可复现的 E_REPLAY 场景。
                injected = False
                for target in ordered:
                    if not any(
                            g.cause == "true_loss" and
                            g.start_seq <= target.seq + 1 <= g.end_seq
                            for g in conclusions.get(
                                target.flow_key,
                                _EmptyConclusion()).gaps):
                        continue
                    fill_seq = target.seq + 1
                    if any(f.seq == fill_seq and
                           f.flow_key == target.flow_key for f in ordered):
                        continue
                    ordered.append(Frame(
                        offset=target.offset,
                        timestamp_us=target.timestamp_us + 1000,
                        src_ip=target.src_ip, dst_ip=target.dst_ip,
                        src_port=target.src_port,
                        dst_port=target.dst_port,
                        protocol=target.protocol, flags=target.flags,
                        seq=fill_seq, payload_len=0,
                        flow_tag=target.flow_tag, payload=b"",
                        index=target.index + 1))
                    injected = True
                    break
                if not injected:
                    raise ReplayMismatchError(
                        "tamper 模式未找到可注入的真实丢包序号",
                        {"reason": "no_injectable_gap"})
            for frame in ordered:
                wire_blob.extend(_frame_bytes(frame))
                frames_sent += 1
            client.sendall(_LENGTH_PREFIX.pack(len(wire_blob)) + wire_blob)
            length_buf = _recv_exact(client, _LENGTH_PREFIX.size)
            (resp_len,) = _LENGTH_PREFIX.unpack(length_buf)
            import json
            response = json.loads(_recv_exact(client, resp_len).decode())
    finally:
        holder["thread"].join(timeout=5)
        server.close()

    if "error" in holder:
        raise holder["error"]

    observed_frames = holder.get("frames", [])
    observed_summary = response
    observed_gaps = {
        flow: _peer_gaps(dirs) for flow, dirs in observed_summary.items()}

    matched = 0
    for flow in sorted(conclusions):
        conclusion = conclusions[flow]
        peer_per_dir = observed_gaps.get(flow, {})
        for gap in conclusion.gaps:
            peer_intervals = peer_per_dir.get(gap.direction, [])
            if (gap.start_seq, gap.end_seq) not in peer_intervals:
                if gap.cause in ("retransmit_fill", "ooo_late"):
                    # 被填充的缺口在对端最终序号集合中不再缺失，符合预期
                    matched += 1
                    continue
                raise ReplayMismatchError(
                    f"流 {flow} 重放缺口区间不一致",
                    {"flow": flow, "direction": gap.direction,
                     "expected_interval": [gap.start_seq, gap.end_seq],
                     "expected_cause": gap.cause,
                     "peer_intervals": peer_intervals,
                     "reason": "interval_missing_at_peer"})
            # 区间一致：true_loss 必须在对端仍为缺失
            if gap.cause == "true_loss":
                matched += 1
                continue
            raise ReplayMismatchError(
                f"流 {flow} 成因标签与对端转发行为不一致",
                {"flow": flow, "direction": gap.direction,
                 "interval": [gap.start_seq, gap.end_seq],
                 "diagnosed_cause": gap.cause,
                 "peer_sees_interval_as": "missing",
                 "reason": "cause_label_mismatch"})

    return ReplayReport(
        host=host, port=port, flows=len(conclusions),
        frames_sent=frames_sent, frames_observed=len(observed_frames),
        matched_gaps=matched, repeatable=True,
        peer_observations=observed_summary)
