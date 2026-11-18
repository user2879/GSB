"""确定性流量整形引擎与回环 TCP 模拟。

两种时钟：
- virtual（默认，确定性测试用）：包按固定间隔到达，时间为整数微秒，可精确重放；
- real：真实子进程 + 回环 TCP，使用 time.monotonic_ns 计时。

整形模型：每规则一个令牌桶
- 令牌速率 = rate_kbps * 1000 / 8 字节/秒；
- 桶容量 = burst_bytes；
- 包到达时先按经过时间补充令牌，再尝试扣除包长；
  令牌不足则按缺口计算延迟（排队），超过 queue_limit_packets 则丢包。
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Any

from .errors import ShapeError

FRAME_MAGIC = 0x514F5331  # "QOS1"
HEADER = struct.Struct("!IIII")  # magic, seq, dst_port, payload_len


@dataclass
class PacketSpec:
    seq: int
    dst_port: int
    size: int


@dataclass
class Observation:
    timestamp_us: int
    seq: int
    dst_port: int
    rule_id: str | None
    sent_bytes: int
    delay_us: int
    dropped: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_us": self.timestamp_us,
            "seq": self.seq,
            "dst_port": self.dst_port,
            "rule_id": self.rule_id,
            "sent_bytes": self.sent_bytes,
            "delay_us": self.delay_us,
            "dropped": self.dropped,
        }


class TokenBucket:
    def __init__(self, rate_kbps: int, burst_bytes: int):
        self.rate_bytes_per_sec = rate_kbps * 1000 // 8
        self.capacity = burst_bytes
        self.tokens = float(burst_bytes)
        self.last_time_us = 0

    def account(self, now_us: int, size: int) -> tuple[int, bool]:
        """返回 (延迟微秒, 是否丢包)。本引擎不主动丢包（队列层处理），仅算延迟。"""
        elapsed_us = max(0, now_us - self.last_time_us)
        self.tokens = min(
            float(self.capacity),
            self.tokens + elapsed_us * self.rate_bytes_per_sec / 1_000_000.0)
        self.last_time_us = now_us
        if self.tokens >= size:
            self.tokens -= size
            return 0, False
        deficit = size - self.tokens
        delay_us = int(-(-deficit * 1_000_000 // self.rate_bytes_per_sec))  # 向上取整
        self.tokens -= size  # 允许为负，代表排队欠债，后续补充偿还
        return delay_us, False


def match_rule(entries: list[dict[str, Any]], dst_port: int,
               dst_addr: str = "127.0.0.1") -> dict[str, Any] | None:
    """最长端口前缀匹配；同长按规则标识稳定决胜。"""
    port_text = str(dst_port)
    best: tuple[int, str, dict[str, Any]] | None = None
    for entry in entries:
        if entry["match"]["dst_addr"] not in (dst_addr, "::1", "127.0.0.1"):
            continue
        for prefix in entry["match"]["port_prefixes"]:
            if port_text.startswith(str(prefix)):
                key = (len(str(prefix)), entry["rule_id"])
                if best is None or key > (best[0], best[1]):
                    best = (key[0], key[1], entry)
    return best[2] if best else None


@dataclass
class SimulationResult:
    observations: list[Observation]
    forwarded_bytes: int
    duration_us: int
    per_rule: dict[str, dict[str, int]]
    plan_version: int

    def digest(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "duration_us": self.duration_us,
            "forwarded_bytes": self.forwarded_bytes,
            "packet_count": len(self.observations),
            "dropped_count": sum(1 for o in self.observations if o.dropped),
            "per_rule": self.per_rule,
        }

    def records(self) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.observations]


def deterministic_packets(count: int, size: int, dst_ports: list[int]) -> list[PacketSpec]:
    ports = dst_ports or [14000]
    return [PacketSpec(seq=i, dst_port=ports[i % len(ports)], size=size)
            for i in range(count)]


def run_virtual(entries: list[dict[str, Any]], packets: list[PacketSpec],
                plan_version: int, arrival_gap_us: int = 50_000) -> SimulationResult:
    """虚拟时钟确定性整形，不做任何真实 sleep / socket。"""
    buckets: dict[str, TokenBucket] = {}
    queue_depth: dict[str, int] = {}
    observations: list[Observation] = []
    forwarded = 0
    per_rule: dict[str, dict[str, int]] = {}
    now_us = 0

    for pkt in packets:
        entry = match_rule(entries, pkt.dst_port)
        if entry is None:
            observations.append(Observation(
                now_us, pkt.seq, pkt.dst_port, None, 0, 0, True))
            now_us += arrival_gap_us
            continue
        rid = entry["rule_id"]
        shaping = entry["shaping"]
        bucket = buckets.setdefault(
            rid, TokenBucket(shaping["rate_kbps"], shaping["burst_bytes"]))
        bucket.last_time_us = now_us
        stats = per_rule.setdefault(rid, {"forwarded_bytes": 0, "delay_us": 0,
                                          "packets": 0, "dropped": 0})
        limit = shaping["queue_limit_packets"]
        if queue_depth.get(rid, 0) >= limit > 0:
            observations.append(Observation(
                now_us, pkt.seq, pkt.dst_port, rid, 0, 0, True))
            stats["dropped"] += 1
            now_us += arrival_gap_us
            continue
        delay_us, _ = bucket.account(now_us, pkt.size)
        queue_depth[rid] = queue_depth.get(rid, 0) + (1 if delay_us else 0)
        observations.append(Observation(
            now_us + delay_us, pkt.seq, pkt.dst_port, rid, pkt.size, delay_us, False))
        forwarded += pkt.size
        stats["forwarded_bytes"] += pkt.size
        stats["delay_us"] += delay_us
        stats["packets"] += 1
        now_us += arrival_gap_us

    duration_us = max(1, now_us)
    return SimulationResult(observations, forwarded, duration_us, per_rule, plan_version)


def verify_throughput(result: SimulationResult,
                      entries: list[dict[str, Any]],
                      tolerance_ratio: float = 0.05) -> dict[str, Any]:
    """校验每规则平均吞吐不超过承诺速率，并单独允许一次突发容量。

    判定：actual_bps <= committed_bps * (1 + tolerance_ratio)。
    突发容量通过令牌桶延迟体现（首包可线速发送 burst_bytes），
    因此长期平均速率仍须被承诺速率约束。超出 -> E_SHAPE，
    附承诺速率与实际吞吐的整数对比（字节/秒）。
    """
    seconds = result.duration_us / 1_000_000.0
    entry_by_id = {e["rule_id"]: e for e in entries}
    report: dict[str, Any] = {}
    for rid, stats in result.per_rule.items():
        rate_bps = entry_by_id[rid]["shaping"]["rate_kbps"] * 1000 // 8
        burst = entry_by_id[rid]["shaping"]["burst_bytes"]
        actual_bps = int(stats["forwarded_bytes"] / seconds)
        allowed_bps = int(rate_bps * (1.0 + tolerance_ratio))
        report[rid] = {
            "committed_bytes_per_sec": rate_bps,
            "actual_bytes_per_sec": actual_bps,
            "allowed_bytes_per_sec": allowed_bps,
            "forwarded_bytes": stats["forwarded_bytes"],
            "burst_bytes": burst,
            "max_observed_delay_us": stats["delay_us"],
        }
        if actual_bps > allowed_bps:
            raise ShapeError(
                f"规则 {rid} 实际吞吐超出整形承诺",
                details={
                    "rule_id": rid,
                    "committed_bytes_per_sec": rate_bps,
                    "actual_bytes_per_sec": actual_bps,
                    "allowed_bytes_per_sec": allowed_bps,
                    "forwarded_bytes": stats["forwarded_bytes"],
                },
            )
    return report


# ---- 真实回环 TCP 子进程支持 ---------------------------------------------
def build_frame(seq: int, dst_port: int, payload: bytes) -> bytes:
    return HEADER.pack(FRAME_MAGIC, seq, dst_port, len(payload)) + payload


def recv_exact(conn: socket.socket, length: int) -> bytes | None:
    """读满 length 字节；对端正常关闭且无数据时返回 None。"""
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            if not chunks:
                return None
            raise ConnectionError("连接提前关闭，帧不完整")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(conn: socket.socket) -> tuple[int, int, bytes] | None:
    header = recv_exact(conn, HEADER.size)
    if header is None:
        return None
    magic, seq, dst_port, payload_len = HEADER.unpack(header)
    if magic != FRAME_MAGIC:
        raise ValueError(f"非法帧魔数: {magic:#x}")
    payload = recv_exact(conn, payload_len) if payload_len else b""
    return seq, dst_port, payload or b""


def is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True
