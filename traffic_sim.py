#!/usr/bin/env python3
"""本地回环确定性流量发生器 + 整形器 + 接收端模拟脚本。

仅使用标准库，监听地址固定为 127.0.0.1 高位端口，不依赖容器或线上账号。

子命令：
  sink      启动接收端，统计收到的字节并写出 JSON
  shaper    启动整形器（令牌桶 + 真实 sleep 延迟），转发到接收端
  generator 启动发生器，向整形器发送确定性整数编号字节流
  run       以子进程拉起 sink/shaper，本进程运行 generator 并汇总结果

帧格式（qos.shaper.HEADER）：magic(u32) seq(u32) dst_port(u32) payload_len(u32) + payload
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qos.shaper import (  # noqa: E402
    HEADER, FRAME_MAGIC, TokenBucket, build_frame, is_port_free, match_rule,
    recv_exact,
)

HOST = "127.0.0.1"


def _load_plan(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):
        return data
    raise ValueError("计划文件需为 entries 数组或含 entries 的对象")


def _payload(seq: int, size: int) -> bytes:
    # 确定性整数负载：循环写入 seq 的大端字节，保证可重放。
    base = seq.to_bytes(4, "big")
    return (base * ((size // 4) + 1))[:size]


def run_sink(port: int, out_path: str, ready_file: str | None = None) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, port))
        srv.listen(1)
        if ready_file:
            Path(ready_file).write_text("ready", encoding="utf-8")
        conn, _ = srv.accept()
        with conn:
            total = 0
            frames = 0
            while True:
                header = recv_exact(conn, HEADER.size)
                if header is None:
                    break
                magic, _seq, _port, payload_len = HEADER.unpack(header)  # type: ignore[union-attr]
                if magic != FRAME_MAGIC:
                    raise ValueError(f"非法帧魔数 {magic:#x}")
                if payload_len:
                    recv_exact(conn, payload_len)
                total += payload_len
                frames += 1
    Path(out_path).write_text(json.dumps(
        {"port": port, "frames": frames, "total_bytes": total}, indent=2),
        encoding="utf-8")


def run_shaper(listen_port: int, sink_port: int, plan_path: str,
               obs_path: str, ready_file: str | None = None) -> None:
    entries = _load_plan(plan_path)
    buckets: dict[str, TokenBucket] = {}
    observations: list[dict] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, listen_port))
        srv.listen(1)
        if ready_file:
            Path(ready_file).write_text("ready", encoding="utf-8")
        gen_conn, _ = srv.accept()
        with gen_conn, socket.create_connection((HOST, sink_port)) as sink_conn:
            start = time.monotonic_ns()
            while True:
                frame = _read_frame_or_none(gen_conn)
                if frame is None:
                    break
                seq, dst_port, payload = frame
                now_us = (time.monotonic_ns() - start) // 1_000
                entry = match_rule(entries, dst_port)
                rid = entry["rule_id"] if entry else None
                delay_us = 0
                if entry is not None:
                    shaping = entry["shaping"]
                    bucket = buckets.setdefault(
                        rid, TokenBucket(shaping["rate_kbps"],
                                         shaping["burst_bytes"]))
                    delay_us, _ = bucket.account(now_us, len(payload))
                    if delay_us > 0:
                        time.sleep(delay_us / 1_000_000.0)
                sink_conn.sendall(build_frame(seq, dst_port, payload))
                observations.append({
                    "timestamp_us": now_us + delay_us,
                    "seq": seq,
                    "dst_port": dst_port,
                    "rule_id": rid,
                    "sent_bytes": len(payload),
                    "delay_us": delay_us,
                    "dropped": False,
                })
    Path(obs_path).write_text(
        json.dumps(observations, ensure_ascii=False, indent=2),
        encoding="utf-8")


def _read_frame_or_none(conn: socket.socket):
    header = recv_exact(conn, HEADER.size)
    if header is None:
        return None
    magic, seq, dst_port, payload_len = HEADER.unpack(header)
    if magic != FRAME_MAGIC:
        raise ValueError(f"非法帧魔数 {magic:#x}")
    payload = recv_exact(conn, payload_len) if payload_len else b""
    return seq, dst_port, payload


def run_generator(shaper_port: int, count: int, size: int,
                  dst_ports: list[int], gap_ms: int) -> None:
    with socket.create_connection((HOST, shaper_port), timeout=10) as conn:
        for seq in range(count):
            dst_port = dst_ports[seq % len(dst_ports)]
            conn.sendall(build_frame(seq, dst_port, _payload(seq, size)))
            if gap_ms:
                time.sleep(gap_ms / 1000.0)


def run_all(args: argparse.Namespace) -> dict:
    for port in (args.shaper_port, args.sink_port):
        if not is_port_free(HOST, port):
            raise SystemExit(f"端口 {HOST}:{port} 已被占用，请更换高位端口")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sink_out = out_dir / "sink_report.json"
    obs_out = out_dir / "observations.json"
    sink_ready = out_dir / ".sink.ready"
    shaper_ready = out_dir / ".shaper.ready"
    for marker in (sink_ready, shaper_ready):
        marker.unlink(missing_ok=True)
    py = sys.executable
    sink_proc = subprocess.Popen(
        [py, str(Path(__file__).resolve()), "sink",
         "--port", str(args.sink_port), "--out", str(sink_out),
         "--ready-file", str(sink_ready)])
    try:
        _wait_file(sink_ready)  # sink 完成 listen，且未消耗其唯一 accept 槽位
        shaper_proc = subprocess.Popen(
            [py, str(Path(__file__).resolve()), "shaper",
             "--listen-port", str(args.shaper_port),
             "--sink-port", str(args.sink_port),
             "--plan", args.plan, "--obs", str(obs_out),
             "--ready-file", str(shaper_ready)])
        _wait_file(shaper_ready)  # shaper 完成 listen 后发生器才连接
        run_generator(args.shaper_port, args.count, args.size,
                      args.ports, args.gap_ms)
    finally:
        shaper_proc.wait(timeout=30)
        sink_proc.wait(timeout=30)
    sink_report = json.loads(sink_out.read_text(encoding="utf-8"))
    observations = json.loads(obs_out.read_text(encoding="utf-8"))
    summary = {
        "mode": "real-loopback-tcp",
        "host": HOST,
        "shaper_port": args.shaper_port,
        "sink_port": args.sink_port,
        "packets_sent": args.count,
        "packet_size": args.size,
        "sink": sink_report,
        "observation_count": len(observations),
        "max_delay_us": max((o["delay_us"] for o in observations), default=0),
    }
    (out_dir / "traffic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _wait_file(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise TimeoutError(f"等待就绪文件超时: {path}")


def _wait_port(port: int, timeout: float = 5.0) -> None:
    """轮询直到端口完成 listen（connect 被接受即说明 backlog 已就绪）。"""
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.5):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f"等待 {HOST}:{port} 可连接超时: {last_error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回环流量整形模拟脚本")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sink = sub.add_parser("sink")
    p_sink.add_argument("--port", type=int, required=True)
    p_sink.add_argument("--out", required=True)
    p_sink.add_argument("--ready-file")
    p_sink.set_defaults(func=lambda a: run_sink(a.port, a.out, a.ready_file))

    p_sh = sub.add_parser("shaper")
    p_sh.add_argument("--listen-port", type=int, required=True)
    p_sh.add_argument("--sink-port", type=int, required=True)
    p_sh.add_argument("--plan", required=True)
    p_sh.add_argument("--obs", required=True)
    p_sh.add_argument("--ready-file")
    p_sh.set_defaults(func=lambda a: run_shaper(
        a.listen_port, a.sink_port, a.plan, a.obs, a.ready_file))

    p_gen = sub.add_parser("generator")
    p_gen.add_argument("--shaper-port", type=int, required=True)
    p_gen.add_argument("--count", type=int, default=20)
    p_gen.add_argument("--size", type=int, default=1024)
    p_gen.add_argument("--ports", type=int, nargs="+", default=[14000])
    p_gen.add_argument("--gap-ms", type=int, default=0)
    p_gen.set_defaults(func=lambda a: run_generator(
        a.shaper_port, a.count, a.size, a.ports, a.gap_ms))

    p_run = sub.add_parser("run")
    p_run.add_argument("--plan", required=True)
    p_run.add_argument("--shaper-port", type=int, default=49100)
    p_run.add_argument("--sink-port", type=int, default=49101)
    p_run.add_argument("--count", type=int, default=20)
    p_run.add_argument("--size", type=int, default=1024)
    p_run.add_argument("--ports", type=int, nargs="+", default=[14000, 14100])
    p_run.add_argument("--gap-ms", type=int, default=20)
    p_run.add_argument("--out-dir", default="data/traffic")
    p_run.set_defaults(func=run_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
