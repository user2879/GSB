"""可重复演示脚本（确定性时钟 + 本地样本，无任何线上访问）。

异常注入：
  --inject zone-corrupt        区文件损坏      -> E_ZONE
  --inject cache-version       缓存版本不匹配  -> E_VER
  --inject upstream-txn        上游事务标识错  -> E_TXN

常规演示覆盖：三版本归一/降级/补齐、TTL 递减与过期、负应答缓存、
LRU 逐出、TCP 上游转发与链校验、版本整体替换、SQLite 快照恢复校验。
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dns_workbench.app import DnsWorkbench  # noqa: E402
from dns_workbench.errors import WorkbenchError  # noqa: E402
from dns_workbench.records import export_view  # noqa: E402
from dns_workbench.wire import (  # noqa: E402
    TYPE_A,
    DnsMessage,
    Question,
    encode_message,
    encode_tcp_frame,
    parse_message,
)

SAMPLES = ROOT / "dns_workbench" / "samples"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def start_mock_upstream(*, bad_txn: bool = False, u1: bool = False, port: int = 5354) -> subprocess.Popen:
    command = [sys.executable, str(ROOT / "scripts" / "mock_upstream.py"), "--port", str(port)]
    if bad_txn:
        command.append("--bad-txn")
    if u1:
        command.append("--u1")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc
        except OSError:
            time.sleep(0.05)
    proc.terminate()
    raise RuntimeError("模拟上游未能在 3 秒内启动")


def tcp_query(port: int, qname: str, qtype: int = TYPE_A, txn_id: int = 0x1234) -> DnsMessage:
    request = DnsMessage(txn_id, 0x0100, [Question(qname, qtype)])
    with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
        sock.sendall(encode_tcp_frame(request))
        header = _recv(sock, 2)
        (length,) = struct.unpack("!H", header)
        return parse_message(_recv(sock, length))


def _recv(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("连接提前关闭")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def run(inject: str | None, data_dir: Path) -> int:
    workbench = DnsWorkbench(sample_dir=SAMPLES, data_dir=data_dir, deterministic=True)
    try:
        if inject == "zone-corrupt":
            return demo_zone_corrupt(workbench)
        if inject == "cache-version":
            return demo_cache_version(workbench)
        if inject == "upstream-txn":
            return demo_upstream_txn(workbench)
        return demo_full(workbench, data_dir)
    finally:
        workbench.close()


def demo_zone_corrupt(workbench: DnsWorkbench) -> int:
    section("异常注入：区文件损坏（期望 E_ZONE）")
    _result, error = workbench.safe(workbench.load_zone_file, SAMPLES / "zone_corrupt.json")
    return _report_error(error, "E_ZONE")


def demo_cache_version(workbench: DnsWorkbench) -> int:
    section("异常注入：缓存记录版本不匹配（期望 E_VER）")
    workbench.load_zone_file(SAMPLES / "zone.json")
    _result, error = workbench.safe(workbench.load_cache_files, [SAMPLES / "cache_bad_version.json"])
    return _report_error(error, "E_VER")


def demo_upstream_txn(workbench: DnsWorkbench) -> int:
    section("异常注入：上游应答事务标识错误（期望 E_TXN）")
    mock = start_mock_upstream(bad_txn=True, port=workbench.upstream_client.port)
    try:
        from dns_workbench.upstream import TcpUpstreamClient

        client = TcpUpstreamClient(port=workbench.upstream_client.port, retries=0)
        request = DnsMessage(0x4242, 0x0100, [Question("www.example.test", TYPE_A)])
        try:
            client.forward(request)
            print("FAIL: 未抛出 E_TXN")
            return 1
        except WorkbenchError as error:
            workbench.logger.error("%s: %s", error.code, error.message, extra={"data": error.to_dict()})
            return _report_error(error, "E_TXN")
    finally:
        mock.terminate()
        mock.wait(timeout=3)


def _report_error(error: Exception | None, expected: str) -> int:
    if error is None:
        print(f"FAIL: 期望 {expected} 但未发生异常")
        return 1
    assert isinstance(error, WorkbenchError)
    print(json.dumps(error.to_dict(), ensure_ascii=False, indent=2))
    ok = error.code == expected
    print("结果:", "PASS" if ok else f"FAIL（实际 {error.code}）")
    return 0 if ok else 1


def demo_full(workbench: DnsWorkbench, data_dir: Path) -> int:
    section("1. 加载区文件与三版本缓存样本并归一")
    workbench.bootstrap(
        zone_path=SAMPLES / "zone.json",
        cache_paths=[SAMPLES / "cache_v1.json", SAMPLES / "cache_v2.json", SAMPLES / "cache_v3.json"],
    )
    generation = workbench.ledger.current()
    print("代数", generation.generation, "区版本", generation.zone_version, "格式集合", generation.format_versions)
    stored = workbench.ledger.cache().view()
    for item in stored:
        entry = item.entry
        v1 = export_view(entry, "v1", remaining_ttl=item.remaining_ttl_seconds(workbench.clock.now_ms()))
        v3 = export_view(entry, "v3", remaining_ttl=item.remaining_ttl_seconds(workbench.clock.now_ms()))
        print(
            f"  {entry.name:28s} type={entry.qtype} from={entry.source_version} "
            f"strip={v1.stripped_fields} fill={v3.filled_fields} digest={entry.content_digest[:10]}"
        )

    section("2. 视图降级/补齐的逐字节一致性")
    sample_entry = stored[0].entry
    v1 = export_view(sample_entry, "v1", remaining_ttl=45)
    v3 = export_view(sample_entry, "v3", remaining_ttl=45)
    assert v1.record["name"] == v3.record["name"]
    assert v1.record["qtype"] == v3.record["qtype"]
    assert v1.record["ttl"] == v3.record["ttl"] == 45
    print("v1 视图:", json.dumps(v1.record, ensure_ascii=False))
    print("v3 视图:", json.dumps(v3.record, ensure_ascii=False))
    print("剥离字段:", v1.stripped_fields, "补齐字段:", v3.filled_fields)

    section("3. TTL 单调时钟递减、过期回收与负应答缓存")
    key = ("legacy.example.test", TYPE_A)
    hit = workbench.ledger.cache().get(key)
    assert hit is not None
    workbench.record_hit_event(*key, hit.remaining_ttl_s)
    print("首次命中剩余 TTL(s):", hit.remaining_ttl_s, "hits:", hit.hits)
    workbench.clock.advance_ms(30_000)
    hit2 = workbench.ledger.cache().get(key)
    print("推进 30s 后剩余 TTL(s):", hit2.remaining_ttl_s)
    workbench.clock.advance_ms(120_000)
    workbench.ledger.cache().reap_expired()
    print("再推进 120s 后条目仍存在:", key in workbench.ledger.cache())
    negative_ttl = workbench.ledger.cache().negative_ttl(workbench.resolver.zone.soa_minimum())
    print("负应答 TTL = min(SOA minimum=60, cap=300) =", negative_ttl)

    section("4. LRU 逐出（小容量，稳定规则）")
    from dns_workbench.cache import TtlCache
    from dns_workbench.records import CacheEntry, canonical_digest

    small = TtlCache(capacity=3, clock=workbench.clock)
    for index in range(5):
        entry = CacheEntry(f"lru{index}.test", TYPE_A, f"A 10.1.0.{index}", 60, "v1", "")
        entry.content_digest = canonical_digest(entry.normalized_payload())
        small.put(entry)
        workbench.clock.advance_ms(10)
    print("容量 3 写入 5 条后保留:", [stored.entry.name for stored in small.view()])
    print("逐出记录:", [(ev.key[0], ev.reason) for ev in small.evictions])

    section("5. 回环 TCP 上游转发与应答链校验（u2 默认）")
    mock = start_mock_upstream(port=workbench.upstream_client.port)
    try:
        workbench.bootstrap(
            zone_path=SAMPLES / "zone.json",
            cache_paths=[SAMPLES / "cache_v1.json", SAMPLES / "cache_v2.json", SAMPLES / "cache_v3.json"],
        )
        from dns_workbench.server import DnsServerPair

        server_pair = DnsServerPair(workbench.engine, udp_port=5353, tcp_port=5353)
        server_pair.start()
        try:
            # forward.example.test 仅存在于区与上游，未被缓存样本预热 -> 触发权威解析与 TCP 转发
            reply = tcp_query(5353, "forward.example.test")
            print("应答 rcode:", reply.rcode, "answers:", len(reply.answers), "AA:", reply.aa)
            report = workbench.engine.reports[-1]
            print("来源:", report.source, "上游格式:", report.upstream_format, "一致性:", report.consistent, "尝试次数:", report.attempts)
            cached = workbench.ledger.cache().get(("forward.example.test", TYPE_A))
            print("上游应答已入缓存:", cached is not None)
        finally:
            server_pair.stop()
    finally:
        mock.terminate()
        mock.wait(timeout=3)

    section("6. 区版本升级后的整体替换与陈旧防护")
    upgraded = json.loads((SAMPLES / "zone.json").read_text(encoding="utf-8"))
    upgraded["zone_version"] = 2
    upgraded_path = data_dir / "zone_v2.json"
    upgraded_path.write_text(json.dumps(upgraded), encoding="utf-8")
    before = len(workbench.ledger.history)
    workbench.reload_zone(upgraded_path)
    diff = workbench.ledger.last_diff
    print("旧代数标记陈旧:", workbench.ledger.history[-1].stale)
    print("整体替换差异:", json.dumps(diff.to_dict(), ensure_ascii=False))

    section("7. SQLite 快照保存与恢复重放校验")
    snapshot_id = workbench.save_snapshot()
    report = workbench.verify_snapshot(snapshot_id)
    print("快照 ID:", snapshot_id, "匹配:", report.matched, "格式集合变化:", report.format_versions_changed)
    # 制造不一致：写入新条目后重放校验，应稳定列出差异并交由整体替换
    new_entry = CacheEntry("newly.added.test", TYPE_A, "A 10.9.9.9", 30, "v2", "")
    new_entry.rcode, new_entry.aa, new_entry.source = 0, True, "authoritative"
    new_entry.content_digest = canonical_digest(new_entry.normalized_payload())
    workbench.ledger.cache().put(new_entry)
    report2 = workbench.verify_snapshot(snapshot_id)
    print("失效后校验 matched:", report2.matched, "差异:", json.dumps(report2.diff, ensure_ascii=False))
    print("快照列表:", [(item["snapshot_id"], item["format_versions"]) for item in workbench.snapshots()])
    print("\n全部演示完成。结构化日志:", workbench.data_dir / "workbench.jsonl")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DNS 诊断工作台可重复演示")
    parser.add_argument("--inject", choices=["zone-corrupt", "cache-version", "upstream-txn"])
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "run"))
    args = parser.parse_args()
    return run(args.inject, Path(args.data_dir))


if __name__ == "__main__":
    raise SystemExit(main())
