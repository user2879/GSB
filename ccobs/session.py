"""客户端会话: 连接回环对端, 收集脚本化应答并生成确定性事件流。

逻辑单调时钟约定(毫秒):
  * ack / delay / reorder / dup_ack / gap 应答: 时钟推进脚本 delay_ms
  * loss 条目: 客户端按固定逻辑超时 LOSS_TIMEOUT_MS 推进时钟
  * gap 条目: 对端跳过 skip 条脚本条目不应答, 客户端不为其生成事件,
    应答流因此出现序号缺口, 由账本以 E_SCRIPT 判定
时钟只依赖脚本内容, 与真实墙上时间无关, 因此任意次执行结果一致。
"""
import socket

from .ledger import (KIND_ACK, KIND_DUP_ACK, KIND_TIMEOUT, Event, Ledger)
from .peer import PeerServer, _recv_msg, _send_msg

LOSS_TIMEOUT_MS = 200
IDLE_TIMEOUT_S = 1.0


def run_session(config, script):
    """执行一次回环会话, 返回按事件序号排序的确定性事件列表。"""
    peer = PeerServer(script, config.peer_host, config.peer_port)
    port = peer.start()
    replies = []
    try:
        with socket.create_connection((config.peer_host, port), timeout=5.0) as conn:
            for entry in script.entries:
                _send_msg(conn, {"seq": entry.event_seq})
            skipped = _gap_skipped_seqs(script)
            expected = sum(1 for e in script.entries
                           if e.action != "loss" and e.event_seq not in skipped)
            conn.settimeout(IDLE_TIMEOUT_S)
            arrival = 0
            while len(replies) < expected:
                try:
                    msg = _recv_msg(conn)
                except (socket.timeout, TimeoutError):
                    break
                if msg is None:
                    break
                replies.append((arrival, msg))
                arrival += 1
    finally:
        peer.wait_done(timeout=5.0)

    reply_by_seq = {}
    for arrival, msg in replies:
        reply_by_seq.setdefault(msg["event_seq"], []).append((arrival, msg))

    events = []
    clock_ms = 0
    for entry in script.entries:
        if entry.action == "loss":
            clock_ms += LOSS_TIMEOUT_MS
            events.append(Event(event_seq=entry.event_seq, kind=KIND_TIMEOUT,
                                rtt_ms=0, clock_ms=clock_ms, arrival_order=-1))
            continue
        clock_ms += entry.delay_ms
        matched = reply_by_seq.get(entry.event_seq)
        if matched:
            arrival, msg = matched.pop(0)
            kind = KIND_DUP_ACK if msg["type"] == "dup_ack" else KIND_ACK
            events.append(Event(event_seq=entry.event_seq, kind=kind,
                                rtt_ms=entry.delay_ms, clock_ms=clock_ms,
                                arrival_order=arrival))
    # gap 故障注入产生错位序号的应答: 以应答自身序号登记为额外事件
    for seq, queued in reply_by_seq.items():
        for arrival, msg in queued:
            events.append(Event(event_seq=seq, kind=KIND_ACK,
                                rtt_ms=msg.get("delay_ms", 0),
                                clock_ms=clock_ms, arrival_order=arrival))
    events.sort(key=lambda e: (e.event_seq, e.arrival_order))
    return events


def _gap_skipped_seqs(script):
    """脚本中因 gap 注入而被对端跳过的事件序号集合(含 gap 条目自身)。"""
    skipped = set()
    skip_remaining = 0
    for entry in script.entries:
        if entry.action == "gap":
            skip_remaining = entry.skip - 1
            skipped.add(entry.event_seq)
            continue
        if skip_remaining > 0:
            skip_remaining -= 1
            skipped.add(entry.event_seq)
    return skipped


def build_ledger(config, config_version, events):
    ledger = Ledger(config, config_version)
    for event in events:
        ledger.ingest_event(event)
    return ledger


def execute(config, script, config_version=1):
    """执行会话并推进账本, 返回 (ledger, events)。"""
    events = run_session(config, script)
    return build_ledger(config, config_version, events), events
