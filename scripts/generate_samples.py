"""生成确定性样本：通道声明 + 发送脚本（乱序/重复/缺失片/人工时钟偏置）。

不访问网络，仅写本地 JSON。固定随机种子保证逐字节可复现。
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

CHANNELS = [
    {
        "channel_id": "tun-alpha",
        "host": "127.0.0.1",
        "port": 39101,
        "payload_limit": 512,
        "window_ms": 120,
        "max_inflight_ms": 10,
        "slot_count": 2,
        "min_estimate_samples": 3,
    },
    {
        "channel_id": "tun-beta",
        "host": "127.0.0.1",
        "port": 39102,
        "payload_limit": 256,
        "window_ms": 90,
        "max_inflight_ms": 500,
        "slot_count": 1,
        "min_estimate_samples": 2,
    },
    {
        "channel_id": "tun-gamma",
        "host": "127.0.0.1",
        "port": 39103,
        "payload_limit": 256,
        "window_ms": 100,
        "max_inflight_ms": 1000,
        "slot_count": 4,
        "min_estimate_samples": 3,
    },
]

# 每组：(负载字节数, 片大小, 发送间隔ms, 到达间隔ms, 人工偏置变化)
# 偏置序列刻意先稳定后跳变，驱动同一组在估计版本间发生窗口翻转。
GROUP_PLAN_ALPHA = [
    (900, 300, 40, 55, 10),
    (720, 240, 40, 55, 10),
    (600, 300, 40, 55, 10),
    (480, 240, 40, 70, 26),   # 偏置跳变，估计中位数漂移
    (540, 270, 40, 70, 26),
    (660, 330, 160, 185, 26), # 大间隔，落入新窗口
    (360, 180, 40, 55, 26),
]
# 发送间隔统一 30ms（< window_ms=90），多组并发落在同一窗口；
# slot_count=1 使后到组稳定触发 SLOT_DROPPED。
GROUP_PLAN_BETA = [
    (400, 200, 30, 45, -8),
    (300, 150, 10, 45, -8),
    (250, 125, 10, 45, 12),
    (350, 175, 10, 45, 12),
]


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _split(payload: bytes, frag_size: int) -> list[bytes]:
    return [payload[i : i + frag_size] for i in range(0, len(payload), frag_size)]


def build_trace(channel: dict, plan: list, rng: random.Random) -> dict:
    groups = []
    send_cursor = 1_000
    arrive_cursor = 5_000
    for gseq, (size, frag_size, send_gap, arrive_gap, bias) in enumerate(plan):
        payload = bytes(rng.randrange(256) for _ in range(size))
        pieces = _split(payload, frag_size)
        send_cursor += send_gap
        arrive_cursor += arrive_gap
        # 人工时钟偏置：arrive = send + bias + 固定网络项(4000)
        arrive_ms = send_cursor + 4000 + bias
        if groups:
            arrive_ms = max(arrive_ms, groups[-1]["arrive_ms"] + 1)

        frags = []
        offset = 0
        for piece in pieces:
            frags.append(
                {
                    "frag_offset": offset,
                    "frag_length": len(piece),
                    "payload_b64": base64.b64encode(piece).decode("ascii"),
                }
            )
            offset += len(piece)

        # 乱序：交换相邻片顺序（>=3 片时）
        if len(frags) >= 3:
            frags[1], frags[2] = frags[2], frags[1]
        # 重复：复制首片（去重规则验证）
        frags.append(dict(frags[0]))

        group = {
            "group_seq": gseq,
            "send_ms": send_cursor,
            "arrive_ms": arrive_ms,
            "total_frags": len(pieces),
            "expected_digest": _digest(payload),
            "fragments": frags,
        }
        groups.append(group)

    # 缺失片：最后一组删掉一个非首片（仅在 alpha 通道演示 INCOMPLETE）
    if channel["channel_id"] == "tun-alpha":
        last = groups[-1]
        unique = {f["frag_offset"]: f for f in last["fragments"]}
        drop_offset = sorted(unique)[-1]
        last["fragments"] = [f for f in last["fragments"] if f["frag_offset"] != drop_offset]
        last["note"] = f"注入缺失片 offset={drop_offset}"

    return {"channel_id": channel["channel_id"], "groups": groups}


# gamma：人工构造的偏差翻转演示。
# 网络固定项 4000ms；前 3 组偏差 +10，后 3 组偏差 +20。
# v1 中位数=4010，v2 中位数=4020，组 2/3/4 在两版本间发生窗口归属翻转。
GAMMA_GROUP_COUNT = 8


def build_gamma_trace(channel: dict, rng: random.Random) -> dict:
    # 受控两段偏差：前 3 组 skew=4010，后 5 组 skew=4030。
    # 下中位数在第 7 个样本干净翻转一次（v1=4010 -> v2=4030），
    # 固定锚定窗口下边界组发生窗口内外/序号翻转。
    groups = []
    send_base = 2_000
    gap = 40
    skew_table = [4010 if g < 3 else 4030 for g in range(GAMMA_GROUP_COUNT)]
    for gseq in range(GAMMA_GROUP_COUNT):
        send_ms = send_base + gseq * gap
        arrive_ms = send_ms + skew_table[gseq]
        payload = bytes(rng.randrange(256) for _ in range(128))
        pieces = _split(payload, 64)
        frags = []
        off = 0
        for piece in pieces:
            frags.append(
                {
                    "frag_offset": off,
                    "frag_length": len(piece),
                    "payload_b64": base64.b64encode(piece).decode("ascii"),
                }
            )
            off += len(piece)
        # 乱序：两片组交换顺序
        frags.reverse()
        groups.append(
            {
                "group_seq": gseq,
                "send_ms": send_ms,
                "arrive_ms": arrive_ms,
                "total_frags": len(pieces),
                "expected_digest": _digest(payload),
                "fragments": frags,
            }
        )
    return {"channel_id": channel["channel_id"], "groups": groups}


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    rng = random.Random(20261121)
    config = {"min_estimate_samples": 3, "channels": CHANNELS}
    trace = {
        "channels": [
            build_trace(CHANNELS[0], GROUP_PLAN_ALPHA, rng),
            build_trace(CHANNELS[1], GROUP_PLAN_BETA, rng),
            build_gamma_trace(CHANNELS[2], rng),
        ]
    }
    (SAMPLES / "channels.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (SAMPLES / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {SAMPLES/'channels.json'}")
    print(f"wrote {SAMPLES/'trace.json'}")


if __name__ == "__main__":
    main()
