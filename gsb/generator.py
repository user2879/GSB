"""固定种子程序化样本生成器。

同一逻辑事件流以三种承载写出：
  events.jsonl  自有 JSON 行（按场景混合 schema_version）
  events.bin    紧凑二进制帧（4 字节小端长度前缀 + 帧体，轮换字节序/索引宽度/位图）
  legacy.jsonl  旧版 JSON（数字 peer、毫秒时间戳、字段裁剪）
"""

from __future__ import annotations

import json
import os

import numpy as np

from .formats import binary as fmt_binary
from .formats import jsonl as fmt_jsonl
from .formats import legacy as fmt_legacy
from .model import Event
from .pipeline import write_binary_frames

SCENARIOS = ("normal", "mixed", "upgrade", "conflict", "kill")
BASE_SEED = 250


def _rng_for(scenario: str) -> np.random.Generator:
    seed = BASE_SEED * 1000 + sum(ord(c) for c in scenario)
    return np.random.Generator(np.random.PCG64(seed))


def _gen_events(scenario: str) -> list[Event]:
    rng = _rng_for(scenario)
    peers = ["peer-1", "peer-2", "peer-3"]
    n_ticks = {"normal": 10, "mixed": 6, "upgrade": 6, "conflict": 5, "kill": 12}[scenario]
    seq = {p: 0 for p in peers}
    entity_counter = 0
    live: list[str] = []
    events: list[Event] = []

    def emit(tick: int, peer: str, op_type: str, payload: dict) -> None:
        nonlocal entity_counter
        events.append(Event(
            tick=tick, peer_id=peer, local_seq=seq[peer], op_type=op_type,
            payload=payload, base_snapshot_version=tick, source_format="",
        ))
        seq[peer] += 1

    for tick in range(n_ticks):
        for peer in sorted(peers):
            n_ops = int(rng.integers(1, 3))
            for _ in range(n_ops):
                roll = float(rng.random())
                if not live or roll < 0.35:
                    entity = f"e{entity_counter}"
                    entity_counter += 1
                    emit(tick, peer, "spawn", {
                        "entity": entity,
                        "kind": str(rng.choice(["knight", "archer", "golem"])),
                        "x": int(rng.integers(0, 20)), "y": int(rng.integers(0, 20)),
                    })
                    live.append(entity)
                elif roll < 0.65:
                    emit(tick, peer, "move", {
                        "entity": str(rng.choice(live)),
                        "dx": int(rng.integers(-3, 4)), "dy": int(rng.integers(-3, 4)),
                    })
                elif roll < 0.9:
                    emit(tick, peer, "set_attr", {
                        "entity": str(rng.choice(live)),
                        "field": str(rng.choice(["hp", "mp", "team"])),
                        "value": int(rng.integers(0, 100)),
                    })
                else:
                    victim = str(rng.choice(live))
                    live.remove(victim)
                    emit(tick, peer, "destroy", {"entity": victim})
        if scenario == "conflict" and live:
            # 同 tick 多端写同一实体同一字段，制造确定性并发冲突
            target = live[0]
            emit(tick, "peer-1", "set_attr",
                 {"entity": target, "field": "hp", "value": 10 + tick})
            emit(tick, "peer-2", "set_attr",
                 {"entity": target, "field": "hp", "value": 90 - tick})
            emit(tick, "peer-3", "move", {"entity": target, "dx": 1, "dy": -1})
    return events


def _jsonl_version(scenario: str, idx: int) -> int:
    if scenario == "upgrade":
        return (idx % 3) + 1  # v1/v2/v3 混合，触发迁移表
    return 3


def _binary_bitmap(scenario: str, idx: int) -> int:
    bitmap = fmt_binary.KNOWN_BITS
    if scenario == "upgrade" and idx % 4 == 1:
        bitmap &= ~fmt_binary.BIT_LOCAL_SEQ  # 位图未声明可选字段，由迁移表补齐
    return bitmap


def write_scenario(root: str, scenario: str) -> dict:
    events = _gen_events(scenario)
    out_dir = os.path.join(root, scenario)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "events.jsonl"), "w", encoding="utf-8") as fh:
        for idx, event in enumerate(events):
            fh.write(fmt_jsonl.dump_line(event, _jsonl_version(scenario, idx)) + "\n")

    frames = []
    for idx, event in enumerate(events):
        endian = fmt_binary.ENDIAN_LITTLE if idx % 2 == 0 else fmt_binary.ENDIAN_BIG
        width = 4 if idx % 3 else 2
        frames.append(fmt_binary.encode_frame(
            event, endian=endian, width=width, bitmap=_binary_bitmap(scenario, idx)))
    write_binary_frames(os.path.join(out_dir, "events.bin"), frames)

    with open(os.path.join(out_dir, "legacy.jsonl"), "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(fmt_legacy.dump_line(event) + "\n")

    manifest = {
        "scenario": scenario,
        "seed": BASE_SEED,
        "event_count": len(events),
        "formats": {
            "jsonl": "events.jsonl",
            "binary": "events.bin",
            "legacy": "legacy.jsonl",
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2)
        fh.write("\n")
    return manifest


def generate_all(root: str) -> dict:
    os.makedirs(root, exist_ok=True)
    return {scenario: write_scenario(root, scenario) for scenario in SCENARIOS}
