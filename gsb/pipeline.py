"""解析管线：三种承载 -> 统一内部事件模型（含序号赋予、迁移、定稿）。"""

from __future__ import annotations

from .formats import binary as fmt_binary
from .formats import jsonl as fmt_jsonl
from .formats import legacy as fmt_legacy
from .migrate import migrate_event
from .model import Event

FORMATS = ("jsonl", "binary", "legacy")


def read_binary_frames(path: str) -> list[bytes]:
    """二进制样本文件：4 字节小端长度前缀 + 帧体。"""
    with open(path, "rb") as fh:
        blob = fh.read()
    frames = []
    offset = 0
    while offset < len(blob):
        n = int.from_bytes(blob[offset:offset + 4], "little")
        frames.append(blob[offset + 4:offset + 4 + n])
        offset += 4 + n
    return frames


def write_binary_frames(path: str, frames: list[bytes]) -> None:
    with open(path, "wb") as fh:
        for frame in frames:
            fh.write(len(frame).to_bytes(4, "little"))
            fh.write(frame)


def parse_stream(scenario_dir: str, fmt: str) -> list[Event]:
    events: list[Event] = []
    if fmt == "jsonl":
        with open(f"{scenario_dir}/events.jsonl", encoding="utf-8") as fh:
            for no, line in enumerate(fh):
                if line.strip():
                    events.append(fmt_jsonl.parse_line(line, no))
    elif fmt == "legacy":
        with open(f"{scenario_dir}/legacy.jsonl", encoding="utf-8") as fh:
            for no, line in enumerate(fh):
                if line.strip():
                    events.append(fmt_legacy.parse_line(line, no))
    elif fmt == "binary":
        for frame in read_binary_frames(f"{scenario_dir}/events.bin"):
            events.append(fmt_binary.decode_frame(frame))
    else:
        raise ValueError(f"未知格式 {fmt}")
    return events


def normalize_stream(events: list[Event], audit_log: list[dict] | None = None) -> list[Event]:
    """赋予缺失本地序号 -> 版本迁移 -> 定稿。迁移幂等由 migrate_event 保证。"""
    counters: dict[str, int] = {}
    out = []
    for event in events:
        if event.local_seq is None:
            seq = counters.get(event.peer_id, 0)
            event.local_seq = seq
        counters[event.peer_id] = event.local_seq + 1
        event, audit = migrate_event(event)
        if audit_log is not None:
            for record in audit:
                record["source_format"] = event.source_format
                record["peer_id"] = event.peer_id
                record["tick"] = event.tick
            audit_log.extend(audit)
        out.append(event.finalize())
    return out


def load_format(scenario_dir: str, fmt: str, audit_log: list[dict] | None = None) -> list[Event]:
    return normalize_stream(parse_stream(scenario_dir, fmt), audit_log)


def load_mixed(scenario_dir: str, audit_log: list[dict] | None = None) -> list[Event]:
    """混格式局：每个逻辑事件按事件下标在三种承载中轮转选取。"""
    per_format = {fmt: parse_stream(scenario_dir, fmt) for fmt in FORMATS}
    count = len(per_format[FORMATS[0]])
    for fmt, stream in per_format.items():
        if len(stream) != count:
            raise ValueError(f"格式 {fmt} 事件数 {len(stream)} 与基准 {count} 不一致")
    # 序号语义按各承载自身流序计算，再按轮转选取。
    normalized = {fmt: normalize_stream(stream, audit_log) for fmt, stream in per_format.items()}
    return [normalized[FORMATS[idx % len(FORMATS)]][idx] for idx in range(count)]


def load_all_formats(scenario_dir: str) -> dict[str, list[Event]]:
    """分别归一化三种承载的完整事件流，供跨格式等价性核对。"""
    return {fmt: load_format(scenario_dir, fmt) for fmt in FORMATS}
