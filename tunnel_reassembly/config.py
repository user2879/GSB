"""通道声明与确定性发送脚本的 JSON 加载与强校验。

任一不合法均以 E_CHANNEL 整体拒绝（先收集全部出错字段路径再抛出），
绝不部分加载。config_version / trace_version 为规范化 JSON 的 sha256，
用于快照版本识别与整体替换恢复。
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from .errors import BusinessError, E_CHANNEL
from .model import ChannelSpec


def canonical_digest(obj: object) -> str:
    """对任意 JSON 可序列化对象计算确定性 sha256（sort_keys、无多余空白）。"""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.items.append(f"{path}: {message}")

    def raise_if_any(self) -> None:
        if self.items:
            raise BusinessError(E_CHANNEL, "声明整体拒绝；出错字段：" + " | ".join(self.items))


def load_json(path: str | Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise BusinessError(E_CHANNEL, f"文件不存在: {path}")
    except json.JSONDecodeError as exc:
        raise BusinessError(E_CHANNEL, f"JSON 解析失败: {exc}", path=str(path))


def load_channels(config_obj: dict) -> tuple[list[ChannelSpec], str]:
    """校验并返回通道声明列表与 config_version 摘要。失败整体拒绝。"""
    errs = _Errors()

    channels_raw = config_obj.get("channels")
    if not isinstance(channels_raw, list) or not channels_raw:
        errs.add("channels", "必须是非空数组")
        errs.raise_if_any()

    global_min = config_obj.get("min_estimate_samples", 3)
    if not _is_int(global_min) or global_min <= 0:
        errs.add("min_estimate_samples", "必须是正整数")

    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    specs: list[ChannelSpec] = []

    for idx, ch in enumerate(channels_raw):
        base = f"channels[{idx}]"
        if not isinstance(ch, dict):
            errs.add(base, "必须是对象")
            continue

        channel_id = ch.get("channel_id")
        if not isinstance(channel_id, str) or not channel_id:
            errs.add(f"{base}.channel_id", "必须是非空字符串")
        elif channel_id in seen_ids:
            errs.add(f"{base}.channel_id", f"通道标识重复: {channel_id}")
        else:
            seen_ids.add(channel_id)

        host = ch.get("host")
        if host != "127.0.0.1":
            errs.add(f"{base}.host", "必须是 127.0.0.1 字面量")

        port = ch.get("port")
        if not _is_int(port) or not (1024 <= port <= 65535):
            errs.add(f"{base}.port", "必须是 1024..65535 的整数端口")
        elif port in seen_ports:
            errs.add(f"{base}.port", f"端口全局冲突: {port}")
        else:
            seen_ports.add(port)

        payload_limit = ch.get("payload_limit")
        if not _is_int(payload_limit) or payload_limit <= 0:
            errs.add(f"{base}.payload_limit", "必须是正整数")

        window_ms = ch.get("window_ms")
        if not _is_int(window_ms) or window_ms <= 0:
            errs.add(f"{base}.window_ms", "必须是正整数")

        max_inflight_ms = ch.get("max_inflight_ms")
        if not _is_int(max_inflight_ms) or max_inflight_ms < 0:
            errs.add(f"{base}.max_inflight_ms", "必须是非负整数")

        slot_count = ch.get("slot_count")
        if not _is_int(slot_count) or slot_count < 0:
            errs.add(f"{base}.slot_count", "必须是非负整数")

        min_samples = ch.get("min_estimate_samples", global_min)
        if not _is_int(min_samples) or min_samples <= 0:
            errs.add(f"{base}.min_estimate_samples", "必须是正整数")

        if errs.items:
            continue
        specs.append(
            ChannelSpec(
                channel_id=channel_id,
                host=host,
                port=port,
                payload_limit=payload_limit,
                window_ms=window_ms,
                max_inflight_ms=max_inflight_ms,
                slot_count=slot_count,
                min_estimate_samples=min_samples,
            )
        )

    errs.raise_if_any()
    config_version = canonical_digest(config_obj)
    return specs, config_version


def load_trace(trace_obj: dict, specs: list[ChannelSpec]) -> tuple[dict[str, list[dict]], str]:
    """校验发送脚本并按通道返回原始组字典（含解码后 payload 字节）。

    校验：通道存在、组序号通道内连续、片区间不重叠、片长自洽、
    偏移+长度不超 payload_limit、总片数为正整数。
    """
    errs = _Errors()
    spec_by_id = {s.channel_id: s for s in specs}

    channels_raw = trace_obj.get("channels")
    if not isinstance(channels_raw, list) or not channels_raw:
        errs.add("channels", "必须是非空数组")
        errs.raise_if_any()

    result: dict[str, list[dict]] = {}

    for cidx, ch_block in enumerate(channels_raw):
        cbase = f"channels[{cidx}]"
        if not isinstance(ch_block, dict):
            errs.add(cbase, "必须是对象")
            continue
        channel_id = ch_block.get("channel_id")
        if channel_id not in spec_by_id:
            errs.add(f"{cbase}.channel_id", f"未在通道声明中定义: {channel_id}")
            continue
        spec = spec_by_id[channel_id]

        groups = ch_block.get("groups")
        if not isinstance(groups, list) or not groups:
            errs.add(f"{cbase}.groups", "必须是非空数组")
            continue

        decoded_groups: list[dict] = []
        for gidx, group in enumerate(groups):
            gbase = f"{cbase}.groups[{gidx}]"
            group_seq = group.get("group_seq")
            if not _is_int(group_seq) or group_seq < 0:
                errs.add(f"{gbase}.group_seq", "必须是非负整数")
            elif group_seq != gidx:
                errs.add(f"{gbase}.group_seq", f"组序号必须通道内连续，期望 {gidx} 实际 {group_seq}")

            total_frags = group.get("total_frags")
            if not _is_int(total_frags) or total_frags <= 0:
                errs.add(f"{gbase}.total_frags", "必须是正整数")

            send_ms = group.get("send_ms")
            arrive_ms = group.get("arrive_ms")
            if not _is_int(send_ms) or send_ms < 0:
                errs.add(f"{gbase}.send_ms", "必须是非负整数")
            if not _is_int(arrive_ms) or arrive_ms < 0:
                errs.add(f"{gbase}.arrive_ms", "必须是非负整数")

            digest = group.get("expected_digest")
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                errs.add(f"{gbase}.expected_digest", "必须是 sha256: 前缀字符串")

            frags = group.get("fragments")
            if not isinstance(frags, list) or not frags:
                errs.add(f"{gbase}.fragments", "必须是非空数组")
                continue

            intervals: list[tuple[int, int]] = []
            decoded_frags: list[dict] = []
            for fidx, frag in enumerate(frags):
                fbase = f"{gbase}.fragments[{fidx}]"
                offset = frag.get("frag_offset")
                length = frag.get("frag_length")
                if not _is_int(offset) or offset < 0:
                    errs.add(f"{fbase}.frag_offset", "必须是非负整数")
                if not _is_int(length) or length <= 0:
                    errs.add(f"{fbase}.frag_length", "必须是正整数")
                if not isinstance(frag.get("payload_b64"), str):
                    errs.add(f"{fbase}.payload_b64", "必须是 base64 字符串")
                if errs.items and any(p.startswith(fbase) for p in errs.items):
                    continue
                try:
                    payload = base64.b64decode(frag["payload_b64"], validate=True)
                except (ValueError, base64.binascii.Error):
                    errs.add(f"{fbase}.payload_b64", "base64 解码失败")
                    continue
                if len(payload) != length:
                    errs.add(f"{fbase}.frag_length", f"与实际负载长度不自洽，声明 {length} 实际 {len(payload)}")
                if length > spec.payload_limit:
                    errs.add(f"{fbase}.frag_length", f"超过单包负载上限 {spec.payload_limit}")
                if offset + length > 65535 + spec.payload_limit:
                    errs.add(f"{fbase}.frag_offset", "偏移加长度超过组总长度上限")
                duplicate = False
                for prev_off, prev_len in intervals:
                    if offset == prev_off and length == prev_len:
                        duplicate = True
                        break
                    if offset < prev_off + prev_len and prev_off < offset + length:
                        errs.add(f"{fbase}.frag_offset", f"片区间与偏移 {prev_off} 部分重叠")
                        break
                if not duplicate:
                    intervals.append((offset, length))
                decoded_frags.append(
                    {
                        "frag_offset": offset,
                        "frag_length": length,
                        "payload": payload,
                    }
                )

            if decoded_frags:
                decoded_groups.append(
                    {
                        "group_seq": group_seq,
                        "send_ms": send_ms,
                        "arrive_ms": arrive_ms,
                        "total_frags": total_frags,
                        "expected_digest": digest,
                        "fragments": decoded_frags,
                    }
                )
        result[channel_id] = decoded_groups

    errs.raise_if_any()
    trace_version = canonical_digest(trace_obj)
    return result, trace_version
