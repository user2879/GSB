"""确定性裁决引擎（纯函数）。

输入同一条通道的逐片事件集合，输出：
  - 偏差估计版本序列（整数中位数、量化误差上界、稳定性说明）
  - 窗口开闭事件序列
  - 估计版本变化引起的窗口内外翻转集合
  - 槽位预算 / 在途偏移约束下的稳定丢弃集合
  - 组重组结局与确定性内容摘要

所有判定只依赖事件内容而不依赖喂入顺序：内部统一按
(group_seq, frag_offset) 排序后处理。
"""

from __future__ import annotations

import hashlib

from .errors import BusinessError, E_ESTIMATE
from .model import (
    ChannelSpec,
    EstimateVersion,
    FlipRecord,
    FragmentEvent,
    GroupOutcome,
    ReassemblyResult,
    WindowEvent,
)

QUANTIZATION_ERROR_MS = 1
UNKNOWN_STATUS = "UNKNOWN"
COMPLETE_STATUS = "COMPLETE"
DIGEST_STATUS = "E_DIGEST"
SLOT_DROP_STATUS = "SLOT_DROPPED"
OFFSET_DROP_STATUS = "OFFSET_DROPPED"
INCOMPLETE_STATUS = "INCOMPLETE"
PENDING_STATUS = "PENDING"


def _sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def integer_median(samples: list[int]) -> int:
    """显式整数中位数规则。

    排序后取“下中位数”：下标 (n-1)//2（零基）。
    偶数样本时偏低位一侧，保证纯整数、无四舍五入歧义、跨进程逐位一致。
    """
    if not samples:
        raise BusinessError(E_ESTIMATE, "空样本无法计算中位数")
    ordered = sorted(samples)
    return ordered[(len(ordered) - 1) // 2]


def build_groups(events: list[FragmentEvent]) -> list[dict]:
    """把逐片事件聚合为组，重复片按 (偏移) 去重，顺序无关。"""
    grouped: dict[int, list[FragmentEvent]] = {}
    for event in events:
        grouped.setdefault(event.group_seq, []).append(event)

    result: list[dict] = []
    for group_seq in sorted(grouped):
        frags = grouped[group_seq]
        total_frags = frags[0].total_frags
        send_ms = min(f.send_ms for f in frags)
        arrive_ms = min(f.arrive_ms for f in frags)

        unique: dict[int, FragmentEvent] = {}
        for frag in sorted(frags, key=lambda f: (f.frag_offset, f.arrive_ms)):
            if frag.total_frags != total_frags:
                from .errors import E_PROTOCOL

                raise BusinessError(
                    E_PROTOCOL,
                    f"组 {group_seq} 内 total_frags 不一致: {frag.total_frags} != {total_frags}",
                )
            unique.setdefault(frag.frag_offset, frag)

        result.append(
            {
                "group_seq": group_seq,
                "total_frags": total_frags,
                "send_ms": send_ms,
                "arrive_ms": arrive_ms,
                "fragments": [unique[off] for off in sorted(unique)],
            }
        )
    return result


def estimate_versions(groups: list[dict], min_samples: int) -> list[EstimateVersion]:
    """按组序号顺序累积样本，生成单调递增的偏差估计版本序列。

    样本规则：每组一个样本，样本值 = 组最早到达时刻 - 组最早发送时刻
    （即 arrive - send；折算公式 adjusted = send + skew）。
    仅当中位数整数值发生变化才新增版本；版本号从 1 开始，
    版本 0 为未定（样本数低于 min_samples）。
    """
    versions: list[EstimateVersion] = [
        EstimateVersion(
            version=0,
            skew_ms=None,
            sample_count=0,
            qerror_ms=QUANTIZATION_ERROR_MS,
        )
    ]
    samples: list[int] = []
    current_skew: int | None = None
    current_version = 0

    for group in groups:
        samples.append(group["arrive_ms"] - group["send_ms"])
        if len(samples) < min_samples:
            versions.append(
                EstimateVersion(
                    version=0,
                    skew_ms=None,
                    sample_count=len(samples),
                    qerror_ms=QUANTIZATION_ERROR_MS,
                )
            )
            continue
        skew = integer_median(samples)
        if skew != current_skew:
            current_skew = skew
            current_version += 1
        versions.append(
            EstimateVersion(
                version=current_version,
                skew_ms=current_skew,
                sample_count=len(samples),
                qerror_ms=QUANTIZATION_ERROR_MS,
            )
        )
    return versions


def estimate_at(versions: list[EstimateVersion], sample_index: int) -> EstimateVersion:
    """返回处理第 sample_index 个组（零基）时刻可用的最新估计版本。"""
    return versions[sample_index]


def stability_notes(versions: list[EstimateVersion]) -> list[str]:
    """估计稳定性的可复核文字说明（同样确定性生成）。"""
    notes = [
        "定点单位=整数毫秒；中位数采用下中位数规则，无浮点参与。",
        f"量化误差上界={QUANTIZATION_ERROR_MS}ms：真实偏差落在 [skew-1, skew+1] 毫秒区间。",
    ]
    real = [v for v in versions if v.skew_ms is not None]
    if len(real) <= 1:
        notes.append("有效估计版本不足两个，无法评估版本间漂移。")
    else:
        drifts = [real[i].skew_ms - real[i - 1].skew_ms for i in range(1, len(real))]
        notes.append(
            f"有效版本 {len(real)} 个；版本间漂移序列={drifts}；"
            f"最大绝对漂移={max(abs(d) for d in drifts)}ms。"
        )
        if all(abs(d) <= QUANTIZATION_ERROR_MS for d in drifts):
            notes.append("全部漂移不超过量化误差上界，估计稳定。")
        else:
            notes.append("存在超过量化误差上界的漂移，估计不稳定并已触发翻转分析。")
    return notes


def assign_windows(groups: list[dict], versions: list[EstimateVersion], window_ms: int) -> dict:
    """固定锚定窗口 + 逐版本重新折算判定。

    窗口基准在首个有效估计版本出现时，以该版本下折算时刻最早的组锚定，
    随后按 window_ms 整数倍向右铺出固定窗口区间，跨版本保持不变。
    每个有效估计版本只重新折算各组的时刻并对照固定区间判定归属，
    因此偏差平移会让边界组在窗口内外/窗口序号之间翻转。

    返回：
      assignments: {version: {group_seq: window_seq|None}}
      windows:     [(window_seq, base_ms, close_ms, anchor_group)]（固定，只一份）
    """
    real_versions: list[EstimateVersion] = []
    for ver in versions:
        if ver.skew_ms is not None and (not real_versions or real_versions[-1].version != ver.version):
            real_versions.append(ver)
    if not real_versions:
        return {"assignments": {}, "windows": []}

    first = real_versions[0]
    anchor_adjusted = sorted(
        ((g["send_ms"] + first.skew_ms, g["group_seq"]) for g in groups)
    )
    origin_ms, anchor_group = anchor_adjusted[0]
    last_adjusted = max(g["send_ms"] + real_versions[-1].skew_ms for g in groups)
    window_count = (last_adjusted - origin_ms) // window_ms + 1
    windows = [
        (seq, origin_ms + (seq - 1) * window_ms, origin_ms + seq * window_ms, anchor_group)
        for seq in range(1, window_count + 1)
    ]

    assignments: dict[int, dict[int, int | None]] = {}
    for ver in real_versions:
        group_to_window: dict[int, int | None] = {}
        for g in groups:
            adjusted = g["send_ms"] + ver.skew_ms
            window_seq = None
            for seq, base_ms, close_ms, _ in windows:
                if base_ms <= adjusted < close_ms:
                    window_seq = seq
                    break
            group_to_window[g["group_seq"]] = window_seq
        assignments[ver.version] = group_to_window
    return {"assignments": assignments, "windows": windows}


def build_flips(groups: list[dict], versions: list[EstimateVersion], plan: dict) -> list[FlipRecord]:
    """相邻估计版本之间，逐组比较窗口归属，生成确定性翻转集合。"""
    flips: list[FlipRecord] = []
    assignments = plan["assignments"]
    # 仅在有效估计版本之间分析翻转；版本 0（未定）到首个有效版本属于初始化，不记翻转。
    versions_seen: list[int] = []
    for ver in versions:
        if ver.skew_ms is not None and ver.version not in versions_seen:
            versions_seen.append(ver.version)

    skew_by_version = {0: None}
    for ver in versions:
        if ver.skew_ms is not None:
            skew_by_version[ver.version] = ver.skew_ms

    for group in groups:
        gseq = group["group_seq"]
        for idx in range(1, len(versions_seen)):
            from_ver = versions_seen[idx - 1]
            to_ver = versions_seen[idx]
            from_win = assignments.get(from_ver, {}).get(gseq)
            to_win = assignments[to_ver].get(gseq)
            if from_win == to_win:
                continue
            if from_win is None and to_win is not None:
                kind = "OUT_TO_IN"
            elif from_win is not None and to_win is None:
                kind = "IN_TO_OUT"
            else:
                kind = "WINDOW_CHANGE"
            from_adj = group["send_ms"] + skew_by_version[from_ver]
            to_adj = group["send_ms"] + skew_by_version[to_ver]
            flips.append(
                FlipRecord(
                    channel_id=group["fragments"][0].channel_id,
                    group_seq=gseq,
                    from_version=from_ver,
                    to_version=to_ver,
                    from_adjusted_ms=from_adj,
                    to_adjusted_ms=to_adj,
                    from_window=from_win,
                    to_window=to_win,
                    flip_kind=kind,
                )
            )
    flips.sort(key=lambda f: (f.group_seq, f.from_version))
    return flips


def build_window_events(channel_id: str, plan: dict) -> list[WindowEvent]:
    """把固定锚定窗口的开闭轨迹展开为有序账本事件。

    窗口区间跨估计版本固定，事件记录锚定所用的首个估计版本；
    版本变化只改变组归属（见 flips），不改变窗口区间本身。
    """
    anchor_version = min(plan["assignments"]) if plan["assignments"] else 0
    events: list[WindowEvent] = []
    for window_seq, base_ms, close_ms, anchor in plan["windows"]:
        events.append(
            WindowEvent(channel_id, window_seq, "OPEN", anchor_version, base_ms, None, anchor)
        )
        events.append(
            WindowEvent(channel_id, window_seq, "CLOSE", anchor_version, base_ms, close_ms, anchor)
        )
    return events


def adjudicate(
    spec: ChannelSpec,
    events: list[FragmentEvent],
    expected_digests: dict[int, str] | None = None,
) -> ReassemblyResult:
    """单通道完整裁决。事件喂入顺序不影响结果。"""
    expected_digests = expected_digests or {}
    ordered_events = sorted(events, key=lambda e: (e.group_seq, e.frag_offset))
    groups = build_groups(ordered_events)
    versions = estimate_versions(groups, spec.min_estimate_samples)
    plan = assign_windows(groups, versions, spec.window_ms)
    flips = build_flips(groups, versions, plan)
    window_events = build_window_events(spec.channel_id, plan)

    latest_version = max((v.version for v in versions if v.skew_ms is not None), default=0)
    final_assign = plan["assignments"].get(latest_version, {})
    base_map: dict[int, int] = {
        window_seq: base_ms for window_seq, base_ms, _c, _a in plan["windows"]
    }

    outcomes = _adjudicate_groups(
        spec, groups, versions, latest_version, final_assign, base_map, expected_digests
    )

    return ReassemblyResult(
        channel_id=spec.channel_id,
        estimates=versions,
        window_events=window_events,
        flips=flips,
        outcomes=outcomes,
        fragments=ordered_events,
    )


def _reassemble(group: dict) -> tuple[str | None, bytes | None]:
    """按偏移升序拼装并计算摘要；区间不连续时返回 None。"""
    fragments = group["fragments"]
    expected_offsets: list[int] = []
    cursor = 0
    for frag in fragments:
        if frag.frag_offset != cursor:
            return None, None
        expected_offsets.append(cursor)
        cursor += frag.frag_length
    payload = b"".join(frag.payload for frag in fragments)
    return _sha256_hex(payload), payload


def _adjudicate_groups(
    spec: ChannelSpec,
    groups: list[dict],
    versions: list[EstimateVersion],
    latest_version: int,
    final_assign: dict[int, int | None],
    base_map: dict[int, int],
    expected_digests: dict[int, str],
) -> list[GroupOutcome]:
    outcomes: list[GroupOutcome] = []
    active: dict[int, dict] = {}
    current_window: int | None = None

    for idx, group in enumerate(groups):
        gseq = group["group_seq"]
        effective = estimate_at(versions, idx)
        if effective.skew_ms is None:
            outcomes.append(
                GroupOutcome(
                    spec.channel_id,
                    gseq,
                    PENDING_STATUS,
                    0,
                    None,
                    None,
                    reason=f"样本数 {effective.sample_count} 低于下限 {spec.min_estimate_samples}，估计未定，拒绝据此开启窗口",
                )
            )
            continue

        adjusted_ms = group["send_ms"] + effective.skew_ms
        assigned_window = final_assign.get(gseq)
        if assigned_window is None:
            outcomes.append(
                GroupOutcome(
                    spec.channel_id,
                    gseq,
                    INCOMPLETE_STATUS,
                    latest_version,
                    None,
                    adjusted_ms,
                    reason="折算时刻落在全部窗口之外",
                )
            )
            continue

        if current_window is None or assigned_window != current_window:
            for prior_seq in sorted(active):
                outcomes.append(_close_group(spec, active[prior_seq], latest_version, expected_digests))
            active.clear()
            current_window = assigned_window
        window_base = base_map[assigned_window]

        if adjusted_ms - window_base > spec.max_inflight_ms:
            outcomes.append(
                GroupOutcome(
                    spec.channel_id,
                    gseq,
                    OFFSET_DROP_STATUS,
                    latest_version,
                    assigned_window,
                    adjusted_ms,
                    reason=(
                        f"在途偏移 {adjusted_ms - window_base}ms 超过上限 "
                        f"{spec.max_inflight_ms}ms"
                    ),
                )
            )
            continue

        if len(active) >= spec.slot_count:
            victim_seq = min(active, key=lambda s: (active[s]["adjusted_ms"], s))
            victim = active.pop(victim_seq)
            outcomes.append(
                GroupOutcome(
                    spec.channel_id,
                    victim_seq,
                    SLOT_DROP_STATUS,
                    latest_version,
                    assigned_window,
                    victim["adjusted_ms"],
                    reason=f"槽位预算 {spec.slot_count} 耗尽，丢弃折算时刻最早的组 {victim_seq}",
                )
            )

        active[gseq] = {"group": group, "adjusted_ms": adjusted_ms, "window": assigned_window}

    for gseq in sorted(active):
        outcomes.append(_close_group(spec, active[gseq], latest_version, expected_digests))

    outcomes.sort(key=lambda item: item.group_seq)
    return outcomes


def _close_group(
    spec: ChannelSpec,
    active_entry: dict,
    estimate_version: int,
    expected_digests: dict[int, str],
) -> GroupOutcome:
    group = active_entry["group"]
    gseq = group["group_seq"]
    adjusted_ms = active_entry["adjusted_ms"]
    window_seq = active_entry["window"]

    if len(group["fragments"]) != group["total_frags"]:
        return GroupOutcome(
            spec.channel_id,
            gseq,
            INCOMPLETE_STATUS,
            estimate_version,
            window_seq,
            adjusted_ms,
            reason=f"窗口关闭时仅到 {len(group['fragments'])}/{group['total_frags']} 片",
        )

    digest, _payload = _reassemble(group)
    if digest is None:
        return GroupOutcome(
            spec.channel_id,
            gseq,
            INCOMPLETE_STATUS,
            estimate_version,
            window_seq,
            adjusted_ms,
            reason="片区间不连续，无法拼装",
        )
    expected = expected_digests.get(gseq)
    if expected is not None and digest != expected:
        return GroupOutcome(
            spec.channel_id,
            gseq,
            DIGEST_STATUS,
            estimate_version,
            window_seq,
            adjusted_ms,
            digest=digest,
            expected_digest=expected,
            reason="重组摘要与发送端登记值不一致",
        )
    return GroupOutcome(
        spec.channel_id,
        gseq,
        COMPLETE_STATUS,
        estimate_version,
        window_seq,
        adjusted_ms,
        digest=digest,
        expected_digest=expected,
    )
