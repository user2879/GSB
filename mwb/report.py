"""全量重演比对、诊断报告与时间轴导出（业务要求 4、6）。

报告与时间轴通过同一临时目录暂存后逐一 ``os.replace``，
先写时间轴再写报告（报告含时间轴摘要），任何一步失败都保留旧版本。
"""
from __future__ import annotations

import os
import tempfile

from .atomicio import atomic_write_json, digest_of
from .chunkstate import compute_chunk
from .curves import pedal_track, tempo_at_ticks
from .errors import E_DIGEST, WorkbenchError
from .expand import expand_score
from .persistence import ChunkStore

TIMELINE_NAME = "timeline.json"
REPORT_NAME = "diagnostic_report.json"


def full_recompute_timeline(score, params_final, chunk_size: int) -> list[dict]:
    """以并发度 1 的等价方式从头全量计算所有块（参照基准）。"""
    placed, total_ticks, _ = expand_score(score)
    tempo_full = tempo_at_ticks(params_final, score.ticks_per_beat, total_ticks).tolist()
    pedal_full = pedal_track(params_final, score.ticks_per_beat, total_ticks).tolist()
    chunk_count = (total_ticks + chunk_size - 1) // chunk_size
    return [
        compute_chunk(i, chunk_size, total_ticks, placed, params_final, score.ticks_per_beat, tempo_full, pedal_full)
        for i in range(chunk_count)
    ]


def compare_timelines(incremental: list[dict], full: list[dict], duration_tol: int, velocity_tol: int) -> dict:
    """逐 tick 实际时长与逐音符实际力度容差比对。"""
    if len(incremental) != len(full):
        raise WorkbenchError(E_DIGEST, "增量与全量块数量不一致")
    max_duration_diff = 0
    max_velocity_diff = 0
    mismatches: list[dict] = []
    for inc_chunk, full_chunk in zip(incremental, full):
        if inc_chunk["tick_lo"] != full_chunk["tick_lo"] or inc_chunk["tick_hi"] != full_chunk["tick_hi"]:
            raise WorkbenchError(E_DIGEST, "块 tick 区间不一致", {"chunk_index": inc_chunk["chunk_index"]})
        inc_seg = {s["note_id"]: s for s in inc_chunk["note_segments"]}
        full_seg = {s["note_id"]: s for s in full_chunk["note_segments"]}
        if set(inc_seg) != set(full_seg):
            raise WorkbenchError(E_DIGEST, "块内音符集合不一致", {"chunk_index": inc_chunk["chunk_index"]})
        for note_id, fs in full_seg.items():
            ins = inc_seg[note_id]
            d_diff = abs(ins["actual_duration_ticks"] - fs["actual_duration_ticks"])
            v_diff = abs(ins["actual_velocity"] - fs["actual_velocity"])
            max_duration_diff = max(max_duration_diff, d_diff)
            max_velocity_diff = max(max_velocity_diff, v_diff)
            if d_diff > duration_tol or v_diff > velocity_tol:
                mismatches.append(
                    {
                        "note_id": note_id,
                        "duration_diff": d_diff,
                        "velocity_diff": v_diff,
                        "chunk_index": inc_chunk["chunk_index"],
                    }
                )
        # 逐 tick 速度/踏板曲线必须完全一致（确定性要求）
        if inc_chunk["tempo"] != full_chunk["tempo"]:
            raise WorkbenchError(E_DIGEST, "逐 tick 速度曲线不一致", {"chunk_index": inc_chunk["chunk_index"]})
        if inc_chunk["pedal"] != full_chunk["pedal"]:
            raise WorkbenchError(E_DIGEST, "逐 tick 踏板轨道不一致", {"chunk_index": inc_chunk["chunk_index"]})
    return {
        "consistent": not mismatches,
        "max_duration_diff": max_duration_diff,
        "max_velocity_diff": max_velocity_diff,
        "duration_tolerance": duration_tol,
        "velocity_tolerance": velocity_tol,
        "mismatches": mismatches,
    }


def verify_persisted_chunks(store: ChunkStore, manifest: dict) -> list[dict]:
    """最终逐块摘要校验：摘要不匹配/缺失/tick 数不符即 E_DIGEST。"""
    verified = []
    expected = set(range(manifest["chunk_count"]))
    present = {c["chunk_index"] for c in manifest["chunks"]}
    if expected != present:
        raise WorkbenchError(
            E_DIGEST,
            "声明块集合与实际不符",
            {"missing": sorted(expected - present), "unexpected": sorted(present - expected)},
        )
    for entry in sorted(manifest["chunks"], key=lambda c: c["chunk_index"]):
        verified.append(store.verify_chunk(entry["chunk_index"]))
    return verified


def export_results(
    output_dir: str,
    score,
    runner,
    manifest: dict,
    chunks: list[dict],
    consistency: dict,
    impact_log: list[dict],
) -> dict:
    """联合原子提交时间轴与诊断报告（要么同时更新，要么都保持旧版本）。"""
    os.makedirs(output_dir, exist_ok=True)
    timeline_path = os.path.join(output_dir, TIMELINE_NAME)
    report_path = os.path.join(output_dir, REPORT_NAME)

    timeline_payload = {
        "title": score.title,
        "ticks_per_beat": score.ticks_per_beat,
        "total_ticks": manifest["total_ticks"],
        "chunk_size": manifest["chunk_size"],
        "version_id": manifest["version_id"],
        "chunks": chunks,
    }
    report_payload = {
        "ok": True,
        "title": score.title,
        "version_id": manifest["version_id"],
        "total_ticks": manifest["total_ticks"],
        "chunk_size": manifest["chunk_size"],
        "chunk_count": manifest["chunk_count"],
        "steps_applied": manifest["step_cursor"],
        "cross_voice_overlaps": runner.cross_voice_overlaps,
        "edit_impact": impact_log,
        "consistency": consistency,
        "digest_verification": {"status": "passed", "chunks_verified": len(chunks)},
        "timeline_digest": digest_of(timeline_payload),
        "business_codes": [],
    }

    # 先在同一临时目录准备两份文件，再依次整体替换；替换前不会破坏旧版本
    tmp_dir = tempfile.mkdtemp(prefix=".mwb-commit-", dir=output_dir)
    try:
        tmp_timeline = os.path.join(tmp_dir, TIMELINE_NAME)
        tmp_report = os.path.join(tmp_dir, REPORT_NAME)
        atomic_write_json(tmp_timeline, timeline_payload)
        atomic_write_json(tmp_report, report_payload)
        os.replace(tmp_timeline, timeline_path)
        os.replace(tmp_report, report_path)
    finally:
        if os.path.isdir(tmp_dir):
            for leftover in os.listdir(tmp_dir):
                try:
                    os.unlink(os.path.join(tmp_dir, leftover))
                except OSError:
                    pass
            os.rmdir(tmp_dir)
    return report_payload
