"""独立验证器。

只消费生成文件、持久化目录与注入配置（纯标准库 IO），不导入
engine / tempo / persistence / lifecycle 中的任何内存状态或函数，
速度插值、力度包络、分片哈希均在此独立重算。
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from pathlib import Path

from .model import content_hash, read_json

ART_DURATION_FACTOR = {"legato": 1.0, "normal": 0.95, "staccato": 0.5}
ART_VELOCITY_FACTOR = {"legato": 1.0, "normal": 0.95, "staccato": 0.9}


def _beat_to_ms(points: list[dict], beat: float) -> float:
    """验证器自有实现：与引擎规格一致的分段线性 bpm 积分公式。"""
    beats = [float(p["beat"]) for p in points]
    bpms = [float(p["bpm"]) for p in points]
    anchors = [0.0]
    for idx in range(len(points) - 1):
        b0, b1 = beats[idx], beats[idx + 1]
        q0, q1 = bpms[idx], bpms[idx + 1]
        anchors.append(anchors[-1] + (b1 - b0) * 60000.0 / ((q0 + q1) / 2.0))
    beat = float(beat)
    if beat <= 0:
        return beat * 60000.0 / bpms[0]
    if beat >= beats[-1]:
        idx = len(beats) - 1
        return anchors[idx] + (beat - beats[idx]) * 60000.0 / bpms[idx]
    idx = bisect_right(beats, beat) - 1
    b0, b1 = beats[idx], beats[idx + 1]
    q0, q1 = bpms[idx], bpms[idx + 1]
    q_at = q0 + (q1 - q0) * (beat - b0) / (b1 - b0)
    return anchors[idx] + (beat - b0) * 60000.0 / ((q0 + q_at) / 2.0)


def _r6(value: float) -> float:
    return round(float(value), 6)


def _gain(dynamics: list[dict], beat: float) -> float:
    for interval in dynamics:
        lo, hi = float(interval["start_beat"]), float(interval["end_beat"])
        if lo - 1e-12 <= beat < hi - 1e-12 or abs(beat - hi) <= 1e-12:
            ratio = 0.0 if hi <= lo else (beat - lo) / (hi - lo)
            return float(interval["gain"]) + (
                float(interval["end_gain"]) - float(interval["gain"])
            ) * ratio
    return 1.0


def _expected_note(note: dict, tempo: list[dict], dynamics: list[dict]) -> dict:
    start_ms = _r6(_beat_to_ms(tempo, note["start_beat"]))
    end_ms = _r6(_beat_to_ms(tempo, note["start_beat"] + note["duration_beats"]))
    factor = ART_DURATION_FACTOR[note["articulation"]]
    gain = _gain(dynamics, float(note["start_beat"]))
    return {
        "note_id": note["note_id"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": _r6((end_ms - start_ms) * factor),
        "velocity_envelope": _r6(
            note["velocity"] * gain * ART_VELOCITY_FACTOR[note["articulation"]]
        ),
        "dynamics_gain": _r6(gain),
    }


def validate_workspace(workspace: str) -> dict:
    workspace = Path(workspace)
    sample_dir = workspace / "sample"
    state_dir = workspace / "state"
    score = read_json(sample_dir / "score.json")
    manifest = read_json(state_dir / "manifest.json")
    truth = {
        path.name: read_json(path)
        for path in sorted(sample_dir.glob("truth_*.json"))
    }

    failures: list[dict] = []
    hash_failures: list[dict] = []
    checked_notes = 0
    checked_shards = 0
    tempo = manifest["tempo"]
    dynamics = manifest["dynamics"]

    # 1) 逐声部逐音符复核时刻/时长/力度，并定位首个不一致音符
    blobs_dir = state_dir / "blobs"
    persisted_rows: dict[str, dict] = {}
    for shard_id, digest in sorted(manifest["shard_blobs"].items()):
        blob = read_json(blobs_dir / f"{digest}.json")
        outer = hashlib.sha256(
            json.dumps(
                blob,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        inner = content_hash({k: v for k, v in blob.items() if k != "content_hash"})
        if outer != digest:
            hash_failures.append({"type": "shard_hash", "shard_id": shard_id})
        if blob.get("content_hash") != inner:
            hash_failures.append({"type": "shard_inner_hash", "shard_id": shard_id})
        checked_shards += 1
        for row in blob["notes"]:
            persisted_rows[row["note_id"]] = row

    score_notes = {note["note_id"]: note for part in score["parts"] for note in part["notes"]}
    for part in score["parts"]:
        part_id = part["part_id"]
        truth_notes = truth[f"truth_{part_id}.json"]["notes"]
        for index, note in enumerate(part["notes"]):
            if truth_notes[index]["note_id"] != note["note_id"]:
                failures.append(
                    {
                        "type": "truth_order",
                        "part_id": part_id,
                        "first_bad_note": note["note_id"],
                    }
                )
                continue
            expected = _expected_note(note, tempo, dynamics)
            actual = persisted_rows.get(note["note_id"])
            if actual is None:
                failures.append(
                    {
                        "type": "missing_note",
                        "part_id": part_id,
                        "first_bad_note": note["note_id"],
                    }
                )
                continue
            mismatches = {}
            for field in ("start_ms", "end_ms", "duration_ms", "velocity_envelope", "dynamics_gain"):
                if abs(float(actual[field]) - float(expected[field])) > 1e-6:
                    mismatches[field] = {"expected": expected[field], "actual": actual[field]}
            if mismatches:
                failures.append(
                    {
                        "type": "note_mismatch",
                        "part_id": part_id,
                        "first_bad_note": note["note_id"],
                        "fields": mismatches,
                    }
                )
                break
            checked_notes += 1

    failures.extend(hash_failures)
    # 2) 踏板区间毫秒复核（以清单当前表达状态为准，独立重算毫秒）
    current_pedal_intervals = [
        {
            "part_id": pedal["part_id"],
            "start_beat": pedal["start_beat"],
            "end_beat": pedal["end_beat"],
        }
        for pedal in manifest["pedals"]
    ]
    expected_pedals = sorted(
        (
            {
                "part_id": pedal["part_id"],
                "start_beat": _r6(pedal["start_beat"]),
                "end_beat": _r6(pedal["end_beat"]),
                "start_ms": _r6(_beat_to_ms(tempo, pedal["start_beat"])),
                "end_ms": _r6(_beat_to_ms(tempo, pedal["end_beat"])),
            }
            for pedal in current_pedal_intervals
        ),
        key=lambda item: (item["part_id"], item["start_beat"]),
    )
    if expected_pedals != manifest["pedals"]:
        failures.append({"type": "pedal_mismatch"})

    # 3) 时间轴整体哈希复核
    recomputed_hash = content_hash(
        {
            "schema": "timeline-hash/1",
            "shard_hashes": [
                (manifest.get("shard_inner_hashes") or manifest["shard_blobs"])[key]
                for key in sorted(manifest.get("shard_inner_hashes") or manifest["shard_blobs"])
            ],
            "pedals": manifest["pedals"],
        }
    )
    if recomputed_hash != manifest["timeline_hash"]:
        failures.append(
            {
                "type": "timeline_hash",
                "expected": recomputed_hash,
                "actual": manifest["timeline_hash"],
            }
        )

    # 4) 操作序列与清单版本戳一致性
    ops_path = state_dir / "operations.jsonl"
    operations = (
        [json.loads(line) for line in ops_path.read_text("utf-8").splitlines() if line]
        if ops_path.exists()
        else []
    )
    expected_versions = list(range(1, len(operations) + 1))
    actual_versions = [op["version"] for op in operations]
    if actual_versions != expected_versions or manifest["version"] != len(operations):
        failures.append(
            {
                "type": "version_chain",
                "manifest_version": manifest["version"],
                "operation_versions": actual_versions,
            }
        )

    return {
        "schema": "validation-report/1",
        "ok": not failures,
        "stable_version": manifest["version"],
        "timeline_hash": manifest["timeline_hash"],
        "checked_note_total": checked_notes,
        "score_note_total": len(score_notes),
        "checked_shard_total": checked_shards,
        "operation_total": len(operations),
        "first_failure": failures[0] if failures else None,
        "failure_total": len(failures),
    }
