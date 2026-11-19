"""生命周期编排：固定操作序列、工作进程、强制终止、恢复续写。

固定序列所有输入均由乐谱确定派生（无随机、无时间戳、无主机信息），
因此两次完整运行产出的报告与哈希逐字节一致。
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

from .engine import Engine, Operation, make_operation
from .generator import generate_score, write_score_bundle
from .injection import InjectionConfig
from .model import read_json, round6, write_json
from .persistence import Store
from .tempo import TempoMap

WORKER_OPS = (1, 2, 3)


def planned_operations(score) -> list[dict]:
    """固定增量序列：v1 速度、v2 力度区间、v3 踏板区间。"""
    tempo_v1 = [dict(point) for point in score.tempo]
    tempo_v1[1]["bpm"] = round6(tempo_v1[1]["bpm"] + 12.0)
    tempo_v1[2]["bpm"] = round6(tempo_v1[2]["bpm"] - 8.0)

    dynamics_v2 = [dict(interval) for interval in score.dynamics]
    dynamics_v2[1]["end_gain"] = round6(dynamics_v2[1]["end_gain"] + 0.1)

    pedals_v3 = [dict(pedal) for pedal in score.pedals]
    pedals_v3[0]["end_beat"] = round6(pedals_v3[0]["end_beat"] + 1.0)

    return [
        {"version": 1, "kind": "tempo", "tempo": tempo_v1},
        {"version": 2, "kind": "dynamics", "dynamics": dynamics_v2},
        {"version": 3, "kind": "pedals", "pedals": pedals_v3},
    ]


def _build_op(engine: Engine, parent, spec: dict, total_beats: float) -> Operation:
    return make_operation(
        version=spec["version"],
        parent=parent,
        kind=spec["kind"],
        tempo=spec.get("tempo"),
        dynamics=spec.get("dynamics"),
        pedals=spec.get("pedals"),
        total_beats=total_beats,
    )


def prepare_workspace(workspace: str, seed: int = 20261119) -> dict:
    """生成确定性乐谱并返回基础信息。"""
    workspace = Path(workspace)
    sample_dir = workspace / "sample"
    state_dir = workspace / "state"
    report_dir = workspace / "reports"
    for directory in (workspace, sample_dir, state_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    score = generate_score(seed=seed)
    gen_report = write_score_bundle(score, str(sample_dir))
    return {
        "workspace": str(workspace),
        "sample_dir": str(sample_dir),
        "state_dir": str(state_dir),
        "report_dir": str(report_dir),
        "score": score,
        "generation_report": gen_report,
    }


def run_worker(workspace: str, injection: InjectionConfig, max_op: int | None = None) -> dict:
    """工作进程入口：基线合成 + 按计划提交增量；注入提交中断时硬退出。"""
    info = _load_context(workspace)
    score = info["score"]
    store = Store(info["state_dir"], injection=injection)
    engine = Engine(score)
    run_record = {"commits": [], "hard_exit": None}

    if not store.has_manifest():
        baseline = engine.full_rebuild(version=0)
        store.commit_initial(baseline)
        run_record["commits"].append(
            {"version": 0, "mode": "full_rebuild", "timeline_hash": baseline.timeline_hash}
        )

    state = engine.full_rebuild(
        tempo=store.read_manifest()["tempo"],
        dynamics=store.read_manifest()["dynamics"],
        pedals=_pedal_intervals_from_manifest(store.read_manifest()),
        version=store.read_manifest()["version"],
    )
    specs = planned_operations(score)
    limit = len(specs) if max_op is None else min(max_op, len(specs))
    for spec in specs[:limit]:
        op = _build_op(engine, state, spec, score.total_beats)
        state, closure = engine.incremental_apply(state, op)
        store.commit(state, closure_record=closure, op_dict=op.to_dict())
        run_record["commits"].append(
            {
                "version": op.version,
                "mode": "incremental",
                "kind": op.kind,
                "recomputed_note_total": closure["recomputed_note_total"],
                "touched_shard_ids": closure["touched_shard_ids"],
                "timeline_hash": state.timeline_hash,
            }
        )
    write_json(Path(info["report_dir"]) / "worker_run.json", run_record)
    return run_record


def _pedal_intervals_from_manifest(manifest: dict) -> list[dict]:
    return [
        {
            "part_id": row["part_id"],
            "start_beat": row["start_beat"],
            "end_beat": row["end_beat"],
        }
        for row in manifest["pedals"]
    ]


def _load_context(workspace: str) -> dict:
    workspace = Path(workspace)
    score_data = read_json(workspace / "sample" / "score.json")
    from .model import Score

    score = Score.from_dict(score_data)
    return {
        "score": score,
        "state_dir": str(workspace / "state"),
        "report_dir": str(workspace / "reports"),
    }


def resume_after_crash(workspace: str) -> dict:
    """恢复并续写未完成的固定序列，返回恢复报告与续写提交记录。"""
    info = _load_context(workspace)
    score = info["score"]
    store = Store(info["state_dir"])
    recovery = store.recover()

    manifest = store.read_manifest()
    engine = Engine(score)
    state = engine.full_rebuild(
        tempo=manifest["tempo"],
        dynamics=manifest["dynamics"],
        pedals=_pedal_intervals_from_manifest(manifest),
        version=manifest["version"],
    )
    resume_commits = []
    for spec in planned_operations(score):
        if spec["version"] <= manifest["version"]:
            continue
        op = _build_op(engine, state, spec, score.total_beats)
        state, closure = engine.incremental_apply(state, op)
        store.commit(state, closure_record=closure, op_dict=op.to_dict())
        resume_commits.append(
            {
                "version": op.version,
                "mode": "incremental_resumed",
                "kind": op.kind,
                "timeline_hash": state.timeline_hash,
            }
        )
    write_json(
        Path(info["report_dir"]) / "recovery.json",
        {"recovery": recovery, "resume_commits": resume_commits},
    )
    return {"recovery": recovery, "resume_commits": resume_commits, "final_state": state}


def controlled_close(workspace: str) -> dict:
    """受控退出：完整清单 + CLOSED 标记 + 退出报告。"""
    info = _load_context(workspace)
    store = Store(info["state_dir"])
    manifest = store.read_manifest()
    operations = store.load_operations()
    report = {
        "status": "controlled_exit",
        "stable_version": manifest["version"],
        "timeline_hash": manifest["timeline_hash"],
        "operation_total": len(operations),
        "shard_total": len(manifest["shard_blobs"]),
        "orphan_total": len(list((store.orphans).glob("*"))),
        "exit_report_present": True,
    }
    store.write_controlled_close(report)
    write_json(Path(info["report_dir"]) / "controlled_exit.json", report)
    return report


def hard_exit(point: str, exit_code: int = 137) -> None:
    """模拟强制终止：跳过缓冲刷新与所有 finally，直接结束进程。"""
    sys.stderr.write(
        '{"ok":false,"forced_termination":true,"point":"%s"}\n' % point
    )
    sys.stderr.flush()
    os._exit(exit_code)
