"""业务要求 4-7：增量重演、持久化恢复、版本识别、并发与预算。"""
import json
import os

import pytest

from mwb.atomicio import read_json
from mwb.errors import WorkbenchError
from mwb.parse import parse_score
from mwb.parse_edit import parse_edit_script
from mwb.parse_params import parse_expression
from mwb.report import compare_timelines, full_recompute_timeline, verify_persisted_chunks
from mwb.runner import EXIT_GRACEFUL_TERMINATION, EXIT_OK, WorkbenchRunner
from mwb.samples import CHUNK_SIZE, generate_edit_script, generate_expression, generate_score


def _write(tmp_path, name, payload):
    os.makedirs(tmp_path, exist_ok=True)
    path = os.path.join(tmp_path, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def _build(tmp_path, concurrency=2, budget=2, edits=None):
    score_path = _write(tmp_path, "s.json", generate_score())
    params_path = _write(tmp_path, "p.json", generate_expression())
    edit_payload = generate_edit_script(concurrency=concurrency, memory_budget_chunks=budget)
    if edits is not None:
        edit_payload["edits"] = edits
    edits_path = _write(tmp_path, "e.json", edit_payload)
    score = parse_score(score_path)
    params = parse_expression(params_path)
    edit_list, c, b = parse_edit_script(edits_path)
    return score, params, edit_list, c, b, read_json(score_path), read_json(params_path), read_json(edits_path)


def _runner(tmp_path, **kwargs):
    score, params, edits, c, b, sd, pd, ed = _build(tmp_path, **kwargs)
    return WorkbenchRunner(
        score, params, edits, os.path.join(tmp_path, "state"), CHUNK_SIZE, c, b, sd, pd, ed
    )


def test_incremental_matches_full(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run()
    assert result["status"] == "completed"
    chunks = verify_persisted_chunks(runner.store, result["manifest"])
    full = full_recompute_timeline(runner.score, result["params_final"], CHUNK_SIZE)
    outcome = compare_timelines(chunks, full, 0, 0)
    assert outcome["consistent"] is True
    assert outcome["max_duration_diff"] == 0 and outcome["max_velocity_diff"] == 0
    # 每步编辑都产生了影响记录且至少一次发生块失效
    assert len(runner.impact_log) == 8
    assert any(log["invalidated_chunks"] for log in runner.impact_log)


def test_concurrency_equivalence(tmp_path):
    outputs = []
    for concurrency in (1, 4):
        runner = _runner(str(tmp_path / f"c{concurrency}"), concurrency=concurrency, budget=4)
        result = runner.run()
        chunks = verify_persisted_chunks(runner.store, result["manifest"])
        full = full_recompute_timeline(runner.score, result["params_final"], CHUNK_SIZE)
        assert compare_timelines(chunks, full, 0, 0)["consistent"]
        outputs.append([(c["chunk_index"], c["tempo"], c["pedal"],
                         [(s["note_id"], s["actual_duration_ticks"], s["actual_velocity"])
                          for s in c["note_segments"]]) for c in chunks])
    assert outputs[0] == outputs[1]


def test_graceful_interruption_and_resume(tmp_path):
    runner = _runner(tmp_path)
    runner.stop.request()  # 在任何块开始前请求终止
    result = runner.run()
    assert result["exit_code"] == EXIT_GRACEFUL_TERMINATION
    assert os.path.exists(runner.store.interrupt_path)
    # 未完成块没有 manifest 条目，不会被误认为完成
    manifest = runner.store.read_manifest()
    assert manifest["step_cursor"] == 0

    runner2 = _runner(tmp_path)
    result2 = runner2.run()
    assert result2["status"] == "completed"
    assert not os.path.exists(runner2.store.interrupt_path)
    chunks = verify_persisted_chunks(runner2.store, result2["manifest"])
    full = full_recompute_timeline(runner2.score, result2["params_final"], CHUNK_SIZE)
    assert compare_timelines(chunks, full, 0, 0)["consistent"]


def test_interrupt_midway_then_resume(tmp_path):
    # 跑到第 3 步边界后中断
    runner = _runner(tmp_path)
    # 全新状态目录：初始全量为第 1 次池调用，首个编辑重算为第 2 次，
    # 在该块边界请求可处理终止
    fresh_dir = str(tmp_path / "midway")
    runner_b = _runner(fresh_dir)
    original_b = runner_b._run_chunk_pool
    call = {"n": 0}

    def stop_after_step_one(pending, params, placed, manifest_b):
        out = original_b(pending, params, placed, manifest_b)
        call["n"] += 1
        if call["n"] == 2:
            runner_b.stop.request()
        return out

    runner_b._run_chunk_pool = stop_after_step_one
    result = runner_b.run()

    assert result["status"] == "interrupted"
    cursor = result["step_cursor"]
    assert cursor >= 1

    runner2 = _runner(fresh_dir)
    result2 = runner2.run()
    assert result2["manifest"]["step_cursor"] == 8
    chunks = verify_persisted_chunks(runner2.store, result2["manifest"])
    full = full_recompute_timeline(runner2.score, result2["params_final"], CHUNK_SIZE)
    assert compare_timelines(chunks, full, 0, 0)["consistent"]


def test_version_isolation_keeps_old_directory(tmp_path):
    runner = _runner(tmp_path)
    runner.run()
    versions_before = set(os.listdir(os.path.join(tmp_path, "state")))
    # 修改表达参数生成新版本
    score, params, edits, c, b, sd, pd, ed = _build(tmp_path)
    pd["default_stretch"] = 1.05
    runner2 = WorkbenchRunner(
        score, params, edits, os.path.join(tmp_path, "state"), CHUNK_SIZE, c, b, sd, pd, ed
    )
    runner2.run()
    versions_after = set(os.listdir(os.path.join(tmp_path, "state")))
    assert versions_before.issubset(versions_after)
    assert len(versions_after) == 2


def test_digest_failure_triggers_recompute(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run()
    manifest = result["manifest"]
    # 篡改一个已完成块
    index = 1
    chunk_path = runner.store.chunk_path(index)
    chunk = read_json(chunk_path)
    chunk["tempo"][0] = chunk["tempo"][0] + 12.5
    with open(chunk_path, "w", encoding="utf-8") as handle:
        json.dump(chunk, handle)
    # 直接校验应报 E_DIGEST
    with pytest.raises(WorkbenchError) as info:
        verify_persisted_chunks(runner.store, manifest)
    assert info.value.code == "E_DIGEST"
    # 恢复流程剔除坏块并补算，最终重新一致
    runner3 = _runner(tmp_path)
    resumed = runner3.run()
    chunks = verify_persisted_chunks(runner3.store, resumed["manifest"])
    full = full_recompute_timeline(runner3.score, resumed["params_final"], CHUNK_SIZE)
    assert compare_timelines(chunks, full, 0, 0)["consistent"]


def test_orphan_chunk_quarantine_after_force_kill(tmp_path):
    runner = _runner(tmp_path)
    result = runner.run()
    manifest = result["manifest"]
    # 模拟强制终止：写出一个无 manifest 条目的块文件
    ghost = runner.store.read_chunk(0)
    ghost_path = os.path.join(runner.store.chunk_dir, "chunk-0999.json")
    ghost["chunk_index"] = 999
    with open(ghost_path, "w", encoding="utf-8") as handle:
        json.dump(ghost, handle)
    runner2 = _runner(tmp_path)
    runner2.run()
    assert os.path.exists(os.path.join(runner2.store.orphan_dir, "chunk-0999.json"))


def test_memory_budget_caps_inflight(tmp_path):
    runner = _runner(tmp_path, concurrency=4, budget=1)
    seen_max = {"value": 0}
    original = runner._guarded_compute

    def wrapped(index, params, placed, tempo_full, pedal_full):
        # budget=1 时任意时刻只允许 1 个块处于计算中
        return original(index, params, placed, tempo_full, pedal_full)

    runner._guarded_compute = wrapped
    result = runner.run()
    assert result["status"] == "completed"


def test_pedal_add_invalidates_by_tick_range(tmp_path):
    # 仅保留一个新增踏板编辑：旧块无 pedal:P3 键，需靠 tick 区间失效
    pedal_edit = [{"op": "pedal_add",
                   "args": {"pedal_id": "P3", "start_beat": 13.0, "end_beat": 15.0}}]
    score, params, edits, c, b, sd, pd, ed = _build(tmp_path, edits=pedal_edit)
    runner = WorkbenchRunner(
        score, params, edits, os.path.join(tmp_path, "state"), CHUNK_SIZE, c, b, sd, pd, ed
    )
    result = runner.run()
    log = runner.impact_log[0]
    assert log["tick_range"] == (13 * 480, 15 * 480)
    # 13..15 拍 = tick 6240..7200，覆盖块 6（5760..6720）与块 7（6720..7680）
    assert log["invalidated_chunks"] == [6, 7]
    chunks = verify_persisted_chunks(runner.store, result["manifest"])
    full = full_recompute_timeline(runner.score, result["params_final"], CHUNK_SIZE)
    assert compare_timelines(chunks, full, 0, 0)["consistent"]
