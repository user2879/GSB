"""异常注入链路覆盖：六个业务错误码逐一触发并记录。

所有场景输入与结果都序列化到 reports/error_scenarios.json，
不使用随机源，保证可精确复现。
"""

from __future__ import annotations

import copy
from pathlib import Path

from .engine import Engine
from .errors import PipelineError
from .generator import generate_score
from .injection import (
    FAULT_NOTE_NON_POSITIVE,
    FAULT_PEDAL_OVERLAP,
    FAULT_SNAPSHOT_STALE,
    FAULT_TEMPO_NON_MONOTONIC,
    FAULT_TIMELINE_HASH,
    FAULT_VERSION_REGRESS,
    build_faulty_payload,
)
from .model import Note, content_hash, round6, write_json
from .persistence import Store


def _capture(fn) -> dict:
    try:
        fn()
    except PipelineError as exc:
        record = exc.to_record()
        record["raised"] = True
        return record
    return {"ok": True, "raised": False, "code": None}


def run_error_scenarios(workspace: str) -> dict:
    workspace = Path(workspace)
    scenario_root = workspace / "error_scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)
    score = generate_score(seed=20261119)
    engine = Engine(score)
    results = []

    # 1) E_VELOCITY_MONO：速度曲线非单调拍号
    _, bad_tempo, dynamics, pedals = build_faulty_payload(
        FAULT_TEMPO_NON_MONOTONIC, score, score.tempo, score.dynamics, score.pedals
    )
    results.append(
        {"fault": FAULT_TEMPO_NON_MONOTONIC,
         "result": _capture(lambda: engine.full_rebuild(tempo=bad_tempo))}
    )

    # 2) E_BEAT_RANGE：音符时值为零
    score_copy = generate_score(seed=20261119)
    build_faulty_payload(
        FAULT_NOTE_NON_POSITIVE, score_copy,
        score_copy.tempo, score_copy.dynamics, score_copy.pedals,
    )
    engine_bad_note = Engine(score_copy)
    results.append(
        {"fault": FAULT_NOTE_NON_POSITIVE,
         "result": _capture(lambda: engine_bad_note.full_rebuild())}
    )

    # 3) E_PEDAL_OVERLAP：踏板区间交叉
    _, _, _, bad_pedals = build_faulty_payload(
        FAULT_PEDAL_OVERLAP, score, score.tempo, score.dynamics, score.pedals
    )
    results.append(
        {"fault": FAULT_PEDAL_OVERLAP,
         "result": _capture(lambda: engine.full_rebuild(pedals=bad_pedals))}
    )

    # 4) E_VERSION_REGRESS：版本戳回退（重复提交同版本）
    store_dir = scenario_root / "store_regress"
    store = Store(store_dir)
    baseline = engine.full_rebuild(version=0)
    store.commit_initial(baseline)
    results.append(
        {"fault": FAULT_VERSION_REGRESS,
         "result": _capture(lambda: store.commit(baseline, None, None))}
    )

    # 5) E_SNAPSHOT_STALE：旧版本快照整体替换
    from .engine import make_operation

    advanced = engine.full_rebuild(version=0)
    spec = {"version": 1, "kind": "tempo", "tempo": [dict(p) for p in score.tempo]}
    spec["tempo"][0]["bpm"] = round6(spec["tempo"][0]["bpm"] + 4.0)
    op = make_operation(1, advanced, "tempo", tempo=spec["tempo"], total_beats=score.total_beats)
    advanced, _ = engine.incremental_apply(advanced, op)
    store.commit(advanced, None, op.to_dict())
    stale_manifest = copy.deepcopy(store.read_manifest())
    stale_manifest["version"] = 0  # 显式落后于当前版本 1
    results.append(
        {"fault": FAULT_SNAPSHOT_STALE,
         "result": _capture(lambda: store.replace_snapshot(stale_manifest))}
    )

    # 6) E_TIMELINE_HASH：破坏 blob 后恢复校验
    corrupt_dir = scenario_root / "store_corrupt"
    corrupt_store = Store(corrupt_dir)
    corrupt_store.commit_initial(baseline)
    corrupt_manifest = corrupt_store.read_manifest()
    any_digest = sorted(corrupt_manifest["shard_blobs"].values())[0]
    blob_path = corrupt_store.blobs / f"{any_digest}.json"
    blob = __import__("json").loads(blob_path.read_text("utf-8"))
    blob["notes"][0]["start_ms"] = blob["notes"][0]["start_ms"] + 999.0
    write_json(blob_path, blob)  # 文件名哈希与新内容不再一致
    results.append(
        {"fault": FAULT_TIMELINE_HASH,
         "result": _capture(lambda: corrupt_store.recover())}
    )

    report = {
        "schema": "error-scenarios/1",
        "scenario_total": len(results),
        "all_codes_covered": all(item["result"].get("raised") for item in results),
        "scenarios": [
            {
                "fault": item["fault"],
                "code": item["result"].get("code"),
                "raised": item["result"].get("raised", False),
                "message": item["result"].get("message"),
            }
            for item in results
        ],
    }
    write_json(workspace / "reports" / "error_scenarios.json", report)
    return report
