import copy
import json
import os
import random
import signal
from pathlib import Path

from dialogue_rag.codes import Code
from dialogue_rag.cli import _install_signal_handlers
from dialogue_rag.commands import apply_command
from dialogue_rag.diagnostics import compare_turn_reports, to_jsonable
from dialogue_rag.engine import IncrementalEngine
from dialogue_rag.fixedpoint import FixedProfile
from dialogue_rag.model import build_state
from dialogue_rag.recall import compute_turn_recall
from dialogue_rag.sample_data import build_sample
from dialogue_rag.snapshot import SnapshotStore
from dialogue_rag.topics import segment_session


def _sample_engine(tmp_path: Path, workers: int = 1):
    config, command_doc = build_sample()
    state, failures = build_state(config)
    state.workers = workers
    store = SnapshotStore(tmp_path / "snapshot")
    engine = IncrementalEngine(state, failures, store=store)
    return engine, command_doc["commands"], config


def _run_full_sequence(tmp_path: Path, *, restart_each: bool = False, workers: int = 1):
    _, command_doc = build_sample()
    return _run_named_commands(command_doc["commands"], tmp_path, restart_each=restart_each, workers=workers)


def _run_named_commands(commands, tmp_path: Path, *, restart_each: bool = False, workers: int = 1, config=None):
    if config is None:
        config, _ = build_sample()
    state, failures = build_state(config)
    state.workers = workers
    store = SnapshotStore(tmp_path / "snapshot")
    engine = IncrementalEngine(state, failures, store=store)
    engine.record_baseline()
    for index, raw_command in enumerate(commands):
        command = dict(raw_command, index=index)
        if restart_each and index > 0:
            snapshot = store.load()
            assert snapshot is not None
            engine = IncrementalEngine(
                snapshot.state,
                snapshot.failures,
                cache=snapshot.cache,
                history=snapshot.history,
                pending=snapshot.pending,
                store=store,
                recovery=snapshot.recovery,
                command_cursor=snapshot.manifest["command_cursor"],
            )
        engine.apply_and_run(command)
    return to_jsonable(engine.build_report())


def test_deterministic_random_sequences_survive_restart(tmp_path):
    query_variants = [
        "退款 进度 查询 到账 处理",
        "物流 地址 修改 配送 电话",
        "发票 抬头 账单 信息 修改",
        "配送 时效 送达 物流 进度",
        "账号 登录 密码 重置 验证",
    ]
    config, _ = build_sample()
    for seed in range(8):
        rng = random.Random(9000 + seed)
        commands = []
        added = False
        deleted = False
        for index in range(14):
            choice = rng.randrange(8)
            if choice == 0:
                commands.append({"op": "replace_query", "session_id": "S1", "turn_id": f"T{rng.randint(1, 8)}", "query": rng.choice(query_variants)})
            elif choice == 1:
                commands.append({"op": "set_topic_rule", "topic": {"similarity_threshold": rng.choice(["0.10", "0.25", "0.40"]), "equal_rule": rng.choice(["merge", "split"])}})
            elif choice == 2:
                commands.append({"op": "set_channel_weight", "channel_id": rng.choice(["ch_graph", "ch_lex", "ch_sem", "ch_missing"]), "weight": round(rng.uniform(0.4, 1.6), 3)})
            elif choice == 3:
                commands.append({"op": "set_evidence_weight", "evidence_id": rng.choice(["E001", "E002", "E008", "E999"]), "weight": round(rng.uniform(0.5, 2.2), 3)})
            elif choice == 4:
                commands.append({"op": "set_fixed_point", "fixed_point": {"fixed_width": rng.choice(["F8", "F16", "F32"]), "quant_gear": rng.choice(["G0", "G1", "G2"])}})
            elif choice == 5 and not added:
                added = True
                commands.append({"op": "add_turn", "session_id": "S1", "turn": {"turn_id": "T7", "query": "发票 抬头 账单 信息 修改", "channel_hits": [{"channel_id": "ch_lex", "evidence_id": "E007", "raw_score": 1.8}]}})
            elif choice == 6 and not deleted:
                deleted = True
                commands.append({"op": "delete_turn", "session_id": "S1", "turn_id": "T4"})
            else:
                commands.append({"op": rng.choice(["unknown_op", "mystery_op"]), "expected": rng.randrange(1000)})
        one = _run_named_commands(commands, tmp_path / f"one-{seed}", config=config)
        restarted = _run_named_commands(commands, tmp_path / f"restart-{seed}", restart_each=True, config=config)
        assert one["steps"] == restarted["steps"]
        assert one["turn_results"] == restarted["turn_results"]
        assert one["failure_list"] == restarted["failure_list"]

        replay_state, replay_failures = build_state(config)
        replay = IncrementalEngine(replay_state, replay_failures)
        replay.record_baseline(commit=False)
        for index, command in enumerate(commands):
            replay.apply_and_run(dict(command, index=index), commit=False)
        clean = IncrementalEngine(copy.deepcopy(replay.state)).evaluate()
        assert clean["turn_results"] == one["turn_results"]


def test_validation_collects_turn_field_errors_without_stopping_session():
    invalid = {
        "sessions": [
            {
                "session_id": "bad",
                "turns": [
                    {"turn_id": "B1", "seq": 1, "query": "ok"},
                    {"turn_id": "B3", "seq": 3, "query": "gap"},
                ],
            },
            {
                "session_id": "good",
                "turns": [{"turn_id": "G1", "seq": 1, "query": "valid"}],
            },
        ],
        "evidence": [
            {"evidence_id": "E1", "text": "one", "weight": 1},
            {"evidence_id": "E1", "text": "duplicate", "weight": 1},
        ],
        "channels": [{"channel_id": "C1", "weight": float("nan")}],
        "max_evidence_per_turn": 0,
        "fixed_point": {"fixed_width": "F7", "quant_gear": "G9"},
    }
    state, failures = build_state(invalid)
    codes = {(item["code"], item.get("field"), item.get("turn_id")) for item in failures}
    assert (Code.TURN_SEQ_GAP, "seq", "B3") in codes
    assert any(item["code"] == Code.DUPLICATE_ID and item.get("evidence_id") == "E1" for item in failures)
    assert any(item["code"] == Code.FIELD_NON_FINITE and "weight" in item["field"] for item in failures)
    assert any(item["code"] == Code.FIELD_OUT_OF_RANGE and "max_evidence" in item["field"] for item in failures)
    assert any(item["code"] == Code.FIELD_NOT_ALLOWED and item["field"].endswith("fixed_width") for item in failures)
    assert "good" in state.turns
    assert state.find_turn("good", "G1") is not None


def test_authority_uses_weighted_score_then_channel_name():
    config, _ = build_sample()
    state, _ = build_state(config)
    profile = FixedProfile()
    t1 = compute_turn_recall(state, "S1", "T1", profile)
    e001 = next(item for item in t1["evidence"] if item["evidence_id"] == "E001")
    e002 = next(item for item in t1["evidence"] if item["evidence_id"] == "E002")
    assert e001["authority_channel_id"] == "ch_sem"
    assert e001["source_channels"] == ["ch_lex", "ch_sem"]
    assert e002["authority_channel_id"] == "ch_graph"
    assert any(item["channel_id"] == "ch_lex" and item["reason"] == "channel_name_later" for item in e002["covered_channels"])

    command = {"op": "set_channel_weight", "channel_id": "ch_sem", "weight": 1.05}
    result = apply_command(state, command)
    assert result["applied"]
    changed = compute_turn_recall(state, "S1", "T1", profile)
    changed_e001 = next(item for item in changed["evidence"] if item["evidence_id"] == "E001")
    assert changed_e001["authority_channel_id"] == "ch_lex"


def test_topic_equal_threshold_has_fixed_decision_and_real_attribution_change():
    config, _ = build_sample()
    state, _ = build_state(config)
    before = segment_session(state, "S1")
    assert [topic["topic_id"] for topic in before["topics"]] == ["TP_T1", "TP_T3", "TP_T5"]
    assert len(before["equal_boundaries"]) == 3
    assert all(item["decision"] == "merge" for item in before["equal_boundaries"])

    state.equal_rule = "split"
    after = segment_session(state, "S1")
    assert len(after["topics"]) == 6
    assert all(item["decision"] == "split" for item in after["equal_boundaries"])


def test_fixed_width_and_gear_change_internal_integers(tmp_path):
    config, _ = build_sample()
    state16, _ = build_state(config)
    report16 = IncrementalEngine(state16).evaluate()
    t1_16 = next(item for item in report16["turn_results"] if item["turn_id"] == "T1")

    state8, _ = build_state(config)
    state8.fixed_width = 8
    report8 = IncrementalEngine(state8).evaluate()
    t1_8 = next(item for item in report8["turn_results"] if item["turn_id"] == "T1")
    assert t1_16["confidence_int"] != t1_8["confidence_int"]
    assert t1_16["confidence"] != t1_8["confidence"]

    state_fine, _ = build_state(config)
    state_fine.quant_gear = "G0"
    t1_fine = next(item for item in IncrementalEngine(state_fine).evaluate()["turn_results"] if item["turn_id"] == "T1")
    state_coarse, _ = build_state(config)
    state_coarse.quant_gear = "G2"
    t1_coarse = next(item for item in IncrementalEngine(state_coarse).evaluate()["turn_results"] if item["turn_id"] == "T1")
    assert t1_fine["confidence_int"] != t1_coarse["confidence_int"]
    assert t1_fine["confidence"] != t1_coarse["confidence"]


def test_quantization_flip_is_distinguished_from_structure():
    config, _ = build_sample()
    state16, _ = build_state(config)
    state16.answer_threshold = __import__("decimal").Decimal("0.517")
    engine16 = IncrementalEngine(state16)
    eval16 = engine16.evaluate()

    state8 = copy.deepcopy(state16)
    state8.fixed_width = 8
    state8.quant_gear = "G1"
    eval8 = IncrementalEngine(state8).evaluate()
    baseline = {"turn_results": eval16["turn_results"], "recall_report": {"turns": eval16["recall_turns"]}}
    candidate = {"turn_results": eval8["turn_results"], "recall_report": {"turns": eval8["recall_turns"]}}
    comparison = compare_turn_reports(baseline, candidate)
    assert comparison["flip_count"] >= 1
    flip = next(item for item in comparison["flips"] if item["turn_id"] == "T1")
    assert flip["classification"] == "quantization_rounding"
    assert flip["before_chain"] == flip["after_chain"]
    assert flip["reference_consistent"] if "reference_consistent" in flip else True
    assert "threshold" in flip and "before_margin" in flip


def test_precise_invalidation_after_baseline(tmp_path):
    engine, commands, _ = _sample_engine(tmp_path)
    engine.record_baseline()
    replace_step = engine.apply_and_run(dict(commands[0], index=0))
    decisions = {item["turn_id"]: item for item in replace_step["invalidation"]["cache_decisions"]}
    assert [tid for tid, item in decisions.items() if item["recall"]["status"] == "recomputed"] == ["T3"]
    assert decisions["T1"]["scoring"]["status"] == "reused"
    assert decisions["T3"]["scoring"]["status"] == "recomputed"
    assert decisions["T6"]["scoring"]["status"] == "recomputed"
    assert len(replace_step["invalidation"]["reused_topics"]) == 2
    assert [item["topic_id"] for item in replace_step["invalidation"]["recompute_topics"]] == ["TP_T3"]


def test_evidence_weight_change_reuses_other_evidence_contributions(tmp_path):
    engine, _, _ = _sample_engine(tmp_path)
    engine.record_baseline()
    step = engine.apply_and_run(
        {"op": "set_evidence_weight", "evidence_id": "E002", "weight": 1.75, "index": 0}
    )
    forensic = [
        item["evidence_id"]
        for item in step["invalidation"]["recompute_evidence"]
        if "forensic_weighting" in item["stages"]
    ]
    assert forensic == ["E002"]
    assert all(
        "cache_status" not in detail
        for turn in step["turn_results"]
        for detail in turn["chain_details"]
    )

    clean = IncrementalEngine(copy.deepcopy(engine.state)).evaluate()
    assert clean["turn_results"] == step["turn_results"]


def test_incremental_restart_matches_single_run_and_clean_full(tmp_path):
    one_shot = _run_full_sequence(tmp_path / "one", restart_each=False)
    restarted = _run_full_sequence(tmp_path / "restart", restart_each=True)
    assert one_shot["steps"] == restarted["steps"]
    assert one_shot["turn_results"] == restarted["turn_results"]
    assert one_shot["failure_list"] == restarted["failure_list"]

    config, _ = build_sample()
    final_state, failures = build_state(config)
    # Replay to obtain the final mutated state through the proven command path.
    replay = IncrementalEngine(final_state, failures)
    replay.record_baseline(commit=False)
    for index, command in enumerate(build_sample()[1]["commands"]):
        replay.apply_and_run(dict(command, index=index), commit=False)
    clean_eval = IncrementalEngine(copy.deepcopy(replay.state)).evaluate()
    assert to_jsonable(clean_eval["turn_results"]) == one_shot["turn_results"]


def test_thread_pool_matches_serial(tmp_path):
    serial = _run_full_sequence(tmp_path / "serial", workers=1)
    concurrent = _run_full_sequence(tmp_path / "concurrent", workers=4)
    assert serial["turn_results"] == concurrent["turn_results"]
    assert serial["steps"] == concurrent["steps"]


def test_budget_pending_and_next_batch_completion(tmp_path):
    engine, _, _ = _sample_engine(tmp_path)
    first = engine.evaluate(budget_override=4)
    assert first["budget"]["used_points"] == 3
    assert first["pending"][0]["turn_id"] == "T2"
    assert first["pending"][0]["code"] == Code.BUDGET_PENDING
    assert all(item["scoring"]["status"] == "pending" for item in first["cache_decisions"] if item["turn_id"] != "T1")

    second = engine.evaluate(budget_override=100000)
    assert second["pending"] == []
    assert len(second["turn_results"]) == 6
    clean = IncrementalEngine(copy.deepcopy(engine.state)).evaluate()
    assert second["turn_results"] == to_jsonable(clean["turn_results"])


def test_snapshot_resume_and_corruption_rebuild(tmp_path):
    report = _run_full_sequence(tmp_path / "persisted", restart_each=True)
    store = SnapshotStore(tmp_path / "persisted" / "snapshot")
    loaded = store.load()
    assert loaded is not None
    assert loaded.recovery["mode"] == "resume"
    assert loaded.manifest["command_cursor"] == 7
    assert report["snapshot"]["global_version"] == loaded.manifest["global_version"]

    stage_rel = loaded.manifest["stage_files"]["scoring"] if "stage_files" in loaded.manifest else "stages/scoring.json"
    stage_path = Path(loaded.manifest["snapshot_dir"]) / stage_rel
    stage_path = store.root / loaded.manifest["snapshot_dir"] / "stages" / "scoring.json"
    stage_path.write_text(stage_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    rebuilt = store.load()
    assert rebuilt is not None
    assert rebuilt.recovery["mode"] == "rebuild"
    assert rebuilt.recovery["code"] == Code.SNAPSHOT_REBUILD


def test_graceful_stop_marks_pending_then_completes(tmp_path):
    engine, _, _ = _sample_engine(tmp_path)
    engine.stop_event.set()
    stopped = engine.evaluate()
    assert stopped["pending"]
    assert all(item["code"] == Code.STOPPING for item in stopped["pending"])
    engine.stop_event.clear()
    completed = engine.evaluate(budget_override=100000)
    assert completed["pending"] == []
    assert len(completed["turn_results"]) == 6


def test_process_signal_handler_sets_graceful_stop(tmp_path):
    engine, _, _ = _sample_engine(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        _install_signal_handlers(engine)
        os.kill(os.getpid(), signal.SIGTERM)
        assert engine.stop_event.is_set()
        assert engine.recovery["stop_code"] == Code.STOPPING
    finally:
        signal.signal(signal.SIGTERM, previous)
