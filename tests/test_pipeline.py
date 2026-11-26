import copy

import pytest

from chaindiag.ledger import CacheStore
from chaindiag.pipeline import compare_reports, engine_from_data, state_to_engine
from chaindiag.samplegen import default_bundle
from chaindiag.state import apply_commands


@pytest.fixture
def bundle():
    return default_bundle()


def make_engine(bundle, **kwargs):
    return engine_from_data(
        config=bundle["config"],
        chain=bundle["chain_e2"],
        tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"],
        **kwargs,
    )


def test_healthy_pipeline_outputs_five_cache_stages_and_stable_order(bundle):
    report = make_engine(bundle, store=CacheStore(), serial=True).run()
    assert report["status"] == "ok"
    assert len(report["token_results"]) == 8
    assert len(report["cache_records"]) == 40  # four operators plus final result
    assert [entry["ordinal"] for entry in report["reuse_ledger"]] == list(range(40))
    assert all(token["status"] == "ok" for token in report["token_results"])
    assert any(token["boundary_token"] for token in report["token_results"])
    attention_summary = next(stage for stage in report["stage_summaries"] if stage["kind"] == "attention")
    weights = report["cache_records"]
    assert attention_summary["rounding_steps_recorded"] > 0
    attention_records = [r for r in weights if r["kind"] == "attention"]
    for record in attention_records:
        for head_weights in record["details"]["weights_permille"].values():
            assert sum(head_weights) == 1000


def test_single_token_replacement_invalidates_only_its_segment_downstream(bundle):
    store = CacheStore()
    baseline = make_engine(bundle, store=store, serial=True).run(with_width_reference=False)
    assert baseline["status"] == "ok"

    changed = copy.deepcopy(bundle)
    changed["tokens"][1]["token"] = 29
    incremental = make_engine(changed, store=store, serial=True).run(with_width_reference=False)
    actions = {
        (entry["op_seq"], entry["kind"], entry["token_uid"]): entry["action"]
        for entry in incremental["reuse_ledger"]
    }

    assert actions[(0, "embedding", "t0001")] == "computed"
    assert actions[(0, "embedding", "t0000")] == "reused"
    assert actions[(0, "embedding", "t0003")] == "reused"
    for uid in ("t0000", "t0001", "t0002"):
        assert actions[(1, "segment_pool", uid)] == "computed"
        assert actions[(2, "attention", uid)] == "computed"
        assert actions[(3, "layer_norm", uid)] == "computed"
    for uid in ("t0003", "t0004", "t0005", "t0006", "t0007"):
        assert actions[(1, "segment_pool", uid)] == "reused"
        assert actions[(2, "attention", uid)] == "reused"

    clean = make_engine(changed, store=CacheStore(), serial=True).run(with_width_reference=False)
    comparison = compare_reports(incremental, clean)
    assert comparison["equal"], comparison["first_mismatch"]


def test_full_edit_sequence_incremental_matches_blank_rebuild(bundle):
    def fresh_state():
        chain = copy.deepcopy(bundle["chain_e2"])
        return {
            "config": copy.deepcopy(bundle["config"]),
            "chain": chain,
            "operators": chain["operators"],
            "tokens": copy.deepcopy(bundle["tokens"]),
            "boundaries": copy.deepcopy(bundle["segments"]["boundaries"]),
        }

    incremental_state = fresh_state()
    store = CacheStore()
    state_to_engine(incremental_state, store=store, serial=True).run(with_width_reference=False)
    step_stats = []
    last_engine = None
    for command in bundle["edits"]["commands"]:
        apply_commands(incremental_state, [command])
        last_engine = state_to_engine(
            incremental_state,
            store=store,
            serial=True,
        )
        report = last_engine.run(with_width_reference=False)
        step_stats.append((command["op"], report["status"]))

    rerun = last_engine.run(with_width_reference=False)
    assert all(entry["action"] == "reused" for entry in rerun["reuse_ledger"])

    clean_state = fresh_state()
    apply_commands(clean_state, bundle["edits"]["commands"])
    clean = state_to_engine(
        clean_state,
        store=CacheStore(),
        serial=True,
    ).run(with_width_reference=False)

    comparison = compare_reports(rerun, clean)
    assert comparison["equal"], comparison["first_mismatch"]
    assert ("delete_token", "ok") in step_stats
    assert any(op == "add_operator" for op, _ in step_stats)
    assert any(op == "remove_operator" for op, _ in step_stats)


def test_serial_and_concurrent_and_budget_decisions_match(bundle):
    serial = make_engine(bundle, store=CacheStore(), serial=True).run(with_width_reference=False)
    concurrent = make_engine(bundle, store=CacheStore(), serial=False).run(with_width_reference=False)
    assert compare_reports(serial, concurrent, scheduler=True)["equal"]
    assert concurrent["scheduler"]["max_workers"] == bundle["config"]["max_workers"]
    assert concurrent["scheduler"]["max_observed_workers"] <= bundle["config"]["max_workers"]

    for overrides in ({"point_budget": 11}, {"operator_quota": 3}):
        left = make_engine(bundle, store=CacheStore(), serial=True, overrides=overrides).run(with_width_reference=False)
        right = make_engine(bundle, store=CacheStore(), serial=False, overrides=overrides).run(with_width_reference=False)
        result = compare_reports(left, right, scheduler=True)
        assert result["equal"], result["first_mismatch"]
        if "point_budget" in overrides:
            assert left["scheduler"]["status"] == "E_POINT_BUDGET_PENDING"
            assert left["scheduler"]["pending"]
        else:
            assert left["scheduler"]["status"] == "E_OP_QUOTA_EXCEEDED"
            assert any(d["code"] == "E_OP_QUOTA_EXCEEDED" for d in left["diagnostics"])


def test_fault_injections_hit_and_do_not_change_other_tokens(bundle):
    healthy = make_engine(bundle, store=CacheStore(), serial=True).run(with_width_reference=False)
    faulted = engine_from_data(
        config=bundle["config"],
        chain=bundle["chain_e2"],
        tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"],
        injections=bundle["injections"]["injections"],
        store=CacheStore(),
        serial=True,
    ).run(with_width_reference=False)

    assert {item["injection_id"] for item in faulted["fault_report"]} == {
        "INJ-DIM-001", "INJ-NAN-002", "INJ-SEG-003"
    }
    healthy_tokens = {t["token_uid"]: t for t in healthy["token_results"]}
    faulted_tokens = {t["token_uid"]: t for t in faulted["token_results"]}
    targeted = {"t0003", "t0005", "t0006"}
    for uid, token in faulted_tokens.items():
        if uid in targeted:
            assert token["status"] == "fault"
            assert token["fault_ids"]
        else:
            assert token["status"] == "ok"
            assert token["vector"] == healthy_tokens[uid]["vector"]
            assert token["cache_keys"] == healthy_tokens[uid]["cache_keys"]


def test_width_reference_marks_first_exceeded_stage(bundle):
    report = make_engine(bundle, store=CacheStore(), serial=True).run(with_width_reference=True)
    width = report["width_reference"]
    assert width["available"] is True
    assert width["reference_frac_bits"] == width["actual_frac_bits"] + 1
    assert width["stages"]
    assert width["first_exceeded_stage"] is not None
    assert width["first_exceeded_stage"]["kind"] == "attention"


def test_exchange_version_changes_cache_keys_but_not_integer_conclusion(bundle):
    e2 = engine_from_data(
        config=bundle["config"], chain=bundle["chain_e2"], tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"], store=CacheStore(), exchange_version="e2", serial=True,
    ).run(with_width_reference=False)
    e1 = engine_from_data(
        config=bundle["config"], chain=bundle["chain_e2"], tokens=bundle["tokens"],
        segments=bundle["segments"]["boundaries"], store=CacheStore(), exchange_version="e1", serial=True,
    ).run(with_width_reference=False)
    assert [t["vector"] for t in e1["token_results"]] == [t["vector"] for t in e2["token_results"]]
    e1_keys = [entry["cache_key"] for entry in e1["reuse_ledger"]]
    e2_keys = [entry["cache_key"] for entry in e2["reuse_ledger"]]
    assert e1_keys != e2_keys


def test_quant_tier_and_fraction_width_change_integer_conclusions(bundle):
    base = make_engine(bundle, store=CacheStore(), serial=True).run(with_width_reference=False)
    for overrides in ({"quant_tier": "q0"}, {"quant_tier": "q3"}, {"frac_bits": 16}):
        changed = make_engine(bundle, store=CacheStore(), serial=True, overrides=overrides).run(with_width_reference=False)
        assert any(
            left["vector"] != right["vector"]
            for left, right in zip(base["token_results"], changed["token_results"])
        ), overrides
        assert any(
            left["cache_keys"] != right["cache_keys"]
            for left, right in zip(base["token_results"], changed["token_results"])
        ), overrides
