from fractions import Fraction

from arg_review.adjudication import adjudicate_event
from arg_review.engine import ReviewEngine
from arg_review.quantization import quantize_score
from arg_review.samplegen import generate_sample_tree
from arg_review.state import load_workspace


def _run_sample(tmp_path, config_mutator=None, name="state"):
    root = tmp_path / "samples"
    generate_sample_tree(root)
    state = load_workspace(root)
    if config_mutator:
        config_mutator(state.config)
    return ReviewEngine(state, tmp_path / name).run()


def test_required_missing_and_mutex_suppression(tmp_path):
    result = _run_sample(tmp_path)
    decisions = {
        d["event_id"]: d
        for d in result["stage_outputs"]["adjudication"]["decisions"]
    }
    assert decisions["E006"]["status"] == "incomplete"
    assert decisions["E006"]["missing_required_roles"] == ["host"]
    e5 = decisions["E005"]
    assert e5["status"] == "complete"
    assert len(e5["suppressed_candidates"]) == 2
    assert {x["role"] for x in e5["suppressed_candidates"]} == {"guest"}
    assert e5["mutex_decisions"][0]["winner_role"] == "host"


def test_equal_confidence_mutex_tie_uses_role_ascending():
    config = {
        "decision": {"accept_threshold_permille": 600},
        "quantization": {"frac_digits": 3, "level": "fine",
                         "level_multipliers": {"fine": 1}},
    }

    def candidate(candidate_id, role):
        score = Fraction(8, 10)
        quantized = quantize_score(score, config["quantization"])
        return {
            "artifact_id": candidate_id,
            "arg_id": candidate_id,
            "role": role,
            "start": 1,
            "end": 2,
            "cache_key": candidate_id,
            "eligible": True,
            "score_numerator": score.numerator,
            "score_denominator": score.denominator,
            "quantized_score_numerator": quantized["quantized_fixed_int"],
            "quantized_score_denominator": quantized["scale"],
            "quantization": quantized,
        }

    decision = adjudicate_event(
        {"event_id": "E-TIE"},
        {"candidates": [candidate("arg:E:z", "zeta"), candidate("arg:E:a", "alpha")]},
        {"required_roles": [], "mutex_groups": [["zeta", "alpha"]]},
        {"revision_id": "r", "version": 1},
        config,
    )
    mutex = decision["mutex_decisions"][0]
    assert mutex["winner_role"] == "alpha"
    assert decision["suppressed_candidates"][0]["role"] == "zeta"


def test_boundary_candidate_outputs_higher_width_reference(tmp_path):
    result = _run_sample(tmp_path, name="boundary")
    candidate = next(
        c
        for item in result["stage_outputs"]["arguments"]["results"]
        for c in item["candidates"]
        if c["arg_id"] == "E007-A2"
    )
    score = Fraction(candidate["score_numerator"], candidate["score_denominator"])
    threshold = (
        2 * score.numerator * 1000 + score.denominator
    ) // (2 * score.denominator)

    def set_threshold(config):
        config["decision"]["accept_threshold_permille"] = threshold

    boundary_result = _run_sample(tmp_path, set_threshold, name="boundary-run")
    decision = next(
        d
        for d in boundary_result["stage_outputs"]["adjudication"]["decisions"]
        if d["event_id"] == "E007"
    )
    boundary = next(
        b for b in decision["boundary_candidates"] if b["arg_id"] == "E007-A2"
    )
    assert boundary["boundary"] is True
    assert boundary["high_width_digits"] == 4
    assert "." in boundary["high_width_reference_value"]


def test_reference_merge_groups_and_threshold(tmp_path):
    result = _run_sample(tmp_path)
    groups = result["stage_outputs"]["references"]["merge_groups"]
    alpha_group = next(g for g in groups if "E005" in g["source_event_ids"])
    assert alpha_group["source_event_ids"] == ["E001", "E003", "E005"]
    assert any(
        set(g["source_event_ids"]) == {"E001", "E003"}
        and len(g["members"]) == 2
        for g in groups
    )
    edges = result["stage_outputs"]["references"]["edge_decisions"]
    assert [e["processing_order"] for e in edges] == list(range(len(edges)))

    def high_threshold(config):
        config["merge"]["dedup_threshold_permille"] = 999

    strict = _run_sample(tmp_path, high_threshold, name="strict")
    references = strict["stage_outputs"]["references"]
    assert references["merge_count"] == 0
    assert any(
        e["business_code"] == "E_DEDUP_BELOW_THRESHOLD"
        for e in references["edge_decisions"]
    )
