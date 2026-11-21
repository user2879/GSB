import pytest

from workbench import scoring


def test_text_score_hit_decomposition(dataset):
    turn = dataset["turns"][0]
    result = scoring.score_text(turn, dataset["lexicon"])
    assert result["term_count"] == len(turn["terms"])
    expected_hits = [t for t in turn["terms"] if t in dataset["lexicon"]]
    assert [h["term"] for h in result["hits"]] == expected_hits
    total = sum(dataset["lexicon"][t] for t in expected_hits)
    assert result["score"] == pytest.approx(total / len(turn["terms"]))


def test_scoring_is_deterministic(dataset):
    turn = dataset["turns"][3]
    acoustic = dataset["acoustics"][turn["turn_id"]]
    tool = dataset["tools"]["sentiment_polarity"]
    params = turn["tool_calls"][0]["params"]
    first = (scoring.score_text(turn, dataset["lexicon"]),
             scoring.score_acoustic(acoustic, ["0.25", "0.5", "0.75"]),
             scoring.call_tool(tool, params))
    second = (scoring.score_text(turn, dataset["lexicon"]),
              scoring.score_acoustic(acoustic, ["0.25", "0.5", "0.75"]),
              scoring.call_tool(tool, params))
    assert first == second


def test_tool_result_carries_fingerprint_and_dimension(dataset):
    tool = dataset["tools"]["sentiment_polarity"]
    result = scoring.call_tool(tool, {"pos_count": 2, "neg_count": 1, "term_count": 6})
    assert result["dimension"] == "sentiment_polarity"
    assert len(result["formula_fingerprint"]) == 64
    assert result["value"] == pytest.approx((2 - 1) / 6 * 0.5 + 0.5)


def test_formula_evaluator_rejects_unsafe_constructs():
    with pytest.raises(ValueError):
        scoring.evaluate_formula("__import__('os').system('id')", {})
    with pytest.raises(ValueError):
        scoring.evaluate_formula("unknown_name + 1", {})
    with pytest.raises(ValueError):
        scoring.evaluate_formula("1 / 0", {})
