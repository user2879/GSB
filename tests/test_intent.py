from qu.config import PipelineConfig
from qu.intent import IntentScorer
from qu.loader import RewriteRule, VocabEntry


def test_score_formula_normalization_and_tie_break():
    vocab = [
        VocabEntry("alpha", "手机", 2.0),
        VocabEntry("beta", "耳机", 2.0),
    ]
    config = PipelineConfig(position_decay=0.5)
    scorer = IntentScorer(vocab, [], config)
    result = scorer.score("手机耳机", "full_scan")
    # Both terms appear once at positions 0 and 2 with equal length/weight;
    # alpha decays at position 0, beta at position 2 -> alpha wins.
    assert result.primary_intent == "alpha"
    assert result.normalized_scores[result.primary_intent] == 0.0
    assert all(score <= 0.0 for score in result.normalized_scores.values())
    assert len(result.contributions) == 2


def test_lexicographic_tie_break_and_determinism():
    vocab = [
        VocabEntry("zeta", "手机", 1.0),
        VocabEntry("alpha", "手机", 1.0),
    ]
    scorer = IntentScorer(vocab, [], PipelineConfig())
    first = scorer.score("手机", "full_scan")
    second = scorer.score("手机", "full_scan")
    assert first.primary_intent == "alpha"
    assert first.raw_scores == second.raw_scores
    assert first.contributions == second.contributions
    assert first.normalized_scores == second.normalized_scores


def test_rule_bonus_and_scan_modes_equivalent():
    vocab = [
        VocabEntry("alpha", "手机", 1.0),
        VocabEntry("beta", "平板", 1.0),
    ]
    rules = [RewriteRule(
        rule_id="r1", rule_type="category_expand", trigger_term="手机",
        replacement="通讯", weight=1.5, ref_terms=("手机",))]
    config = PipelineConfig(rule_bonus_weight=2.0)
    scorer = IntentScorer(vocab, rules, config)
    full = scorer.score("手机 平板", "full_scan")
    pruned = scorer.score("手机 平板", "first_char_pruned")
    assert full.raw_scores == pruned.raw_scores
    assert full.normalized_scores == pruned.normalized_scores
    assert full.primary_intent == pruned.primary_intent
    assert full.contributions == pruned.contributions
    assert full.rule_bonuses == pruned.rule_bonuses
    assert pruned.hit_checks <= full.hit_checks
    # rule bonus = weight(1.5) * rule_bonus_weight(2.0)
    assert full.raw_scores["alpha"] > full.raw_scores["beta"]
    assert full.rule_bonuses[0]["bonus"] == 3.0
