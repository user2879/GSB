from qu.loader import RewriteRule
from qu.rewrite import RewriteEngine


def _rules():
    return [
        RewriteRule("r-syn", "synonym", "耳机", "耳麦", 1.0, ("耳机",)),
        RewriteRule("r-exp", "category_expand", "手机", "通讯数码",
                    2.0, ("手机",)),
        RewriteRule("r-drop", "drop_recover", "保修", "全国联保",
                    1.5, ("保修",)),
    ]


def test_suggestions_sorted_and_capped():
    engine = RewriteEngine(_rules(), max_suggestions=2)
    outcome = engine.generate("手机 耳机", prune=False)
    benefits = [item["estimated_benefit"] for item in outcome.suggestions]
    assert benefits == sorted(benefits, reverse=True)
    assert len(outcome.suggestions) <= 2
    # Drop-recover fires because the required term is absent.
    assert any(item["rule_id"] == "r-drop" for item in outcome.full_candidates)


def test_pruned_path_equivalent_and_cheaper():
    engine = RewriteEngine(_rules(), max_suggestions=8)
    full = engine.generate("手机 耳机", prune=False)
    pruned = engine.generate("手机 耳机", prune=True)
    full_ids = [item["rule_id"] for item in full.full_candidates]
    pruned_ids = [item["rule_id"] for item in pruned.pruned_candidates]
    assert pruned_ids == full_ids
    assert pruned.suggestions == full.suggestions
    assert pruned.pruned_rule_checks <= full.full_rule_checks
    assert pruned.pruning_diagnosis["pruning_lost_suggestion"] is False
    assert pruned.pruning_diagnosis["conclusion"].startswith("pruned path")


def test_cap_vs_pruning_diagnosis_separated():
    engine = RewriteEngine(_rules(), max_suggestions=1)
    outcome = engine.generate("手机 耳机", prune=False)
    diagnosis = outcome.pruning_diagnosis
    assert diagnosis["dropped_by_cap_rule_ids"]
    assert diagnosis["missing_due_to_pruning_rule_ids"] == []
    assert diagnosis["pruning_lost_suggestion"] is False
    assert len(diagnosis["item_by_item_comparison"]) == diagnosis[
        "full_candidate_count"]
