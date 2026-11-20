from qu.config import PipelineConfig
from qu.strategy import StrategySelector


def test_strategy_escalates_with_scale():
    selector = StrategySelector(PipelineConfig(latency_budget_ms=10.0))
    small = selector.choose(query_count=100, vocab_size=100,
                            dict_size=100, rule_count=10, avg_query_len=8)
    medium = selector.choose(query_count=2_000, vocab_size=5_000,
                             dict_size=8_000, rule_count=100,
                             avg_query_len=10)
    large = selector.choose(query_count=10_000, vocab_size=20_000,
                            dict_size=30_000, rule_count=300,
                            avg_query_len=12)
    assert small.entity_mode == "exhaustive"
    assert medium.entity_mode == "inverted"
    assert large.entity_mode == "prefix_group"
    assert small.estimated_latency_ms < large.estimated_latency_ms
    for decision in (small, medium, large):
        assert decision.estimated_costs
        assert decision.metrics["dict_size"] in (100, 8000, 30000)
        assert decision.fingerprint
        assert "regime" in decision.reason


def test_decision_is_auditable_and_deterministic():
    selector = StrategySelector(PipelineConfig())
    kwargs = dict(query_count=500, vocab_size=800, dict_size=1200,
                  rule_count=20, avg_query_len=9)
    first = selector.choose(**kwargs)
    second = selector.choose(**kwargs)
    assert first.to_dict() == second.to_dict()
    assert all("estimated_latency_ms" in item
               for item in first.estimated_costs)


def test_cache_enabled_at_large_query_scale():
    selector = StrategySelector(PipelineConfig())
    decision = selector.choose(query_count=20_000, vocab_size=20_000,
                               dict_size=30_000, rule_count=300,
                               avg_query_len=12)
    assert decision.cache_enabled is True
