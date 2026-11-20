from qu.config import PipelineConfig
from qu.entity import EntityMatcher
from qu.errors import E_EMPTY_QUERY
from qu.experiment import _ReferenceDecision
from qu.intent import IntentScorer
from qu.loader import load_bundle
from qu.pipeline import QueryPipeline, sort_key
from qu.rewrite import RewriteEngine
from qu.runner import TierRunner


def _build(generated, config):
    bundle = load_bundle(generated["samples"], generated["vocab"],
                         generated["entities"], generated["rules"])
    matcher = EntityMatcher(bundle.entities, config.dict_priority)
    scorer = IntentScorer(bundle.vocab, bundle.rules, config)
    rewriter = RewriteEngine(bundle.rules, config.max_rewrites)
    return bundle, QueryPipeline(config, matcher, scorer, rewriter,
                                 _ReferenceDecision())


def test_stage_exception_isolation(generated, config):
    bundle, pipeline = _build(generated, config)
    result = pipeline.process("q-empty", "!!!？？", {}, {})
    assert result.ok is False
    assert result.failures[0]["code"] == E_EMPTY_QUERY
    assert result.failures[0]["stage"] == "normalize"
    assert result.failures[0]["exception_type"] == "EmptyQueryError"
    assert result.sort_key == sort_key("q-empty", "result")


def test_serial_parallel_field_equality(generated):
    config = PipelineConfig(max_workers=4)
    bundle, pipeline = _build(generated, config)
    runner = TierRunner(pipeline, max_workers=4)
    outcome = runner.run(bundle.samples)
    summary = outcome["comparison"]
    assert summary["equal"] is True
    assert summary["field_mismatch_count"] == 0
    assert summary["ordering_equal"] is True
    serial_ids = [r.query_id for r in outcome["results"]]
    parallel_ids = [r.query_id for r in outcome["parallel_results"]]
    assert serial_ids == parallel_ids


def test_different_worker_counts_stay_identical(generated):
    bundle = load_bundle(generated["samples"], generated["vocab"],
                         generated["entities"], generated["rules"])
    payloads = []
    for workers in (1, 2, 8):
        config = PipelineConfig(max_workers=workers)
        _, pipeline = _build(generated, config)
        runner = TierRunner(pipeline, workers)
        outcome = runner.run(bundle.samples[:40])
        payloads.append([r.to_dict() for r in outcome["results"]])
    semantic_keys = (
        lambda record: (record["query_id"], record["ok"],
                        record["normalized"], record["entity"]["slots"],
                        record["entity"]["suppressed"],
                        record["intent"]["primary_intent"],
                        record["intent"]["normalized_scores"],
                        record["rewrite"]["suggestions"]))
    first = [semantic_keys(item) for item in payloads[0]]
    for other in payloads[1:]:
        assert [semantic_keys(item) for item in other] == first
