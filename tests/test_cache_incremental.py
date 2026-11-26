import copy

from decchain.cache import ConclusionCache
from decchain.commands import Command, CommandApplier
from decchain.config import DecodeConfig
from decchain.models import Step
from decchain.pipeline import DecodePipeline


def _order(requests):
    return {r.request_id: i + 1 for i, r in
            enumerate(sorted(requests, key=lambda r: r.request_id))}


def _run(requests, config):
    cache = ConclusionCache()
    pipeline = DecodePipeline(config, cache, _order(requests))
    outcomes = []
    for request in sorted(requests, key=lambda r: r.request_id):
        for seq_id, step in request.all_steps():
            seq = request.sequence_map()[seq_id]
            outcomes.append(pipeline.process_step(request, seq, step))
    return outcomes, cache


def test_unchanged_step_is_cache_hit(simple_request, config):
    outcomes_first, cache = _run([simple_request], config)
    pipeline = DecodePipeline(config, cache, _order([simple_request]))
    seq = simple_request.sequences[0]
    cached = pipeline.process_step(simple_request, seq, seq.steps[0])
    assert cached.from_cache is True
    assert any(e.kind == "HIT" for e in cache.events)


def test_replaced_step_only_invalidates_itself(simple_request, config):
    _, cache = _run([simple_request], config)
    before_fp = simple_request.content_fingerprint()
    applier = CommandApplier([simple_request], config)
    applier.apply(Command.from_dict({
        "command": "replace_step", "request_id": "R001", "seq_id": 1,
        "step_index": 2,
        "logprobs": {100: -0.05, 101: -0.8, 102: -1.5, 103: -2.4,
                     104: -3.3}}))
    cache.invalidate_step("R001", before_fp, config.fingerprint(), 1, 2)

    pipeline = DecodePipeline(config, cache, _order([simple_request]))
    seq = simple_request.sequences[0]
    step1 = pipeline.process_step(simple_request, seq, seq.steps[0])
    step2 = pipeline.process_step(simple_request, seq, seq.steps[1])
    step3 = pipeline.process_step(simple_request, seq, seq.steps[2])
    assert step1.from_cache and step3.from_cache
    assert not step2.from_cache


def test_append_only_adds_new_conclusion(simple_request, config):
    _, cache = _run([simple_request], config)
    applier = CommandApplier([simple_request], config)
    applier.apply(Command.from_dict({
        "command": "append_step", "request_id": "R001", "seq_id": 1,
        "step_index": 4,
        "logprobs": {100: -0.3, 101: -1.0, 102: -2.0, 103: -2.8,
                     104: -3.6}}))
    pipeline = DecodePipeline(config, cache, _order([simple_request]))
    seq = simple_request.sequences[0]
    outcomes = [pipeline.process_step(simple_request, seq, st)
                for st in seq.steps]
    assert all(o.from_cache for o in outcomes[:3])
    assert not outcomes[3].from_cache


def test_config_change_changes_cache_key(simple_request, config):
    _, cache = _run([simple_request], config)
    new_config = config.with_replacements(width=5)
    pipeline = DecodePipeline(new_config, cache, _order([simple_request]))
    seq = simple_request.sequences[0]
    outcome = pipeline.process_step(simple_request, seq, seq.steps[0])
    assert not outcome.from_cache
