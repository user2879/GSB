"""Fingerprint / batching / routing determinism across load order and
concurrency settings."""

import json

from gateway.batching import build_plan, select_strategy
from gateway.fingerprint import fingerprints
from gateway.loader import load_declaration
from gateway.routing import route_requests, stable_bucket


def _shuffled(path, tmp_path, key):
    data = json.loads(open(path).read())
    data[key] = list(reversed(data[key]))
    out = tmp_path / f"shuffled_{key}.json"
    out.write_text(json.dumps(data))
    return str(out)


def test_fingerprint_invariant_to_load_order(sample, tmp_path):
    decl_a = load_declaration(sample["backends"], sample["rules"],
                              sample["requests"])
    decl_b = load_declaration(_shuffled(sample["backends"], tmp_path,
                                        "backends"),
                              sample["rules"], sample["requests"])
    assert fingerprints(decl_a) == fingerprints(decl_b)


def test_batch_plan_invariant_to_declaration_order(sample, tmp_path):
    decl_a = load_declaration(sample["backends"], sample["rules"],
                              sample["requests"])
    decl_b = load_declaration(_shuffled(sample["backends"], tmp_path,
                                        "backends"),
                              sample["rules"], sample["requests"])
    for decl in (decl_a, decl_b):
        routed = route_requests(decl)
        decision = select_strategy(decl, routed)
        plan = build_plan(decl, routed, decision.strategy)
        if decl is decl_a:
            plan_a = plan
        else:
            plan_b = plan
    assert plan_a == plan_b


def test_routing_stable_for_same_client_and_weight(sample):
    decl = load_declaration(sample["backends"], sample["rules"],
                            sample["requests"])
    first = [(r.request.seq, r.backend_id, r.variant)
             for r in route_requests(decl)]
    second = [(r.request.seq, r.backend_id, r.variant)
              for r in route_requests(decl)]
    assert first == second
    # same client key + rule -> same bucket every time
    assert stable_bucket("c:client-1", "r1") == stable_bucket("c:client-1", "r1")
    assert 0 <= stable_bucket("c:client-1", "r1") < 10000


def test_strategy_selection_deterministic(sample):
    decl = load_declaration(sample["backends"], sample["rules"],
                            sample["requests"])
    routed = route_requests(decl)
    d1 = select_strategy(decl, routed)
    d2 = select_strategy(decl, routed)
    assert d1 == d2
    assert d1.tie_threshold == 1
