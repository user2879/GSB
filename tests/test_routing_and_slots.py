from __future__ import annotations

import copy
from fractions import Fraction

from routing_workbench.config import parse_config
from routing_workbench.io_layer import parse_utterance
from routing_workbench.models import Utterance
from routing_workbench.routing import route_domain, route_intent
from routing_workbench.slots import SlotResult, fill_slots, recalibrate_confidence


def _base_config(domains, *, slots=None, patterns=None, negation=None, threshold="0.5", intent_threshold="0.5", **extra):
    vocabulary = set(extra.pop("vocabulary", []))
    raw = {
        "vocabulary": sorted(vocabulary),
        "domains": domains,
        "slots": slots or {},
        "slot_patterns": patterns or {},
        "negation_rules": negation or [],
        "thresholds": {"domain": threshold, "intent": intent_threshold},
        "fixed_decimals": 5,
        "quantization_levels": 100000,
        "quantization_scale": 100000,
        "thread_pool_size": 1,
        "max_thread_pool_size": 4,
        "batch_point_budget": 1000,
        "snapshot_interval_commands": 1,
        "recalibration": {"completeness_weight": "0.1", "suppression_penalty": "0.05"},
    }
    raw.update(extra)
    return parse_config(raw)


def _utterance(tokens, uid="U1"):
    return Utterance(uid, "S1", 1, tuple(tokens), raw={"utterance_id": uid, "session_id": "S1", "turn": 1, "tokens": tokens})


def test_domain_longest_match_suppression_and_lexicographic_tie() -> None:
    config = _base_config(
        {
            "zeta": {
                "triggers": [{"term": "重复扣费", "weight": 5}, {"term": "扣费", "weight": 5}],
                "intents": {"z": {"triggers": ["重复扣费"], "slot_patterns": []}},
            },
            "alpha": {
                "triggers": [{"term": "退款", "weight": 5}, {"term": "其他", "weight": 5}],
                "intents": {"a": {"triggers": ["退款"], "slot_patterns": []}},
            },
        },
        vocabulary=["重复扣费", "扣费", "退款", "其他"],
        threshold="0.5",
    )
    result = route_domain(_utterance(["退款", "重复扣费"]), config)
    assert result.selected == "alpha"
    assert result.score == Fraction(1, 2)
    assert result.boundary is True
    assert set(result.tied_names) == {"alpha", "zeta"}
    assert any(item["term"] == "扣费" and item["reason"] == "overlap_longest_match" for item in result.suppressed)
    converted = result.to_dict(config.quantization_scale, config.fixed_decimals)
    assert converted["score_int"] == 50000
    assert converted["score"] == "0.50000"


def test_intent_negation_cancel_and_downstream_selection() -> None:
    config = _base_config(
        {
            "alpha": {
                "triggers": [{"term": "工单", "weight": 1}],
                "intents": {
                    "a_intent": {"triggers": [{"term": "退款", "weight": 1}], "slot_patterns": []},
                    "b_intent": {"triggers": [{"term": "工单", "weight": 1}], "slot_patterns": []},
                },
            }
        },
        vocabulary=["工单", "退款", "不要"],
        negation=[{"negators": ["不要"], "scope": 1, "mode": "cancel", "direction": "before"}],
        threshold="0.5",
        intent_threshold="0.5",
    )
    domain = route_domain(_utterance(["不要", "退款", "工单"]), config)
    assert domain.selected == "alpha"
    intent = route_intent(_utterance(["不要", "退款", "工单"]), "alpha", config)
    assert intent.selected == "b_intent"
    assert any(item.get("reason") == "negation_cancel" for item in intent.suppressed)
    assert any(item.get("status") == "negated" for item in intent.matched)


def test_slot_candidate_longest_start_and_lexicographic_order() -> None:
    config = _base_config(
        {
            "account": {
                "triggers": [{"term": "手机号", "weight": 1}],
                "intents": {
                    "change_phone": {
                        "triggers": [{"term": "手机号", "weight": 1}],
                        "slot_patterns": ["phone_short", "phone_long"],
                        "required_slots": ["phone"],
                    }
                },
            }
        },
        slots={"phone": {"type": "phone"}},
        patterns={
            "phone_short": {"name": "phone_short", "intent": "change_phone", "slot": "phone", "tokens": ["手机号", "{phone}"], "window": 2},
            "phone_long": {"name": "phone_long", "intent": "change_phone", "slot": "phone", "tokens": ["手机号", "请", "记录", "{phone}"], "window": 4},
        },
        vocabulary=["手机号", "请", "记录", "13800138000", "13900139000"],
    )
    tokens = ["手机号", "请", "记录", "13800138000", "手机号", "13900139000"]
    utterance = _utterance(tokens)
    intent = config.domains["account"].intents["change_phone"]
    result = fill_slots(utterance, intent, config)
    assert result.slots == {"phone": "13800138000"}
    assert result.completeness == 1
    assert len(result.candidates) == 1
    assert len(result.suppression) == 2
    assert {item["value"] for item in result.suppression} == {"请", "13900139000"}
    assert result.suppression[0]["reason"] in {"shorter_match", "lower_rank"}


def test_recalibration_weight_and_quantization_change_fixed_integer() -> None:
    config = _base_config(
        {"d": {"triggers": ["x"], "intents": {"i": {"triggers": ["x"], "slot_patterns": [], "required_slots": []}}}},
        vocabulary=["x"],
    )
    slots = SlotResult({"phone": "1"}, ("phone",), (), Fraction(1), [], [{"value": "2"}])
    base = Fraction(1, 2)
    low = recalibrate_confidence(base, slots, config)
    raw = copy.deepcopy(config.raw)
    raw["recalibration"]["completeness_weight"] = "0.30"
    high = recalibrate_confidence(base, slots, parse_config(raw))
    assert high.after_q > low.after_q
    raw2 = copy.deepcopy(config.raw)
    raw2["fixed_decimals"] = 4
    raw2["quantization_levels"] = 10000
    raw2["quantization_scale"] = 10000
    quantized = recalibrate_confidence(base, slots, parse_config(raw2))
    assert quantized.scale == 10000
    assert quantized.decimals == 4
    assert quantized.after_q != low.after_q
    assert quantized.to_dict()["after"].count(".") == 1
    assert len(quantized.to_dict()["after"].split(".")[1]) == 4
