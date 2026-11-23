"""Fixed-point math, scoring, relations and priority boundary rules."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from escalation_arbiter.fixedpoint import (clamp_low, decay_coeff,
                                           mul_permille, quantize,
                                           round_half_up_div, scale_factor)
from escalation_arbiter.models import (ArbConfig, Evidence, PriorityTier,
                                       RelationEntry, RulePackage, Ticket)
from escalation_arbiter.stages import compute_ticket


def test_half_up_division_signs():
    assert round_half_up_div(5, 2) == 3
    assert round_half_up_div(-5, 2) == -3
    assert round_half_up_div(4, 2) == 2
    assert round_half_up_div(-4, 2) == -2


def test_decay_half_life_changes_coefficient():
    fast = decay_coeff(90000, 90000)
    slow = decay_coeff(90000, 45000)
    assert fast == 500
    assert slow == 250
    assert decay_coeff(0, 90000) == 1000


def test_quantize_and_floor():
    value, residual = quantize(107, 5)
    assert value == 105
    assert residual == 2
    assert clamp_low(-3, 0) == 0


def test_ticket_scoring_and_mutex_bonus():
    rules = RulePackage(
        type_weights={"LOGIN": 800, "DEVICE": 600},
        role_cred={"system": 900},
        mutex=[RelationEntry("1", "mutex", "LOGIN", "DEVICE")],
        support=[],
        tiers=[PriorityTier(1, 1, "LOW"), PriorityTier(1000, 2, "HIGH")],
        reason_dict={"MX1": "login_device_conflict"},
    )
    config = ArbConfig(
        decimals=3, quantum=1, half_life_ms=1000,
        deduct_permille=500, bonus_permille=1200, fusion="sum",
        zero_floor=0, budget_ms=500, concurrency=1,
        snapshot_interval=3, conf_min_permille=0,
        conf_max_permille=1000,
    )
    ticket = Ticket("T1", [
        Evidence("T1", 1, "LOGIN", "system", 0, ("a", "b"), 1000),
        Evidence("T1", 2, "DEVICE", "system", 1000, ("c", "d"), 1000),
    ])
    decision = compute_ticket(ticket, rules, config)
    assert decision.reason_codes == ["MX1"]
    assert decision.boundary in (True, False)
    assert all(score >= 0 for score in
               decision.final_evidence_scores.values())
    for esc in decision.evidence_scores:
        assert esc.round_dir_1 in ("exact", "up", "down")


def test_boundary_ticket_flag():
    rules = RulePackage(
        type_weights={"LOGIN": 500},
        role_cred={"system": 1000},
        mutex=[], support=[],
        tiers=[PriorityTier(400, 1, "LOW")],
        reason_dict={},
    )
    config = ArbConfig(
        decimals=3, quantum=1, half_life_ms=1000,
        deduct_permille=1000, bonus_permille=1000, fusion="sum",
        zero_floor=0, budget_ms=500, concurrency=1,
        snapshot_interval=3, conf_min_permille=0,
        conf_max_permille=1000,
    )
    ticket = Ticket("T1", [
        Evidence("T1", 1, "LOGIN", "system", 0, ("a",), 800),
    ])
    decision = compute_ticket(ticket, rules, config)
    assert decision.decision_score == 400
    assert decision.boundary is True
    assert decision.priority_level == 1
