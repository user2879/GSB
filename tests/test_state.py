import pytest

from faultdiag.errors import DiagError
from faultdiag.spec import Event
from faultdiag.state import Engine, apply_fault


def test_apply_fault_rules():
    assert apply_fault("unknown", "crash") == "unhealthy"
    assert apply_fault("healthy", "degrade") == "degraded"
    assert apply_fault("unhealthy", "degrade") == "unhealthy"
    assert apply_fault("healthy", "flap") == "degraded"
    assert apply_fault("degraded", "flap") == "degraded"
    assert apply_fault("unhealthy", "recover") == "healthy"
    with pytest.raises(DiagError):
        apply_fault("healthy", "bogus")


def test_engine_isolation_and_cooldown(instances, domains):
    eng = Engine(instances, domains, mono_clock=lambda: 1)
    intents = eng.apply_event(Event(1, 0, "s-a1", "crash"))
    assert eng.health["s-a1"] == "unhealthy"
    assert eng.domain_state["dom-a"] == "isolating"
    assert [i.instance for i in intents] == ["s-a1"]
    eng.confirm_isolated("dom-a")
    assert eng.domain_state["dom-a"] == "isolated"
    # 冷却 300ms 未结束(仅过 100ms), 不得转 recovering
    intents = eng.apply_event(Event(2, 100, "s-a1", "recover"))
    assert eng.domain_state["dom-a"] == "isolated"
    assert intents == []
    # 冷却结束后恢复
    intents = eng.apply_event(Event(3, 400, "s-a1", "crash"))
    assert eng.domain_state["dom-a"] == "isolated"  # 重新不健康
    intents = eng.apply_event(Event(4, 500, "s-a1", "recover"))
    assert eng.domain_state["dom-a"] == "recovering"
    assert [i.action for i in intents] == ["recover"]
    eng.confirm_normal("dom-a")
    assert eng.domain_state["dom-a"] == "normal"
    kinds = [(t["kind"], t["entity"], t["to"]) for t in eng.transitions]
    assert ("domain", "dom-a", "isolating") in kinds
    assert all(t["mono_ms"] == 1 for t in eng.transitions)
    assert all(t["event_seq"] >= 1 for t in eng.transitions)


def test_engine_majority_threshold(instances, domains):
    eng = Engine(instances, domains, mono_clock=lambda: 1)
    eng.apply_event(Event(1, 0, "s-b1", "crash"))
    assert eng.domain_state["dom-b"] == "normal"  # 权重 1/2 未过半
    eng.apply_event(Event(2, 1, "s-b2", "crash"))
    assert eng.domain_state["dom-b"] == "isolating"


def test_engine_level_injection(instances, domains):
    import dataclasses
    domains["dom-a"] = dataclasses.replace(domains["dom-a"], isolation_level=99)
    eng = Engine(instances, domains, mono_clock=lambda: 1)
    with pytest.raises(DiagError) as exc:
        eng.apply_event(Event(1, 0, "s-a1", "crash"))
    assert exc.value.code == "E_LEVEL"


def test_engine_event_injection(instances, domains):
    eng = Engine(instances, domains, mono_clock=lambda: 1)
    with pytest.raises(DiagError) as exc:
        eng.apply_event(Event(1, 0, "s-a1", "__bogus__"))
    assert exc.value.code == "E_EVENT"
