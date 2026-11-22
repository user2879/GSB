import pytest

from faultdiag.converge import check_domain_convergence
from faultdiag.dispatch import CommandResult
from faultdiag.errors import DiagError
from faultdiag.spec import Event
from faultdiag.state import Engine


def _result(instance, domain, isolated, cmd_seq=1, attempt=1, status="ok"):
    receipt = {"instance": instance, "cmd_seq": cmd_seq,
               "isolated": isolated, "elapsed_ms": 0} if status == "ok" else None
    return CommandResult(cmd_seq, instance, domain, "isolate", 1, attempt,
                         status, receipt, None, 0, 0, 0)


def _isolating_engine(instances, domains):
    eng = Engine(instances, domains, mono_clock=lambda: 1)
    eng.apply_event(Event(1, 0, "s-a1", "crash"))
    return eng


def test_converge_ok(instances, domains):
    eng = _isolating_engine(instances, domains)
    results = [_result("s-a1", "dom-a", True)]
    view = check_domain_convergence(eng, "dom-a", results)
    assert view["instances"]["s-a1"]["isolated"] is True
    assert view["instances"]["s-a2"]["isolated"] is False


def test_converge_mismatch(instances, domains):
    eng = _isolating_engine(instances, domains)
    results = [_result("s-a1", "dom-a", False)]
    with pytest.raises(DiagError) as exc:
        check_domain_convergence(eng, "dom-a", results)
    assert exc.value.code == "E_CONVERGE"
    dev = exc.value.details["deviations"]
    assert dev[0]["instance"] == "s-a1"
    assert dev[0]["expected"] is True and dev[0]["actual"] is False
    assert exc.value.details["expected_digest"] != exc.value.details["actual_digest"]


def test_converge_missing_receipt(instances, domains):
    eng = _isolating_engine(instances, domains)
    results = [_result("s-a1", "dom-a", False, status="E_REPLY")]
    with pytest.raises(DiagError) as exc:
        check_domain_convergence(eng, "dom-a", results)
    assert exc.value.code == "E_CONVERGE"
    assert exc.value.details["deviations"][0]["actual"] == "<missing>"


def test_converge_retry_distinguished(instances, domains):
    eng = _isolating_engine(instances, domains)
    results = [_result("s-a1", "dom-a", False, status="E_REPLY", attempt=1),
               _result("s-a1", "dom-a", True, cmd_seq=1, attempt=2)]
    view = check_domain_convergence(eng, "dom-a", results)
    assert view["instances"]["s-a1"]["isolated"] is True
