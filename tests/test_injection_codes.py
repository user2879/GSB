import pytest

from lease_middleware.faults import run_injection


EXPECTED = {
    "spec": "E_SPEC",
    "lease": "E_LEASE",
    "snapshot": "E_SNAPSHOT",
    "verify": "E_VERIFY",
    "port": "E_PORT",
    "timeout": "E_TIMEOUT",
    "reply": "E_REPLY",
}


@pytest.mark.parametrize("kind", sorted(EXPECTED))
def test_injection_reports_stable_code(root, kind):
    result = run_injection(kind, root)
    assert result["ok"] is False
    assert result["expected_code"] == EXPECTED[kind]
    assert result["error"]["code"] == EXPECTED[kind]
    if kind == "lease":
        assert result["error"]["details"]["current_tag"] == 2
