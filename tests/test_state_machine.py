import pytest

from lease_middleware.engine import ReplayEngine
from lease_middleware.errors import LeaseError
from lease_middleware.spec import SpecBundle
from lease_middleware.statemachine import match_version_tag, require_transition


def test_explicit_permission_table():
    assert require_transition("granted", "renew") == "renewed"
    assert require_transition("renewed", "renew") == "renewed"
    assert require_transition("granted", "release") == "released"
    assert require_transition("renewed", "expire") == "expired"
    with pytest.raises(LeaseError):
        require_transition("released", "renew")


def test_tag_rollback_and_skip_report_current_tag():
    assert match_version_tag(lease_id="l", current_tag=3, requested_tag=3) == 3
    for bad_tag in (2, 5):
        with pytest.raises(LeaseError) as exc:
            match_version_tag(lease_id="l", current_tag=3, requested_tag=bad_tag)
        assert exc.value.code == "E_LEASE"
        assert exc.value.details["current_tag"] == 3


def test_stale_renew_does_not_extend_expiry(bundle: SpecBundle):
    engine = ReplayEngine(bundle)
    grant = next(event for event in bundle.events if event["seq"] == 1)
    renew = next(event for event in bundle.events if event["seq"] == 6)
    engine.apply_event(grant)
    before = dict(engine.leases["lease-a1"])
    stale_event = dict(renew)
    stale_event["version_tag"] = 99
    with pytest.raises(LeaseError):
        engine.apply_event(stale_event)
    after = engine.leases["lease-a1"]
    assert after["generation"] == before["generation"]
    assert after["expires_at_ms"] == before["expires_at_ms"]
    assert after["state"] == before["state"]


def test_sample_chain_monotonic(bundle: SpecBundle):
    result = ReplayEngine(bundle).process_all()
    for lease in result["leases"]:
        generations = [item["generation_after"] for item in lease["history"]]
        assert generations == sorted(generations)
        assert len(generations) == len(set(generations))
        for item in lease["history"]:
            if item["matched_tag"] is not None:
                assert item["matched_tag"] == item["generation_before"]
                assert item["generation_after"] == item["generation_before"] + 1
