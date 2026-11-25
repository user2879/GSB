from lease_middleware.crash import run_crash_demo
from lease_middleware.service import run_demo
from lease_middleware.spec import load_sample_dir
from lease_middleware.ledger import Ledger
from lease_middleware.verify import verify_ledger


def test_demo_byte_identical_and_ledger_verifies(root):
    report = run_demo(root)
    assert report["comparison"]["byte_identical"] is True
    assert report["forward"]["ledger"]["unknown_leases"] == []
    forward = root / "artifacts" / "demo" / "order-forward" / "ledger.sqlite"
    bundle = load_sample_dir(root / "samples")
    ledger = Ledger.open_readonly(forward)
    try:
        verified = verify_ledger(bundle, ledger)
        assert verified["verified_records"] == 32
    finally:
        ledger.close()


def test_force_kill_recovery_marks_unknown_lease(root):
    report = run_crash_demo(root)
    assert report["victim_returncode"] == -9
    unknown = report["recovery"]["unknown_leases"]
    assert [item["lease_id"] for item in unknown] == ["lease-crash"]
    assert report["snapshot_recovery"]["replaced"] is True
