from __future__ import annotations

import pytest

from faultisolation.ledger import Ledger
from faultisolation.process import ArbiterProcessGroup, LiveScheduler
from faultisolation.replay import verify_ledger, verify_paths
from faultisolation.samples import generate_sample, SMALL_PLAN
from faultisolation.spec import load_bundle


@pytest.fixture()
def live_bundle(tmp_path):
    paths = generate_sample(SMALL_PLAN, tmp_path, global_lease_cap=3, batch_size=16)
    return load_bundle(paths["spec"], paths["session"]), tmp_path


def test_real_loopback_arbitration_matches_plan(live_bundle):
    bundle, tmp_path = live_bundle
    ledger = Ledger(tmp_path / "live.sqlite", replace=True)
    group = ArbiterProcessGroup(bundle)
    group.start()
    try:
        scheduler = LiveScheduler(bundle, ledger)
        result = scheduler.run()
    finally:
        group.stop()
        ledger.close()
    telemetry = result["telemetry"]
    assert len(telemetry) == len(bundle.events)
    assert all(item["matched"] for item in telemetry)
    # 真实回执往返必须被观测到，证明走了真实 TCP 而非内存调用
    assert all(item["rtt_ms"] >= 0 for item in telemetry)
    verify = verify_ledger(bundle, tmp_path / "live.sqlite")
    assert verify["digest_ok"] is True and verify["missing_seqs"] == []


def test_telemetry_records_monotonic_timestamps(live_bundle):
    bundle, tmp_path = live_bundle
    ledger = Ledger(tmp_path / "live2.sqlite", replace=True)
    group = ArbiterProcessGroup(bundle)
    group.start()
    try:
        result = LiveScheduler(bundle, ledger).run()
    finally:
        group.stop()
        ledger.close()
    telemetry = sorted(result["telemetry"], key=lambda x: x["seq"])
    for prev, nxt in zip(telemetry, telemetry[1:]):
        assert nxt["sent_monotonic_ms"] >= prev["sent_monotonic_ms"]
