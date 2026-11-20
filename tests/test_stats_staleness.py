"""Fixed-point error bounds and config-evolution staleness protection."""

import json

from gateway.engine import run
from gateway.ledger import Ledger
from gateway.stats import average_ms, canary_distribution, ratio_bp


def test_ratio_floor_and_bound():
    r = ratio_bp(1, 3)
    assert r.value_bp == 3333  # floor(10000/3)
    assert r.error_bound_bp == 1
    assert r.rounding == "floor"
    # true value 3333.33.. lies in [value, value + bound)
    assert 3333 <= 10000 / 3 < 3333 + 1


def test_average_floor_and_bound():
    a = average_ms(10, 3)
    assert a.value == 3
    assert a.error_bound == 1


def test_distribution_sums_within_bound(sample):
    ledger = Ledger(sample["db"])
    result = run(sample["backends"], sample["rules"], sample["requests"],
                 ledger, global_limit=2, forward_repeat=5)
    rows = ledger.ledger_rows(result.config_version)
    ledger.close()
    dist = canary_distribution(rows)
    for variants in dist.values():
        total = variants["stable"].value_bp + variants["canary"].value_bp
        assert 10000 - 2 <= total <= 10000  # floor truncation loses < 2bp


def test_weight_change_bumps_version_and_stales_old(sample, tmp_path):
    ledger = Ledger(sample["db"])
    r1 = run(sample["backends"], sample["rules"], sample["requests"],
             ledger, global_limit=2, forward_repeat=5)
    assert r1.config_version == 1
    assert r1.stale_versions == []

    rules = json.loads(open(sample["rules"]).read())
    rules["rules"][0]["canary_weight_bp"] = 9000
    path = tmp_path / "rules_v2.json"
    path.write_text(json.dumps(rules))

    r2 = run(sample["backends"], str(path), sample["requests"],
             ledger, global_limit=2, forward_repeat=5)
    assert r2.config_version == 2  # monotonic bump
    assert r2.stale_versions == [1]

    # Current view contains ONLY v2 rows; v1 rows persist but are excluded.
    current = ledger.ledger_rows(2)
    stale = ledger.ledger_rows(1)
    assert len(current) == len(r2.routed)
    assert len(stale) == len(r1.routed)
    assert all(row["config_version"] == 2 for row in current)
    snaps = ledger.snapshots()
    assert [s["config_version"] for s in snaps] == [1, 2]
    ledger.close()


def test_identical_config_keeps_version(sample):
    ledger = Ledger(sample["db"])
    r1 = run(sample["backends"], sample["rules"], sample["requests"],
             ledger, global_limit=2, forward_repeat=5)
    r2 = run(sample["backends"], sample["rules"], sample["requests"],
             ledger, global_limit=2, forward_repeat=5)
    assert r1.config_version == r2.config_version == 1
    ledger.close()
