from __future__ import annotations

from reco_shard_diag.diagnostics import run_injections
from reco_shard_diag.fixtures import build_demo_config
from reco_shard_diag.models import ProjectConfig
from reco_shard_diag.state import materialize


def test_all_injections_hit_and_runtime_failures_are_isolated():
    config = ProjectConfig.from_raw(build_demo_config(include_events=True))
    injections = run_injections(materialize(config))
    hits = {entry["injection_id"] for entry in injections if entry["hit"]}
    assert hits == {
        "I_DEPENDENCY_CYCLE",
        "I_QUOTA_PARENT_MISSING",
        "I_SAMPLE_DIMENSION",
        "I_STALE_LEASE",
        "I_EXPIRED_RENEW",
    }
    assert all(entry.get("isolated") for entry in injections)
    expired = next(
        entry for entry in injections if entry["injection_id"] == "I_EXPIRED_RENEW"
    )
    assert expired["active_after_sweep"] == 0
    assert expired["slot_reused"] is True
