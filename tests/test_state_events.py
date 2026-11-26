from __future__ import annotations

from reco_shard_diag.fixtures import build_demo_config
from reco_shard_diag.models import ProjectConfig
from reco_shard_diag.state import materialize, replay_events


def test_materialized_final_state_covers_full_event_set():
    config = ProjectConfig.from_raw(build_demo_config(include_events=True))
    state = materialize(config)
    assert not state.errors
    assert state.active_quota_version == 2
    assert state.width == 12
    assert set(state.shards) == {"shard-a", "shard-b", "shard-c", "shard-d"}
    assert state.shards["shard-a"].batches[-1].batch_id == "a2"
    assert state.shards["shard-c"].dimensions == 2
    assert state.shards["shard-d"].upstream == ["shard-a", "shard-c"]
    assert state.effective_node("node-c").capacity_slots == 3
    assert "shard-e" not in state.shards
    assert state.quotas[4].parent_version == 3
    assert state.shard_revisions["shard-a"] >= 2


def test_replay_events_from_base_produces_same_state():
    base_raw = build_demo_config(include_events=False)
    source_raw = build_demo_config(include_events=True)
    source_raw["nodes"] = base_raw["nodes"]
    config = ProjectConfig.from_raw(base_raw)
    source = ProjectConfig.from_raw(source_raw)
    replayed = replay_events(config, source.events)
    direct = materialize(source)
    assert replayed.state_fingerprint == direct.state_fingerprint
