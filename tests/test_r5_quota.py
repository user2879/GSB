import copy

import pytest

from ckpt_diag.errors import LeaseError
from ckpt_diag.quota import (
    OWNER_ACTIVE, OWNER_EXPIRED, OWNER_FREE, OWNER_VOID, QuotaArbiter)


def _arbiter():
    return QuotaArbiter(budget=4, shard_lease_caps={"a": 3, "b": 2})


def test_grant_tracks_unique_owner():
    arbiter = _arbiter()
    units = arbiter.grant(event_seq=1, shard_id="a", lease_id="L1",
                          lease_version=1, param_version=1, cost_units=2)
    assert units == (0, 1)
    assert arbiter.state.free_capacity() == 2
    for index in units:
        assert arbiter.state.units[index].owner_kind == OWNER_ACTIVE


def test_shard_lease_cap_enforced():
    arbiter = _arbiter()
    arbiter.grant(event_seq=1, shard_id="b", lease_id="L1", lease_version=1,
                  param_version=1, cost_units=2)
    with pytest.raises(LeaseError):
        arbiter.grant(event_seq=2, shard_id="b", lease_id="L2", lease_version=1,
                      param_version=1, cost_units=1)


def test_global_budget_enforced():
    arbiter = QuotaArbiter(budget=4, shard_lease_caps={"a": 8, "b": 8})
    arbiter.grant(event_seq=1, shard_id="a", lease_id="L1", lease_version=1,
                  param_version=1, cost_units=4)
    with pytest.raises(LeaseError):
        arbiter.grant(event_seq=2, shard_id="a", lease_id="L2", lease_version=1,
                      param_version=1, cost_units=1)


def test_expired_unit_cannot_also_be_voided():
    arbiter = _arbiter()
    units = arbiter.grant(event_seq=1, shard_id="a", lease_id="L1",
                          lease_version=1, param_version=1, cost_units=2)
    arbiter.expire_lease(event_seq=2, lease_id="L1", lease_version=1,
                         param_version=1)
    assert arbiter.state.units[units[0]].owner_kind == OWNER_EXPIRED
    with pytest.raises(LeaseError) as exc:
        arbiter.void_param_version(event_seq=3, shard_id="a", param_version=1,
                                   lease_id="L1", lease_version=1,
                                   unit_indices=units)
    assert exc.value.code == "E_LEASE"


def test_void_then_reclaim_returns_to_free():
    arbiter = _arbiter()
    units = arbiter.grant(event_seq=1, shard_id="a", lease_id="L1",
                          lease_version=1, param_version=1, cost_units=2)
    arbiter.void_param_version(event_seq=2, shard_id="a", param_version=1,
                               lease_id="L1", lease_version=1, unit_indices=units)
    assert all(arbiter.state.units[i].owner_kind == OWNER_VOID for i in units)
    freed = arbiter.reclaim_to_free(event_seq=3, unit_indices=units,
                                    basis="recycled")
    assert freed == units
    assert arbiter.state.free_capacity() == 4


def test_no_unit_held_by_two_shards():
    arbiter = _arbiter()
    units = arbiter.grant(event_seq=1, shard_id="a", lease_id="L1",
                          lease_version=1, param_version=1, cost_units=2)
    owners = {arbiter.state.units[i].shard_id for i in units}
    assert owners == {"a"}


def test_replay_prefixes_produce_identical_ownership():
    arbiter = _arbiter()
    arbiter.grant(event_seq=1, shard_id="a", lease_id="L1", lease_version=1,
                  param_version=1, cost_units=2)
    arbiter.grant(event_seq=2, shard_id="b", lease_id="L2", lease_version=1,
                  param_version=1, cost_units=1)
    snapshot_full = copy.deepcopy(arbiter.snapshot())
    arbiter.expire_lease(event_seq=3, lease_id="L2", lease_version=1,
                         param_version=1)
    # rebuild from scratch replaying only first two events
    replay = _arbiter()
    replay.grant(event_seq=1, shard_id="a", lease_id="L1", lease_version=1,
                 param_version=1, cost_units=2)
    replay.grant(event_seq=2, shard_id="b", lease_id="L2", lease_version=1,
                 param_version=1, cost_units=1)
    assert replay.snapshot() == snapshot_full


def test_decisions_record_basis_seq_versions():
    arbiter = _arbiter()
    arbiter.grant(event_seq=1, shard_id="a", lease_id="L1", lease_version=1,
                  param_version=7, cost_units=1)
    decision = arbiter.state.decisions[0]
    assert decision.event_seq == 1
    assert decision.param_version == 7
    assert decision.lease_version == 1
    assert decision.basis == "grant_from_free_pool"
    assert decision.before_owner == OWNER_FREE
