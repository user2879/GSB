from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdet_render.errors import BusinessError, E_LEASE_EXPIRED, E_LEASE_GENERATION
from mdet_render.lease import LocalExecutor, ManualClock
from mdet_render.renderer import Renderer

from conftest import ledger_entries, load_workspace


def test_lease_generation_mismatch_rejected_without_advancing() -> None:
    clock = ManualClock(100)
    executor = LocalExecutor(clock=clock)
    lease = executor.acquire([0, 1], ttl_ms=50)
    generation = executor.generation
    with pytest.raises(BusinessError) as caught:
        executor.renew(lease.lease_id, lease.generation + 1, ttl_ms=50)
    assert caught.value.code == E_LEASE_GENERATION
    assert executor.generation == generation
    assert len(executor.occupied_units()) == 2
    with pytest.raises(BusinessError) as caught:
        executor.release(lease.lease_id, lease.generation + 1)
    assert caught.value.code == E_LEASE_GENERATION
    assert executor.generation == generation
    assert len(executor.occupied_units()) == 2


def test_expiry_reclaims_units_immediately_and_emits_code() -> None:
    clock = ManualClock(0)
    executor = LocalExecutor(clock=clock)
    lease = executor.acquire([3, 4, 5], ttl_ms=10)
    clock.advance(10)
    events = executor.reclaim_expired()
    assert [event.code for event in events] == [E_LEASE_EXPIRED]
    assert executor.occupied_units() == []
    assert executor.generation == lease.generation
    clock.advance(1)
    replacement = executor.acquire([6], ttl_ms=20)
    assert replacement.generation > lease.generation


def test_expired_renew_fails_and_releases_resources() -> None:
    clock = ManualClock(0)
    executor = LocalExecutor(clock=clock)
    lease = executor.acquire([0, 1], ttl_ms=5)
    clock.advance(6)
    with pytest.raises(BusinessError) as caught:
        executor.renew(lease.lease_id, lease.generation, ttl_ms=5)
    assert caught.value.code == E_LEASE_EXPIRED
    assert executor.occupied_units() == []
    assert any(event["code"] == E_LEASE_EXPIRED for event in executor.events())


def test_renderer_outputs_expiry_and_leaves_no_occupied_units(sample_root: Path, tmp_path: Path) -> None:
    workspace = load_workspace(sample_root)
    clock = ManualClock(0)
    executor = LocalExecutor(clock=clock)
    ledger = tmp_path / "ledger.jsonl"
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=ledger, executor=executor)
    first = renderer.run(output_dir=tmp_path / "out")
    assert first.complete
    assert executor.occupied_units()
    clock.advance(workspace.config.lease_ttl_ms + 1)
    with pytest.raises(BusinessError) as caught:
        renderer.run(output_dir=tmp_path / "out")
    assert caught.value.code == E_LEASE_EXPIRED
    assert executor.occupied_units() == []
    events = [entry for entry in ledger_entries(ledger)
              if entry.get("event") == "event" and entry.get("code") == E_LEASE_EXPIRED]
    assert events
    renderer.close()


def test_close_releases_lease_resources(sample_root: Path, tmp_path: Path) -> None:
    workspace = load_workspace(sample_root)
    renderer = Renderer(workspace, cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.jsonl")
    result = renderer.run(output_dir=tmp_path / "out")
    assert result.complete
    assert renderer.executor.occupied_units()
    renderer.close()
    assert renderer.executor.occupied_units() == []
