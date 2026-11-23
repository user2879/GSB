from __future__ import annotations

import copy
from pathlib import Path

import pytest

from leaseaudit.clock import ManualClock
from leaseaudit.engine import LeaseEngine
from leaseaudit.ledger import AuditLedger
from leaseaudit.scenario import Scenario, validate_scenario


@pytest.fixture
def scenario_doc() -> dict:
    return {
        "format_version": 1,
        "scenario_version": "test-v1",
        "lease_params_version": 7,
        "cluster_capacity": 12,
        "lease_duration_ms": 1000,
        "renew_interval_ms": 200,
        "reclaim_threshold": 250000,
        "reclaim_window_rounds": 2,
        "observation_window_ms": 100,
        "watermark_scale": 1_000_000,
        "load_duration_ms": 5000,
        "sources": [
            {
                "source_id": "alpha",
                "address": "127.0.0.1",
                "port": 55501,
                "lease_share": 4,
                "reserve_quota": 2,
                "mode": "success",
                "processing_ms": 0,
                "request_interval_ms": 100,
            },
            {
                "source_id": "beta",
                "address": "127.0.0.1",
                "port": 55502,
                "lease_share": 3,
                "reserve_quota": 1,
                "mode": "fixed_delay",
                "processing_ms": 5,
                "request_interval_ms": 100,
            },
            {
                "source_id": "gamma",
                "address": "127.0.0.1",
                "port": 55503,
                "lease_share": 2,
                "reserve_quota": 1,
                "mode": "refuse",
                "processing_ms": 0,
                "request_interval_ms": 100,
            },
        ],
    }


@pytest.fixture
def scenario(scenario_doc: dict) -> Scenario:
    return validate_scenario(scenario_doc)


@pytest.fixture
def ledger_bundle(tmp_path: Path, scenario: Scenario):
    ledger_path = tmp_path / "audit.sqlite"
    ledger = AuditLedger(ledger_path, scenario=scenario, verify=True)
    clock = ManualClock(value_ms=1_000_000)
    engine = LeaseEngine(scenario, ledger, clock=clock)
    try:
        yield {
            "path": ledger_path,
            "ledger": ledger,
            "engine": engine,
            "clock": clock,
            "scenario": scenario,
            "tmp_path": tmp_path,
        }
    finally:
        engine.close()


def clone_doc(document: dict) -> dict:
    return copy.deepcopy(document)
