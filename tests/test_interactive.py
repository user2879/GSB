from __future__ import annotations

import time
from pathlib import Path

from iidx.config import AppConfig
from iidx.engine import RunResult
from iidx.interactive import InteractiveSession


class SlowEngine:
    delay = 0.0

    def __init__(self, *, config, **kwargs):
        self.config = config

    def run(self) -> RunResult:
        time.sleep(type(self).delay)
        report = {
            "run": {
                "status": "complete",
                "global_version": 1,
                "config_hash": self.config.config_hash,
            },
            "config": self.config.to_dict(),
            "summary": {
                "completed_shards": ["shard-a"],
                "partial_shards": [],
                "pending_shards": [],
                "block_count": 2,
                "compression_tier": self.config.compression_tier,
                "affected_shards": ["shard-a"],
                "failure_count": 0,
            },
            "failures": [],
            "queries": [],
            "blocks": [],
            "budget_ledger": [],
        }
        return RunResult(report=report, exit_code=0, interrupted=False, report_path=Path("report.json"))


def test_interactive_latency_budget_returns_latest_available_summary(tmp_path: Path) -> None:
    config = AppConfig(interactive_latency_budget_ms=500, thread_pool_size=1, concurrency_quota=1)
    session = InteractiveSession(
        data_dir=tmp_path,
        state_dir=tmp_path / "state",
        config=config,
        engine_factory=SlowEngine,
    )
    first = session.execute_command("run")
    assert first["code"] == "OK"
    assert first["summary"]["compression_tier"] == "auto"

    SlowEngine.delay = 0.12
    started = time.monotonic()
    budget_change = session.execute_command("latency 20")
    elapsed = time.monotonic() - started
    assert elapsed < 0.11
    assert budget_change["code"] == "E_INTERACT_BUDGET"

    started = time.monotonic()
    response = session.execute_command("tier 2")
    elapsed = time.monotonic() - started
    assert elapsed < 0.11
    assert response["code"] == "E_INTERACT_BUDGET"
    assert response["summary"]["compression_tier"] == "auto"
    time.sleep(0.45)
    status = session.execute_command("status")
    assert status["code"] == "OK"
    assert status["summary"]["compression_tier"] == 2
    assert status["summary"]["block_count"] == 2
    session.close()
