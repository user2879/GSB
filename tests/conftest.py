from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcp_workbench.config import load_session_config
from tcp_workbench.service import SessionService
from tcp_workbench.structured_log import StructuredLogger

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture
def sample_config_script():
    return load_session_config(SAMPLES / "session.json")


@pytest.fixture
def sample_events() -> list[dict]:
    return json.loads((SAMPLES / "events.json").read_text(encoding="utf-8"))["events"]


@pytest.fixture
def service_factory(tmp_path):
    created: list[SessionService] = []

    def factory(name: str = "ledger.sqlite3", **kwargs) -> SessionService:
        config, script = load_session_config(SAMPLES / "session.json")
        logger = StructuredLogger(tmp_path / f"{name}.jsonl")
        service = SessionService.open(
            tmp_path / name, config, script, logger=logger, **kwargs
        )
        created.append(service)
        return service

    yield factory
    for service in created:
        try:
            service.close()
        except Exception:  # noqa: BLE001, S110
            pass
