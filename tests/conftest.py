from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from leasebench.models import CallEvent, Declaration, SessionConfig
from leasebench.validation import load_declaration, load_events, parse_declaration


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


@pytest.fixture
def declaration_v1() -> Declaration:
    return load_declaration(SAMPLES / "declarations_v1.json")


@pytest.fixture
def declaration_v2() -> Declaration:
    return load_declaration(SAMPLES / "declarations_v2.json")


@pytest.fixture
def events(declaration_v1: Declaration) -> list[CallEvent]:
    return load_events(SAMPLES / "events.ndjson", declaration_v1)


@pytest.fixture
def session() -> SessionConfig:
    return SessionConfig(watermark_scale=1000, borrow_return_error_ms=50)


def dump_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def dump_events(path: Path, events: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def parse_with_functions(
    raw: dict[str, Any], function_order: list[int]
) -> Declaration:
    mutated = json.loads(json.dumps(raw))
    mutated["functions"] = [mutated["functions"][index] for index in function_order]
    return parse_declaration(mutated)

