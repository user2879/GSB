from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from graygate.spec import GatewaySpec, load_spec, parse_spec


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SPEC_PATH = ROOT / "samples" / "gateway_spec.json"


@pytest.fixture
def sample_spec() -> GatewaySpec:
    return load_spec(SAMPLE_SPEC_PATH)


@pytest.fixture
def raw_sample_spec() -> dict:
    return json.loads(SAMPLE_SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def mutate_raw_spec(raw_sample_spec):
    def mutate(mutator):
        value = copy.deepcopy(raw_sample_spec)
        mutator(value)
        return parse_spec(value)

    return mutate

