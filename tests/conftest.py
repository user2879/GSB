from __future__ import annotations

import json
from pathlib import Path

import pytest

from faultisolation.samples import generate_sample
from faultisolation.spec import load_bundle


@pytest.fixture(scope="session")
def small_plan():
    from faultisolation.samples import SMALL_PLAN
    return SMALL_PLAN


@pytest.fixture()
def small_bundle(tmp_path):
    paths = generate_sample(
        __import__("faultisolation.samples", fromlist=["SMALL_PLAN"]).SMALL_PLAN,
        tmp_path, global_lease_cap=3, batch_size=16)
    return load_bundle(paths["spec"], paths["session"])
