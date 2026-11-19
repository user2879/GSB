"""共享夹具：固定策略集合与临时工作目录。"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audit_engine.budget import BudgetConfig
from audit_engine.generator import generate_all
from audit_engine.models import NormalizationPolicy

POLICIES = [
    NormalizationPolicy(fold_case=True, unicode_nfc=True, resolve_links=True, name="strict"),
    NormalizationPolicy(fold_case=False, unicode_nfc=True, resolve_links=True, name="case_sensitive"),
    NormalizationPolicy(fold_case=True, unicode_nfc=False, resolve_links=True, name="raw_unicode"),
    NormalizationPolicy(fold_case=True, unicode_nfc=True, resolve_links=False, name="no_link_resolve"),
]


@pytest.fixture(scope="session")
def policies() -> list[NormalizationPolicy]:
    return POLICIES


@pytest.fixture()
def manifests(tmp_path):
    directory = tmp_path / "manifests"
    paths = generate_all(str(directory))
    return {os.path.basename(path).replace(".json", ""): path for path in paths}


@pytest.fixture()
def budget() -> BudgetConfig:
    return BudgetConfig(
        concurrency=1,
        per_file_timeout_seconds=30.0,
        max_records_per_file=100_000,
    )
