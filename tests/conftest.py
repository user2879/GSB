from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from search_proto.config import PrototypeConfig, validate_config
from search_proto.model import ControlledVocabulary, Document, QuerySpec, StopwordTable


@pytest.fixture
def workspace_tmp() -> Path:
    path = Path.cwd() / "artifacts" / "test-tmp" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def config_dict(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "fields": {"title": {"weight": 2.0}, "body": {"weight": 1.0}},
        "bm25": {"k1": 1.2, "b": 0.75},
        "fixed": {"decimal_places": 3, "quantum": 1},
        "allowed_decimal_places": [2, 3, 4, 5],
        "allowed_quantums": [1, 2, 5, 10],
        "top_k": 2,
        "max_workers": 4,
        "batch_point_budget": 1000,
        "per_query_point_quota": 1000,
        "stage_tolerance_units": 0.5,
    }
    value.update(overrides)
    return value


def make_config(**overrides: Any) -> PrototypeConfig:
    return validate_config(config_dict(**overrides))


def make_vocab(terms: list[str] | None = None) -> ControlledVocabulary:
    return ControlledVocabulary.create(
        "test-vocab",
        terms or ["alpha", "beta", "gamma", "delta", "the", "of", "zeta"],
    )


def make_docs() -> list[Document]:
    return [
        Document.create(
            "D0001",
            0,
            {"title": ["alpha", "the", "alpha"], "body": ["beta", "delta"]},
            {"body": [1]},
        ),
        Document.create(
            "D0002",
            1,
            {"title": ["gamma"], "body": ["alpha", "beta", "the"]},
        ),
        Document.create(
            "D0003",
            0,
            {"title": ["delta", "alpha"], "body": ["zeta"]},
        ),
    ]


def make_queries() -> list[QuerySpec]:
    return [
        QuerySpec.create("Q0001", ["alpha", "beta"], ["title", "body"]),
        QuerySpec.create("Q0002", ["delta"], ["title"]),
    ]


def make_stopwords() -> StopwordTable:
    return StopwordTable.create(
        "stop-v1", by_field={"title": ["the"], "body": ["the", "of"]}
    )

