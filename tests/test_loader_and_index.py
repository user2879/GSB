from __future__ import annotations

import json
from pathlib import Path

from search_proto.errors import E_INDEX_INPUT
from search_proto.index import build_index
from search_proto.loader import load_inputs
from search_proto.util import write_json

from .conftest import config_dict, make_config, make_docs, make_stopwords


def _write_jsonl(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_loader_records_every_invalid_row(workspace_tmp: Path) -> None:
    write_json(workspace_tmp / "config.json", config_dict())
    write_json(
        workspace_tmp / "vocab.json",
        {"version": "v1", "terms": ["alpha", "beta", "gamma", "the"]},
    )
    write_json(
        workspace_tmp / "stopwords.json",
        {"version": "s1", "global": [], "by_field": {"title": ["the"]}},
    )
    valid_document = {
        "doc_id": "D1",
        "version": 0,
        "fields": {"title": ["alpha"], "body": ["beta"]},
    }
    _write_jsonl(
        workspace_tmp / "documents.jsonl",
        [
            valid_document,
            valid_document,
            {
                "doc_id": "D2",
                "version": -1,
                "fields": {"abstract": ["alpha"]},
            },
            {
                "doc_id": "D3",
                "version": 0,
                "fields": {"title": ["not-in-vocab"], "body": []},
            },
        ],
    )
    with (workspace_tmp / "documents.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
    _write_jsonl(
        workspace_tmp / "queries.jsonl",
        [
            {"query_id": "Q1", "terms": ["alpha"], "field_preferences": ["title"]},
            {"query_id": "Q1", "terms": ["beta"], "field_preferences": ["title"]},
            {"query_id": "Q2", "terms": [], "field_preferences": ["title"]},
            {"query_id": "Q3", "terms": ["alpha"], "field_preferences": ["abstract"]},
            {"query_id": "Q4", "terms": ["outside"], "field_preferences": ["title"]},
        ],
    )
    _write_jsonl(
        workspace_tmp / "edits.jsonl",
        [
            {"op": "not-an-op"},
            {"op": "set_fixed", "decimal_places": True, "quantum": 1},
        ],
    )

    inputs = load_inputs(
        workspace_tmp / "config.json",
        workspace_tmp / "vocab.json",
        workspace_tmp / "stopwords.json",
        workspace_tmp / "documents.jsonl",
        workspace_tmp / "queries.jsonl",
        workspace_tmp / "edits.jsonl",
    )

    assert len(inputs.documents) == 1
    assert len(inputs.queries) == 1
    assert inputs.edits == ()
    assert inputs.failures
    assert all(failure.code == E_INDEX_INPUT for failure in inputs.failures)
    details = {failure.detail_code for failure in inputs.failures}
    assert {
        "DOC_DUPLICATE",
        "DOC_VERSION",
        "DOC_FIELD_NAME",
        "DOC_TERM_NOT_IN_VOCAB",
        "JSON_PARSE",
        "QUERY_DUPLICATE",
        "QUERY_TERMS_EMPTY",
        "QUERY_FIELD_NAME",
        "QUERY_TERM_NOT_IN_VOCAB",
        "EDIT_UNKNOWN_OP",
        "EDIT_FIXED_DECIMAL_PLACES",
    }.issubset(details)
    assert [failure.source for failure in inputs.failures] == sorted(
        failure.source for failure in inputs.failures
    )


def test_index_positions_exclusions_and_determinism() -> None:
    config = make_config()
    stopwords = make_stopwords()
    documents = make_docs()

    serial = build_index(documents, config, stopwords, max_workers=1)
    threaded = build_index(documents, config, stopwords, max_workers=4)
    assert serial.index_fingerprint == threaded.index_fingerprint
    assert serial.to_dict() == threaded.to_dict()

    d1 = serial.document_stats["D0001"]
    assert d1["fields"]["title"]["length"] == 2
    assert d1["fields"]["title"]["terms"]["alpha"]["positions"] == [0, 2]
    assert d1["fields"]["title"]["excluded_positions"] == [
        {
            "doc_id": "D0001",
            "field": "title",
            "position": 1,
            "term": "the",
            "reasons": ["stopword_table"],
        }
    ]
    assert d1["fields"]["body"]["excluded_positions"] == [
        {
            "doc_id": "D0001",
            "field": "body",
            "position": 1,
            "term": "delta",
            "reasons": ["document_mark"],
        }
    ]
    alpha = serial.term_stats["alpha"]
    assert alpha["df"] == 3
    assert alpha["df_by_field"] == {"title": 2, "body": 1}
    assert [posting["doc_id"] for posting in serial.postings["alpha"].values()] == [
        "D0001",
        "D0002",
        "D0003",
    ]
