"""Requirement 1: loading, validation and E_REC_INVALID failure list."""

from __future__ import annotations

import json

from rag_app.errors import E_REC_INVALID
from rag_app.loader import load_documents, load_query_stream


def test_valid_samples_have_no_failures(sample_dir):
    docs, doc_failures = load_documents(sample_dir / "documents.jsonl")
    turns, query_failures = load_query_stream(sample_dir / "queries.jsonl")
    assert len(docs) == 60
    assert len(turns) == 16
    assert doc_failures == []
    assert query_failures == []


def test_duplicate_doc_id_and_empty_body_are_collected(tmp_path):
    path = tmp_path / "docs.jsonl"
    rows = [
        {"doc_id": "D1", "title": "t", "body": "hello world", "source": "s"},
        {"doc_id": "D1", "title": "dup", "body": "again", "source": "s"},
        {"doc_id": "D2", "title": "blank", "body": "   ", "source": "s"},
        {"doc_id": "D3", "title": "ok", "body": "valid body", "source": "s"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    docs, failures = load_documents(path)
    assert [doc.doc_id for doc in docs] == ["D1", "D3"]
    assert [failure.code for failure in failures] == [
        E_REC_INVALID,
        E_REC_INVALID,
    ]
    assert all(failure.sort_key for failure in failures)


def test_broken_json_line_does_not_abort_processing(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps(
            {"doc_id": "D1", "title": "t", "body": "ok", "source": "s"}
        )
        + "\n{not-json\n"
        + json.dumps(
            {"doc_id": "D2", "title": "t", "body": "fine", "source": "s"}
        )
    )
    docs, failures = load_documents(path)
    assert [doc.doc_id for doc in docs] == ["D1", "D2"]
    assert len(failures) == 1 and failures[0].code == E_REC_INVALID


def test_turn_gap_and_followup_without_context_are_rejected(tmp_path):
    path = tmp_path / "queries.jsonl"
    rows = [
        {
            "session_id": "S1",
            "turn_no": 1,
            "query": "q1",
            "is_followup": False,
            "context_ref": [],
        },
        {
            "session_id": "S1",
            "turn_no": 3,
            "query": "q3",
            "is_followup": False,
            "context_ref": [],
        },
        {
            "session_id": "S2",
            "turn_no": 1,
            "query": "q",
            "is_followup": True,
            "context_ref": [],
        },
        {
            "session_id": "S3",
            "turn_no": 1,
            "query": "q",
            "is_followup": False,
            "context_ref": [9],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    turns, failures = load_query_stream(path)
    assert [(t.session_id, t.turn_no) for t in turns] == [("S1", 1)]
    assert len(failures) == 3
    assert {failure.code for failure in failures} == {E_REC_INVALID}


def test_context_ref_must_point_to_earlier_turn(tmp_path):
    path = tmp_path / "queries.jsonl"
    rows = [
        {
            "session_id": "S1",
            "turn_no": 1,
            "query": "q1",
            "is_followup": False,
            "context_ref": [],
        },
        {
            "session_id": "S1",
            "turn_no": 2,
            "query": "q2",
            "is_followup": True,
            "context_ref": [1],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    turns, failures = load_query_stream(path)
    assert len(turns) == 2 and failures == []
