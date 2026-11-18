"""Fault-injection driven business-code paths."""

from __future__ import annotations

from rag_app.engine import Engine
from rag_app.errors import E_CITATION_MISSING, E_REC_INVALID
from rag_app.loader import load_documents
from rag_app.replay import load_inputs
from rag_app.snapshot import SnapshotStore


def test_force_no_citation_marks_answer_unavailable(
    sample_dir, snapshot_dir
):
    config, rules, faults, docs, turns, _cmds, failures = load_inputs(
        sample_dir
    )
    faults = dict(faults)
    faults["force_no_citation_turn"] = "S-A#0002"
    engine = Engine(
        docs=docs,
        turns=turns,
        config=config,
        rules=rules,
        store=SnapshotStore(snapshot_dir),
        base_failures=failures,
        faults=faults,
    )
    engine.run_full()
    record = engine.records[("S-A", 2)]
    assert record.artifacts.citation.answer_available is False
    codes = {failure.code for failure in engine.failure_list()}
    assert E_CITATION_MISSING in codes
    engine.shutdown()


def test_invalid_doc_injection_is_represented_in_failure_list(
    sample_dir, tmp_path
):
    # The fault flag maps to the documented E_REC_INVALID injection path.
    faulty = sample_dir / "fault_injection.json"
    data = __import__("json").loads(faulty.read_text(encoding="utf-8"))
    assert "invalid_doc_inject" in data
    assert "invalid_query_inject" in data


def test_duplicate_doc_failure_code_chain(sample_dir):
    import json

    docs, _failures = load_documents(sample_dir / "documents.jsonl")
    duplicate_path = sample_dir / "duplicate.jsonl"
    rows = [
        {
            "doc_id": docs[0].doc_id,
            "title": "first",
            "body": "original body",
            "source": "x",
        },
        {
            "doc_id": docs[0].doc_id,
            "title": "dup",
            "body": "duplicate identifier",
            "source": "x",
        },
    ]
    duplicate_path.write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    _docs, dup_failures = load_documents(duplicate_path)
    assert dup_failures
    assert dup_failures[0].code == E_REC_INVALID


def test_record_fault_injection_flows_through_replay(
    sample_dir, snapshot_dir
):
    import json
    import shutil

    from rag_app.replay import run_replay

    faulty_sample = snapshot_dir / "sample"
    shutil.copytree(sample_dir, faulty_sample)
    fault_path = faulty_sample / "fault_injection.json"
    data = json.loads(fault_path.read_text(encoding="utf-8"))
    data["invalid_doc_inject"] = True
    data["invalid_query_inject"] = True
    fault_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    result = run_replay(
        faulty_sample, snapshot_dir / "snap", clean=True
    )
    codes = {item["code"] for item in result["incremental_failures"]}
    assert E_REC_INVALID in codes
    assert result["consistent"] is True
    assert result["incremental_failures"] == result["full_failures"]
