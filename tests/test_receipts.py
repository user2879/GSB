import json

from conftest import ROOT
from qcalib.errors import E_RECEIPT_MALFORMED, FailureList
from qcalib.receipts import build_receipt_ledger, parse_receipt_dir


def test_parse_three_versions_and_malformed():
    failures = FailureList()
    entries = parse_receipt_dir(ROOT / "receipts" / "set_a", failures)
    versions = sorted({entry.version for entry in entries})
    assert versions == ["q1", "q2", "q3"]
    q1 = next(entry for entry in entries
              if entry.version == "q1" and entry.layer == "embedding_proj")
    assert "param_version" in q1.missing_fields
    assert "checksum" in q1.missing_fields
    q3 = next(entry for entry in entries if entry.version == "q3")
    assert q3.override_refs is not None
    assert any(failure.code == E_RECEIPT_MALFORMED for failure in failures)


def test_adjudication_rules_q3_override_then_elapsed_then_param():
    failures = FailureList()
    entries = parse_receipt_dir(ROOT / "receipts" / "set_a", failures)
    ledger = build_receipt_ledger(entries)
    embedding = ledger["embedding_proj"]
    assert embedding["winner"]["version"] == "q3"
    assert embedding["adjudication"]["rule"] == "q3_override_ref"
    fusion = ledger["fusion_block"]
    assert fusion["winner"]["version"] == "q2"
    assert fusion["adjudication"]["rule"] == "elapsed_ms_desc"
    retrieval = ledger["retrieval_head"]
    assert retrieval["winner"]["source_file"] == "receipt_q1.json"
    assert retrieval["adjudication"]["rule"] == "filename_asc"


def test_param_version_desc_without_override(tmp_path):
    records = [
        {"version": "q2", "layer": "x", "coverage_permille": 900,
         "param_version": 1, "elapsed_ms": 5},
        {"version": "q2", "layer": "x", "coverage_permille": 950,
         "param_version": 7, "elapsed_ms": 5},
    ]
    (tmp_path / "a.json").write_text(
        json.dumps({"entries": records}), encoding="utf-8")
    failures = FailureList()
    entries = parse_receipt_dir(tmp_path, failures)
    ledger = build_receipt_ledger(entries)
    assert ledger["x"]["winner"]["param_version"] == 7
    assert ledger["x"]["adjudication"]["rule"] == "param_version_desc"
    assert ledger["x"]["candidate_count"] == 2
