from __future__ import annotations

import copy
from pathlib import Path

import pytest

from inference_attribution.errors import DiagnosticError
from inference_attribution.jsonio import canonical_json, parse_json_text
from inference_attribution.planner import build_plan
from inference_attribution.receipts import (
    RawReceipt,
    encode_receipts,
    normalize_receipts,
    read_receipt_file,
    write_receipt_file,
)
from inference_attribution.spec import (
    load_merge_rules,
    load_requests,
    load_workers,
    parse_merge_rules,
    parse_requests,
    parse_workers,
)
from helpers import SAMPLES, load_sample


def _workers():
    return load_workers(SAMPLES / "workers.json")


def _requests(workers):
    return load_requests(SAMPLES / "requests.json", workers)


def _rules():
    return load_merge_rules(SAMPLES / "merge_rules.json")


def test_plan_is_byte_stable_across_input_order():
    workers = _workers()
    requests = _requests(workers)
    plan_a = build_plan(workers, requests)

    reversed_workers_doc = copy.deepcopy(load_sample("workers.json"))
    reversed_workers_doc["workers"].reverse()
    reversed_requests_doc = copy.deepcopy(load_sample("requests.json"))
    reversed_requests_doc["requests"].reverse()
    workers_b = parse_workers(reversed_workers_doc, object_name="reordered-workers")
    requests_b = parse_requests(
        reversed_requests_doc, workers_b, object_name="reordered-requests"
    )
    plan_b = build_plan(workers_b, requests_b)
    assert canonical_json(plan_a.to_dict()) == canonical_json(plan_b.to_dict())
    assert len(plan_a.batches) == 3
    assert [len(batch.request_ids) for batch in plan_a.batches] == [2, 2, 2]
    assert len({batch.batch_id for batch in plan_a.batches}) == 3


@pytest.mark.parametrize(
    ("mutate", "object_fragment", "field"),
    [
        (
            lambda doc: doc["requests"][1].__setitem__(
                "request_id", doc["requests"][0]["request_id"]
            ),
            "requests[1]",
            "request_id",
        ),
        (
            lambda doc: doc["requests"][1].__setitem__(
                "arrival_seq", doc["requests"][0]["arrival_seq"]
            ),
            "requests[1]",
            "arrival_seq",
        ),
        (
            lambda doc: doc["requests"][0].__setitem__("input", [0.0, 1.0]),
            "requests[0]",
            "input",
        ),
    ],
)
def test_request_spec_errors_identify_first_object_and_field(
    mutate, object_fragment, field
):
    workers = _workers()
    document = copy.deepcopy(load_sample("requests.json"))
    mutate(document)
    with pytest.raises(DiagnosticError) as caught:
        parse_requests(document, workers, object_name="req.json")
    error = caught.value
    assert error.code == "E_SPEC"
    assert object_fragment in str(error.object)
    assert error.field == field


@pytest.mark.parametrize(
    ("mutate", "worker_index", "field"),
    [
        (lambda worker: worker.__setitem__("host", "localhost"), 0, "host"),
        (lambda worker: worker.__setitem__("hidden_dim", 0), 0, "hidden_dim"),
        (lambda worker: worker.__setitem__("seed", -1), 0, "seed"),
        (
            lambda worker: worker.__setitem__("port", 28431),
            1,
            "port",
        ),
    ],
)
def test_worker_spec_errors_identify_first_object_and_field(
    mutate, worker_index, field
):
    document = copy.deepcopy(load_sample("workers.json"))
    mutate(document["workers"][worker_index])
    with pytest.raises(DiagnosticError) as caught:
        parse_workers(document, object_name="workers.json")
    error = caught.value
    assert error.code == "E_SPEC"
    assert f"workers[{worker_index}]" in str(error.object)
    assert error.field == field


def test_duplicate_json_keys_are_rejected():
    text = '{"schema_version":1,"schema_version":1,"requests":[]}'
    with pytest.raises(DiagnosticError) as caught:
        parse_json_text(text, object_name="duplicate.json")
    assert caught.value.code == "E_SPEC"
    assert caught.value.field == "schema_version"


def test_merge_rule_strategy_is_strict():
    document = copy.deepcopy(load_sample("merge_rules.json"))
    document["formats"]["r1"]["fill"]["worker_id"] = "guessed.worker"
    with pytest.raises(DiagnosticError) as caught:
        parse_merge_rules(document, object_name="rules.json")
    assert caught.value.code == "E_SPEC"
    assert caught.value.field == "fill.worker_id"


def test_three_receipt_formats_normalize_into_one_ledger():
    workers = _workers()
    requests = _requests(workers)
    plan = build_plan(workers, requests)
    raw = {
        "r1": read_receipt_file(SAMPLES / "receipts_r1.jsonl", "r1"),
        "r2": read_receipt_file(SAMPLES / "receipts_r2.jsonl", "r2"),
        "r3": read_receipt_file(SAMPLES / "receipts_r3.jsonl", "r3"),
    }
    ledger = normalize_receipts(
        raw,
        request_set=requests,
        worker_set=workers,
        plan=plan,
        rules=_rules(),
    )
    assert len(ledger.batches) == 3
    assert {
        tuple(batch["source_formats"]) for batch in ledger.batches
    } == {("r1",), ("r2",), ("r3",)}
    r1_receipt = ledger.batches[0]["receipts"][0]
    assert r1_receipt["source_format"] == "r1"
    assert r1_receipt["filled_fields"] == [
        "latency_ms",
        "param_digest",
        "param_version_seq",
        "rank_in_batch",
        "superseded_request_ids",
        "worker_id",
    ]
    assert r1_receipt["fill_basis"]["latency_ms"] == "constant:0"
    assert len(ledger.batches[0]["attribution_digest"]) == 64


def _normalize_single_batch(format_version: str, rows: list[RawReceipt]):
    workers = _workers()
    requests = _requests(workers)
    plan = build_plan(workers, requests)
    return normalize_receipts(
        {format_version: rows},
        request_set=requests,
        worker_set=workers,
        plan=plan,
        rules=_rules(),
    )


def _raw_r2():
    rows = read_receipt_file(SAMPLES / "receipts_r2.jsonl", "r2")
    return RawReceipt(
        format_version=rows[0].format_version,
        line_number=rows[0].line_number,
        source=rows[0].source,
        value=dict(rows[0].value),
    )


def test_negative_latency_reports_line_and_field():
    row = _raw_r2()
    row.value["latency_ms"] = -1
    with pytest.raises(DiagnosticError) as caught:
        _normalize_single_batch("r2", [row])
    assert caught.value.code == "E_FIELD"
    assert caught.value.line == 1
    assert caught.value.field == "latency_ms"


def test_duplicate_rank_reports_line_and_field():
    rows = read_receipt_file(SAMPLES / "receipts_r3.jsonl", "r3")
    rows = [
        RawReceipt(
            format_version=row.format_version,
            line_number=row.line_number,
            source=row.source,
            value=dict(row.value),
        )
        for row in rows
    ]
    rows[1].value["rank_in_batch"] = rows[0].value["rank_in_batch"]
    with pytest.raises(DiagnosticError) as caught:
        _normalize_single_batch("r3", rows)
    assert caught.value.code == "E_FIELD"
    assert caught.value.line == 2
    assert caught.value.field == "rank_in_batch"


def test_out_of_bounds_rank_reports_line_and_field():
    row = _raw_r2()
    row.value["rank_in_batch"] = 99
    with pytest.raises(DiagnosticError) as caught:
        _normalize_single_batch("r2", [row])
    assert caught.value.code == "E_FIELD"
    assert caught.value.field == "rank_in_batch"


def test_unknown_duplicate_delivery_reference_is_e_ref():
    rows = read_receipt_file(SAMPLES / "receipts_r3.jsonl", "r3")
    rows = [
        RawReceipt(
            format_version=row.format_version,
            line_number=row.line_number,
            source=row.source,
            value=dict(row.value),
        )
        for row in rows
    ]
    rows[1].value["superseded_request_ids"] = ["req-999"]
    with pytest.raises(DiagnosticError) as caught:
        _normalize_single_batch("r3", rows)
    assert caught.value.code == "E_REF"
    assert caught.value.line == 2
    assert caught.value.field == "superseded_request_ids"


def test_missing_receipt_field_is_e_field():
    row = _raw_r2()
    row.value.pop("worker_id")
    with pytest.raises(DiagnosticError) as caught:
        _normalize_single_batch("r2", [row])
    assert caught.value.code == "E_FIELD"
    assert caught.value.field == "worker_id"


def test_receipt_encoder_round_trips_all_three_formats(tmp_path: Path):
    workers = _workers()
    requests = _requests(workers)
    plan = build_plan(workers, requests)
    r3_rows = read_receipt_file(SAMPLES / "receipts_r3.jsonl", "r3")
    complete = [dict(row.value) for row in r3_rows]
    expected = normalize_receipts(
        {"r3": r3_rows},
        request_set=requests,
        worker_set=workers,
        plan=plan,
        rules=_rules(),
    ).batches[0]["attribution_digest"]
    for format_version in ("r1", "r2", "r3"):
        path = tmp_path / f"roundtrip-{format_version}.jsonl"
        write_receipt_file(path, complete, format_version)
        decoded = read_receipt_file(path, format_version)
        normalized = normalize_receipts(
            {format_version: decoded},
            request_set=requests,
            worker_set=workers,
            plan=plan,
            rules=_rules(),
        )
        assert normalized.batches[0]["attribution_digest"] == expected
        assert encode_receipts(complete, format_version)[0].keys() <= {
            "batch_id",
            "request_id",
            "output_digest",
            "param_version_seq",
            "latency_ms",
            "worker_id",
            "rank_in_batch",
            "superseded_request_ids",
            "param_digest",
        }
