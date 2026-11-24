from __future__ import annotations

import json
from pathlib import Path

import pytest

from gsb_ledger.config import ConfigValidation, load_config, load_requests, validate_project
from gsb_ledger.receipts import load_receipts, parse_receipt_text


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample"


def _write_config(tmp_path: Path, mutate) -> Path:
    raw = json.loads((SAMPLE / "config.json").read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_sample_config_and_requests_validate():
    config, requests = validate_project(SAMPLE / "config.json", SAMPLE / "requests.ndjson")
    assert len(config.workers) == 3
    assert [request.seq for request in requests] == [1, 2, 3, 4, 5, 6]
    assert config.batch_size == 3


def test_invalid_worker_identity_port_and_dimensions(tmp_path):
    def duplicate(raw):
        raw["workers"].append(dict(raw["workers"][0]))

    with pytest.raises(ConfigValidation) as error:
        load_config(_write_config(tmp_path, duplicate))
    assert any("duplicate worker_id" in issue.message for issue in error.value.issues)

    def bad_port(raw):
        raw["workers"][0]["port"] = 19000

    with pytest.raises(ConfigValidation) as error:
        load_config(_write_config(tmp_path, bad_port))
    assert any("outside allowed range" in issue.message for issue in error.value.issues)

    def zero_dimension(raw):
        raw["workers"][0]["model_dim"] = 0

    with pytest.raises(ConfigValidation) as error:
        load_config(_write_config(tmp_path, zero_dimension))
    assert any(issue.item.endswith("model_dim") for issue in error.value.issues)


def test_merge_rule_must_reference_known_worker_version(tmp_path):
    def invalid(raw):
        raw["merge_rule"]["entries"].append(
            {"worker_id": "w-alpha", "param_version": 777}
        )

    with pytest.raises(ConfigValidation) as error:
        load_config(_write_config(tmp_path, invalid))
    assert any(issue.item.startswith("merge_rule.entries") for issue in error.value.issues)


def test_request_sequence_arrival_and_batch_size_validation(tmp_path):
    request_path = tmp_path / "requests.ndjson"
    request_path.write_text(
        "\n".join(
            [
                json.dumps({"seq": 2, "request_id": "a", "arrival_ms": 0}),
                json.dumps({"seq": 2, "request_id": "b", "arrival_ms": -1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidation) as error:
        load_requests(request_path, batch_size=3)
    messages = " ".join(issue.message for issue in error.value.issues)
    assert "increase strictly" in messages
    assert "arrival_ms" in messages
    assert "exceeds request count" in messages


def test_three_version_parsers_keep_explicit_missing_markers():
    backend = {
        "batch_id": "b-001",
        "request_id": "r-001",
        "output_digest": "sha256:" + "1" * 64,
    }
    r1 = parse_receipt_text(json.dumps(backend), version="r1")
    assert r1.receipt is not None
    assert set(r1.receipt.missing_fields) == {
        "param_version",
        "duration_ms",
        "worker_id",
        "rank",
        "covered_refs",
        "param_digest",
    }
    r2 = parse_receipt_text(
        json.dumps(
            {
                **backend,
                "param_version": 4,
                "duration_ms": 9,
                "worker_id": "w-alpha",
            }
        ),
        version="r2",
    )
    assert r2.receipt is not None
    assert r2.receipt.version == "r2"
    assert set(r2.receipt.missing_fields) == {"rank", "covered_refs", "param_digest"}
    r3 = parse_receipt_text(
        json.dumps(
            {
                **backend,
                "param_version": 4,
                "duration_ms": 9,
                "worker_id": "w-alpha",
                "rank": 0,
                "covered_refs": [],
                "param_digest": "sha256:" + "2" * 64,
            }
        ),
        version="r3",
    )
    assert r3.receipt is not None
    assert r3.receipt.missing_fields == ()
    assert r3.receipt.completeness == "complete"


@pytest.mark.parametrize(
    "line, detail",
    [
        (
            json.dumps(
                {
                    "batch_id": "b-001",
                    "request_id": "r-001",
                    "output_digest": "sha256:short",
                    "param_version": 4,
                    "duration_ms": 9,
                    "worker_id": "w-alpha",
                }
            ),
            "FIELD_WIDTH",
        ),
        (
            '{"batch_id":"b-001","request_id":"r-001",'
            '"output_digest":"sha256:' + "1" * 64 + '",'
            '"param_version":4,"duration_ms":NaN,"worker_id":"w-alpha"}',
            "NON_FINITE",
        ),
    ],
)
def test_malformed_receipts_report_receipt_malformed(line, detail):
    outcome = parse_receipt_text(line, version="r2")
    assert outcome.receipt is None
    assert [issue.code for issue in outcome.issues] == ["E_RECEIPT_MALFORMED"]
    assert any(issue.detail_code == detail for issue in outcome.issues)


def test_loader_continues_after_bad_row(tmp_path):
    receipt = (
        json.dumps(
            {
                "batch_id": "b-001",
                "request_id": "r-001",
                "output_digest": "sha256:" + "1" * 64,
                "param_version": 3,
                "duration_ms": 5,
                "worker_id": "w-beta",
            }
        )
        + "\n"
    )
    bad = receipt.replace('"duration_ms": 5', '"duration_ms": Infinity')
    path = tmp_path / "w-beta.r2.ndjson"
    path.write_text(bad + receipt.replace("r-001", "r-002"), encoding="utf-8")
    config = load_config(SAMPLE / "config.json")
    config = config.__class__(
        **{
            **config.__dict__,
            "receipt_sources": (
                config.receipt_sources[1].__class__(
                    path=str(path),
                    default_worker_id="w-beta",
                    default_param_version=3,
                    format="r2",
                ),
            ),
        }
    )
    result = load_receipts(config, project_root=tmp_path)
    assert len(result.receipts) == 1
    assert result.receipts[0].request_id == "r-002"
    assert any(issue.code == "E_RECEIPT_MALFORMED" for issue in result.issues)
