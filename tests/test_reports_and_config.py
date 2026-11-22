from __future__ import annotations

from corpus_workbench.codes import E_CONFIG_INVALID
from corpus_workbench.config import ConfigValidationError, load_balance_config
from corpus_workbench.demo_data import generate_demo_workspace
from corpus_workbench.loader import load_documents
from corpus_workbench.pipeline import Workbench


def test_report_contains_injection_and_audit_fields(tmp_path):
    generate_demo_workspace(tmp_path)
    workbench = Workbench(
        data_path=tmp_path / "data" / "corpus",
        balance_path=tmp_path / "configs" / "balance.json",
        dedup_path=tmp_path / "configs" / "dedup.json",
        scheduler_path=tmp_path / "configs" / "scheduler.json",
        injection_path=tmp_path / "configs" / "injection.json",
        output_root=tmp_path / "out",
        run_id="report",
    )
    report = workbench.run(reset=True)

    assert report["status"] == "complete"
    assert report["injection"]["configured_count"] == 4
    assert report["injection"]["matched_count"] == 4
    observed = {
        item["injection_id"]: item["observed_code"] for item in report["injection"]["items"]
    }
    assert observed == {
        "inj-empty-text": "E_ENCODING_INVALID",
        "inj-non-utf8": "E_ENCODING_INVALID",
        "inj-unknown-source": "E_SOURCE_UNKNOWN",
        "inj-duplicate-doc-id": "E_DUP_DOC_ID",
    }
    assert report["sampling"]["sample_count"] == 42
    assert report["documents"]["sampled_unique_before_dedup"] >= report["documents"]["after_dedup"]
    assert report["dedup"]["duplicate_group_count"] >= 1
    assert report["budget"]["consumed_points"] > 0
    assert report["pending_queue"] == []
    assert "load_validate" in report["stage_timings_ms"]
    assert "sample" in report["stage_timings_ms"]
    assert "minhash" in report["stage_timings_ms"]
    assert "dedup" in report["stage_timings_ms"]
    assert "write" in report["stage_timings_ms"]


def test_invalid_config_maps_to_config_invalid(tmp_path):
    generate_demo_workspace(tmp_path)
    path = tmp_path / "configs" / "bad_balance.json"
    path.write_text('{"source_weights":{"books":1},"target_sample_count":0,"sampling_salt":"x"}', encoding="utf-8")
    try:
        load_balance_config(path)
    except ConfigValidationError as exc:
        assert [failure.code for failure in exc.failures] == [E_CONFIG_INVALID]
    else:
        raise AssertionError("invalid config was accepted")

    workbench = Workbench(
        data_path=tmp_path / "data" / "corpus",
        balance_path=path,
        dedup_path=tmp_path / "configs" / "dedup.json",
        scheduler_path=tmp_path / "configs" / "scheduler.json",
        injection_path=tmp_path / "configs" / "injection.json",
        output_root=tmp_path / "out-config",
        run_id="bad-config",
    )
    report = workbench.run(reset=True)
    assert report["status"] == "config_invalid"
    assert report["failures"][0]["code"] == E_CONFIG_INVALID


def test_missing_corpus_and_unpaired_unicode_are_rejected(tmp_path):
    generate_demo_workspace(tmp_path)
    bad_line = '{"source":"books","doc_id":"surrogate","text":"\\ud800","metadata":{}}\n'
    (tmp_path / "data" / "corpus" / "bad_unicode.jsonl").write_text(bad_line, encoding="utf-8")
    balance = load_balance_config(tmp_path / "configs" / "balance.json")
    loaded = load_documents(tmp_path / "data" / "corpus", balance.source_weights)
    assert any(
        failure.code == "E_ENCODING_INVALID" and failure.doc_id == "surrogate"
        for failure in loaded.failures
    )

    missing = Workbench(
        data_path=tmp_path / "does-not-exist",
        balance_path=tmp_path / "configs" / "balance.json",
        dedup_path=tmp_path / "configs" / "dedup.json",
        scheduler_path=tmp_path / "configs" / "scheduler.json",
        injection_path=tmp_path / "configs" / "injection.json",
        output_root=tmp_path / "out-missing",
        run_id="missing",
    ).run(reset=True)
    assert missing["status"] == "input_invalid"
    assert missing["complete"] is False
    assert missing["failures"][0]["code"] == E_CONFIG_INVALID
