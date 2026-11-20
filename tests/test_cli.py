from __future__ import annotations

import json

from trace_diag.cli import main


def _write_valid_inputs(tmp_path):
    from trace_diag.samplegen import generate_sample_files

    return generate_sample_files(tmp_path / "data", record_count=80)


def _args(paths, tmp_path, *extra):
    return [
        "--services",
        str(paths["services"]),
        "--policies",
        str(paths["policies"]),
        "--db",
        str(tmp_path / "data" / "ledger" / "ledger.sqlite"),
        "--log-dir",
        str(tmp_path / "data" / "logs"),
        *extra,
    ]


def test_cli_sample_injection_reports_spec_error(tmp_path, capsys):
    paths = _write_valid_inputs(tmp_path)
    code = main(["analyze", *_args(paths, tmp_path, "--inject", "sample")])
    assert code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "E_SPEC"
    log = (tmp_path / "data" / "logs" / "trace_diag.jsonl").read_text(encoding="utf-8")
    assert '"code":"E_SPEC"' in log


def test_cli_policy_ratio_injection_reports_bucket_error(tmp_path, capsys):
    paths = _write_valid_inputs(tmp_path)
    code = main(["analyze", *_args(paths, tmp_path, "--inject", "policy-ratio")])
    assert code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "E_BUCKET"
    assert output["error"]["path"] == "$.policies[0].head_ratio_bp"


def test_cli_snapshot_injection_reports_snapshot_error(tmp_path, capsys):
    paths = _write_valid_inputs(tmp_path)
    assert main(["analyze", *_args(paths, tmp_path)]) == 0
    capsys.readouterr()
    code = main(["analyze", *_args(paths, tmp_path, "--inject", "snapshot")])
    assert code == 4
    output = json.loads(capsys.readouterr().out)
    assert output["error"]["code"] == "E_SNAPSHOT"
    assert main(["verify", "--db", str(tmp_path / "data" / "ledger" / "ledger.sqlite"), "--log-dir", str(tmp_path / "data" / "logs")]) == 0


def test_cli_generate_and_verify(tmp_path, capsys):
    data_dir = tmp_path / "generated"
    assert main(["generate-samples", "--data-dir", str(data_dir), "--records", "72"]) == 0
    capsys.readouterr()
    db = tmp_path / "ledger.sqlite"
    service_path = data_dir / "services.json"
    policy_path = data_dir / "policies.json"
    assert (
        main(
            [
                "analyze",
                "--services",
                str(service_path),
                "--policies",
                str(policy_path),
                "--db",
                str(db),
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "verify",
                "--db",
                str(db),
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
