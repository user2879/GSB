import json
import subprocess
import sys


def _cli(args, cwd=None):
    return subprocess.run([sys.executable, "-m", "kvdiag"] + args,
                          capture_output=True, text=True, cwd=cwd)


def _read_log(path):
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_inject_sample_corrupt_reports_e_spec(tree, tmp_path):
    log = tmp_path / "inj.log.jsonl"
    out = _cli(["--log", str(log), "inject", "sample-corrupt",
                "--spec", str(tree["spec"]), "--shard", "shard-1"])
    assert out.returncode == 0
    assert json.loads(out.stdout)["expects"] == "E_SPEC"

    run = _cli(["--log", str(log), "run", "--spec", str(tree["spec"]),
                "--db", str(tmp_path / "inj1.sqlite"),
                "--renewals", str(tree["renewals"])])
    assert run.returncode == 1
    err = json.loads(run.stdout)
    assert err["code"] == "E_SPEC"
    assert err["shard_id"] == "shard-1"
    records = _read_log(log)
    assert any(r["code"] == "E_SPEC" and r["shard_id"] == "shard-1"
               and "ts_mono_ms" in r for r in records)


def test_inject_lease_version_regression_reports_e_lease(tree, tmp_path):
    log = tmp_path / "inj.log.jsonl"
    out = _cli(["--log", str(log), "inject", "lease-version-regression",
                "--renewals", str(tree["renewals"])])
    assert out.returncode == 0
    assert json.loads(out.stdout)["expects"] == "E_LEASE"

    run = _cli(["--log", str(log), "run", "--spec", str(tree["spec"]),
                "--db", str(tmp_path / "inj2.sqlite"),
                "--renewals", str(tree["renewals"])])
    assert run.returncode == 1
    err = json.loads(run.stdout)
    assert err["code"] == "E_LEASE"
    assert err["details"]["expected_version"] == 0
    records = _read_log(log)
    assert any(r["code"] == "E_LEASE" and r["shard_id"] is not None
               and "ts_mono_ms" in r for r in records)


def test_inject_ledger_digest_mismatch_reports_e_snapshot(tree, tmp_path):
    log = tmp_path / "inj.log.jsonl"
    db = tmp_path / "inj3.sqlite"
    session = tmp_path / "session.json"
    run = _cli(["--log", str(log), "run", "--spec", str(tree["spec"]),
                "--db", str(db), "--renewals", str(tree["renewals"]),
                "--session-out", str(session)])
    assert run.returncode == 0

    out = _cli(["--log", str(log), "inject", "ledger-digest-mismatch",
                "--session", str(session)])
    assert out.returncode == 0
    assert json.loads(out.stdout)["expects"] == "E_SNAPSHOT"

    rec = _cli(["--log", str(log), "recover", "--spec", str(tree["spec"]),
                "--db", str(db), "--session", str(session)])
    assert rec.returncode == 0
    report = json.loads(rec.stdout)
    assert report["replaced"] is True
    records = _read_log(log)
    assert any(r["code"] == "E_SNAPSHOT" and "ts_mono_ms" in r
               for r in records)
