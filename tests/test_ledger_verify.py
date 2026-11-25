import json
import sqlite3

import pytest

from lease_middleware.engine import replay_bundle
from lease_middleware.errors import SnapshotError, VerifyError
from lease_middleware.ledger import Ledger, load_valid_snapshot, write_snapshot
from lease_middleware.verify import verify_ledger


def test_ledger_reopen_and_digest_tamper(tmp_path, bundle):
    result = replay_bundle(bundle)
    ledger_path = tmp_path / "ledger.sqlite"
    ledger = Ledger(ledger_path)
    ledger.append_records(result["records"])
    digest = ledger.content_digest
    ledger.close()

    reopened = Ledger.open(ledger_path)
    assert reopened.content_digest == digest
    report = verify_ledger(bundle, reopened)
    assert report["verified_records"] == len(result["records"])
    reopened.close()

    connection = sqlite3.connect(ledger_path)
    with connection:
        connection.execute(
            "UPDATE ledger_meta SET value=? WHERE key='content_digest'",
            ("sha256:" + "1" * 64,),
        )
    connection.close()
    with pytest.raises(SnapshotError) as exc:
        Ledger.open(ledger_path)
    assert exc.value.code == "E_SNAPSHOT"


def test_snapshot_version_or_digest_failure_replaces(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    write_snapshot(snapshot, {"answer": 42})
    assert load_valid_snapshot(snapshot) == {"answer": 42}

    doc = json.loads(snapshot.read_text())
    doc["format_version"] = 999
    snapshot.write_text(json.dumps(doc))
    with pytest.raises(SnapshotError):
        load_valid_snapshot(snapshot)


def test_verify_detects_first_extra_record(tmp_path, bundle):
    ledger = Ledger(tmp_path / "wrong.sqlite")
    ledger.append_records(replay_bundle(bundle)["records"][:3])
    with pytest.raises(VerifyError) as exc:
        verify_ledger(bundle, ledger)
    assert exc.value.code == "E_VERIFY"
    assert exc.value.field.startswith("entries[3]")
    ledger.close()
