from __future__ import annotations

import hashlib
import json
from pathlib import Path

from walcrash.content import deterministic_page_bytes, digest_bytes, digest_file
from walcrash.verification import verify_replay
from walcrash.validation import load_sample

from .conftest import make_replayed, make_sample_variant, make_storage
from walcrash.engine import ReplayEngine


def _wal_page_snapshot(storage) -> dict[int, tuple[str, int, str, bool, bytes]]:
    result = {}
    for row in storage.page_versions():
        if row["kind"] != "wal":
            continue
        result[row["lsn"]] = (
            row["page_id"],
            row["version"],
            row["digest"],
            bool(row["visible"]),
            storage.path_for(row["content_path"]).read_bytes(),
        )
    return result


def test_single_thread_page_versions_visibility_and_checkpoints(tmp_path: Path) -> None:
    storage = make_replayed(tmp_path)
    status = storage.status()
    assert status["cursor"] == 12
    assert status["crash_point_lsn"] == 10
    assert status["counts"]["page_versions"] == 12
    assert status["counts"]["invisible_page_versions"] == 2
    assert status["counts"]["checkpoints"] == 3
    assert status["counts"]["batches"] == 3

    versions = {
        row["lsn"]: (row["page_id"], row["version"], row["visible"])
        for row in storage.page_versions()
        if row["kind"] == "wal"
    }
    assert versions[11] == ("p1", 6, False)
    assert versions[12] == ("p3", 3, False)
    page_versions = {page["page_id"]: page["visible_version"] for page in status["pages"]}
    assert page_versions == {"p1": 5, "p2": 7, "p3": 2}

    per_page = {}
    for row in storage.page_versions():
        per_page.setdefault(row["page_id"], []).append(row["version"])
    for page_id, sequence in per_page.items():
        assert sequence == sorted(sequence), page_id
        assert len(sequence) == len(set(sequence)), page_id

    for row in storage.page_versions():
        if row["kind"] == "wal":
            assert digest_file(storage.path_for(row["content_path"])) == row["digest"]

    checkpoint_cursors = [row["ledger_cursor"] for row in storage.checkpoint_rows()]
    assert checkpoint_cursors == [4, 8, 12]
    assert verify_replay(storage)["ok"]
    assert storage.verify_ledger()["ok"]


def test_parallel_and_single_slot_replays_are_byte_identical(tmp_path: Path) -> None:
    parallel = make_replayed(tmp_path, name="parallel", delay_ms=25)
    single_sample = make_sample_variant(
        tmp_path,
        budget={
            "global_slots": 1,
            "per_tx_quota": 1,
            "page_byte_budget": 512,
            "checkpoint_interval": 4,
        },
    )
    single = make_storage(tmp_path, source=single_sample, name="single")
    ReplayEngine(single).replay(delay_ms=3)
    parallel_snapshot = _wal_page_snapshot(parallel)
    single_snapshot = _wal_page_snapshot(single)
    assert parallel_snapshot == single_snapshot
    assert [(p["page_id"], p["visible_version"], p["visible_digest"]) for p in parallel.status()["pages"]] == [
        (p["page_id"], p["visible_version"], p["visible_digest"]) for p in single.status()["pages"]
    ]
    assert verify_replay(single)["ok"]


def test_sqlite_versions_and_files_are_deterministic_across_workspaces(tmp_path: Path) -> None:
    first = make_replayed(tmp_path, name="one")
    second = make_replayed(tmp_path, name="two")
    assert _wal_page_snapshot(first) == _wal_page_snapshot(second)
    assert hashlib.sha256(
        "".join(row["digest"] for row in first.page_versions() if row["kind"] == "wal").encode()
    ).hexdigest() == hashlib.sha256(
        "".join(row["digest"] for row in second.page_versions() if row["kind"] == "wal").encode()
    ).hexdigest()


def test_wal_may_start_at_a_positive_non_one_lsn(tmp_path: Path) -> None:
    source = Path("samples/sample.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (source.parent / document["log_file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pages = {page["page_id"]: page for page in document["pages"]}
    versions = {page_id: page["initial_version"] for page_id, page in pages.items()}
    offset = 9
    shifted = []
    for record in records:
        record = dict(record)
        record["lsn"] += offset
        page = pages[record["page_id"]]
        versions[record["page_id"]] += 1
        record["content_digest"] = digest_bytes(
            deterministic_page_bytes(
                record["page_id"],
                page["size_bytes"],
                versions[record["page_id"]],
                record["lsn"],
            )
        )
        shifted.append(record)
    wal_path = tmp_path / "shifted.ndjson"
    wal_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in shifted),
        encoding="utf-8",
        newline="\n",
    )
    document["log_file"] = str(wal_path)
    sample_path = tmp_path / "shifted.json"
    sample_path.write_text(json.dumps(document), encoding="utf-8")
    sample = load_sample(sample_path)
    assert sample.initial_cursor == offset
    assert sample.crash_point_lsn == 10 + offset
    storage = make_storage(tmp_path, source=sample_path, name="shifted-workspace")
    assert storage.ledger_cursor() == offset
    result = ReplayEngine(storage).replay()
    assert result["cursor_start"] == offset
    assert result["cursor_end"] == 12 + offset
    assert verify_replay(storage)["ok"]
