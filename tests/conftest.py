from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from walcrash.engine import ReplayEngine
from walcrash.storage import Storage
from walcrash.validation import load_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT_ROOT / "samples" / "sample.json"
ALT_SAMPLE = PROJECT_ROOT / "samples" / "alt_pages.json"


@pytest.fixture(autouse=True)
def project_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(PROJECT_ROOT)


def make_sample_variant(
    tmp_path: Path,
    *,
    budget: dict[str, int] | None = None,
    mutations: dict[int, dict[str, Any]] | None = None,
    source: Path = SAMPLE,
) -> Path:
    document = json.loads(source.read_text(encoding="utf-8"))
    log_path = (source.parent / document["log_file"]).resolve()
    document["log_file"] = str(log_path)
    if budget is not None:
        document["budget"] = budget
    target = tmp_path / f"{source.stem}_variant.json"
    if mutations:
        wal_path = tmp_path / "mutated.ndjson"
        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for lsn, changes in mutations.items():
            index = next(i for i, record in enumerate(records) if record["lsn"] == lsn)
            records[index].update(changes)
        wal_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )
        document["log_file"] = str(wal_path)
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target


def make_storage(tmp_path: Path, source: Path = SAMPLE, name: str = "workspace") -> Storage:
    storage = Storage(tmp_path / name)
    storage.load_fresh(load_sample(source))
    return storage


def make_replayed(tmp_path: Path, source: Path = SAMPLE, name: str = "workspace", delay_ms: int = 0) -> Storage:
    storage = make_storage(tmp_path, source=source, name=name)
    ReplayEngine(storage).replay(delay_ms=delay_ms)
    return storage


def run_cli(
    *args: str,
    cwd: Path = PROJECT_ROOT,
    expect: int | None = None,
    timeout: int = 60,
) -> Any:
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "walcrash", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if expect is not None and result.returncode != expect:
        raise AssertionError(
            f"expected rc={expect}, got {result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def max_overlap(tasks: list[dict[str, Any]], txid: str | None = None) -> int:
    events: list[tuple[int, int]] = []
    for task in tasks:
        if txid is not None and task["txid"] != txid:
            continue
        start = task.get("started_ms")
        if start is None:
            continue
        finish = task.get("finished_ms") or start
        events.append((start, 1))
        events.append((finish, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    maximum = 0
    for _, delta in events:
        current += delta
        maximum = max(maximum, current)
    return maximum

